from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Call:
    module: nn.Module
    args: tuple
    kwargs: dict
    output: Any
    position: int
    length: int


@dataclass(frozen=True)
class Probe:
    """A hook with an adapter-supplied interpretation of its inputs and output."""

    name: str
    module: nn.Module
    read: Callable[[Call], Mapping[str, Any] | None]
    kind: str = "custom"


@dataclass(frozen=True)
class Layer:
    """A layer whose adapter can extract the selected input and output vectors."""

    name: str
    module: nn.Module
    read: Callable[[Call], tuple[Tensor | None, Tensor | None]]


@dataclass(frozen=True)
class Plan:
    layers: Sequence[tuple[str, nn.Module] | Layer]
    probes: Sequence[Probe] = field(default_factory=tuple)


class TracePoint(nn.Module):
    """Identity operation for observing a value and metadata inside a runtime."""

    def forward(self, value: Any, **metadata: Any) -> Any:
        return value


def token_row(value: Any, position: int, length: int) -> Tensor | None:
    """Read from an explicitly batch-first [1, sequence, features] tensor."""
    if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[:2] != (1, length):
        return None
    return value[0, position]


def packed_token_row(
    value: Any, token_positions: Any, position: int, length: int,
) -> Tensor | None:
    """Read one packed row using explicit zero-based positions in the input sequence.

    A position of -1 marks padding. For duplicated dispatches, narrow the map to
    one expert or slot before calling this helper.
    """
    if not isinstance(value, Tensor) or value.ndim != 2:
        return None
    if (not isinstance(token_positions, Tensor) or token_positions.ndim != 1
            or token_positions.numel() != value.shape[0]
            or token_positions.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)):
        return None
    # Compare in a wide signed type so padding and bounds cannot wrap.
    token_positions = token_positions.to(dtype=torch.int64)
    if not 0 <= position < length or ((token_positions < -1) | (token_positions >= length)).any():
        return None
    matches = (token_positions == position).nonzero()
    if matches.numel() != 1:
        return None
    return value[matches.item()]


def _batch_first_layer(call: Call) -> tuple[Tensor | None, Tensor | None]:
    hidden = call.args[0] if call.args else call.kwargs.get("hidden_states", call.kwargs.get("input"))
    result = call.output[0] if isinstance(call.output, (tuple, list)) and call.output else call.output
    return (token_row(hidden, call.position, call.length),
            token_row(result, call.position, call.length))


def _vector(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64)


def _summary(value: Tensor, full_vectors: bool) -> dict:
    data = _vector(value)
    summary = {"size": data.numel(), "l2": data.norm().item()}
    if full_vectors:
        summary["values"] = data.tolist()
    return summary


def _feature_vector(value: Any) -> Tensor | None:
    return value if isinstance(value, Tensor) and value.ndim == 1 else None


def linear_terms(
    signal: Tensor,
    weight: Tensor,
    output_index: int,
    *,
    bias: Tensor | None = None,
    measured: Tensor | None = None,
    top: int = 5,
    input_index: int | None = None,
    all_terms: bool = False,
) -> dict:
    """Decompose a scalar affine output. Weight axes are [output, input]."""
    if signal.ndim != 1 or weight.ndim != 2 or weight.shape[1] != signal.numel():
        raise ValueError("expected a one-dimensional input and [output, input] weights")
    if top < 0 or not 0 <= output_index < weight.shape[0]:
        raise ValueError("invalid top count or output index")
    if input_index is not None and not 0 <= input_index < signal.numel():
        raise ValueError("input index out of range")
    data = _vector(signal)
    row = _vector(weight[output_index])
    terms = data * row
    constant = 0.0 if bias is None else bias[output_index].item()
    reconstructed = terms.sum().item() + constant

    def term(index):
        return {"input": index, "activation": data[index].item(),
                "weight": row[index].item(), "product": terms[index].item()}

    if all_terms:
        indices = range(terms.numel())
    else:
        indices = torch.argsort(terms.abs(), descending=True, stable=True)[:top].tolist()
    result = {"output": output_index, "terms": [term(i) for i in indices],
              "bias": constant, "reconstructed": reconstructed}
    if input_index is not None:
        result["selected_term"] = term(input_index)
    if measured is not None:
        value = measured[output_index].item()
        result.update(measured=value, measured_minus_reconstructed=value - reconstructed)
    return result


def trace(
    model: nn.Module, plan: Plan, input_ids: Tensor, position: int, **options: Any,
) -> dict:
    """Run one forward pass and report one selected token."""
    return trace_many(model, plan, input_ids, [position], **options)["traces"][0]


def trace_many(
    model: nn.Module,
    plan: Plan,
    input_ids: Tensor,
    positions: Sequence[int] | None = None,
    *,
    full_vectors: bool = False,
    forward_kwargs: Mapping[str, Any] | None = None,
    runner: Callable[[nn.Module, Tensor, Mapping[str, Any]], Any] | None = None,
    logits_reader: Callable[[Any, int, int], Tensor | None] | None = None,
    on_layer: Callable[[str, list[dict]], None] | None = None,
) -> dict:
    """Observe multiple positions during one forward pass.

    A custom Layer, runner, or logits_reader can map nonstandard runtime layouts.
    The synchronous on_layer callback runs once per layer invocation. The
    event_index orders layer and probe hooks by observed completion.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, sequence] with at least one token")
    length = input_ids.shape[1]
    wanted = tuple(range(length) if positions is None else
                   (position if position >= 0 else length + position for position in positions))
    if not wanted or any(not 0 <= position < length for position in wanted):
        raise ValueError("token position out of range or selection empty")
    if len(set(wanted)) != len(wanted):
        raise ValueError("token positions must be unique")
    if not plan.layers:
        raise ValueError("trace plan must list at least one layer")
    layers = [entry if isinstance(entry, Layer) else Layer(*entry, _batch_first_layer)
              for entry in plan.layers]
    if len({id(layer.module) for layer in layers}) != len(layers):
        raise ValueError("list each layer module once; repeated calls are recorded in execution order")
    if len({layer.name for layer in layers}) != len(layers):
        raise ValueError("layer names must be unique")
    if len({probe.name for probe in plan.probes}) != len(plan.probes):
        raise ValueError("probe names must be unique")
    layer_records = {position: [] for position in wanted}
    probe_records = {position: [] for position in wanted}
    handles = []
    events = count()

    def layer_hook(layer):
        def capture(module, args, kwargs, output):
            event = []
            event_index = next(events)
            for position in wanted:
                before, after = layer.read(Call(module, args, kwargs, output, position, length))
                before, after = _feature_vector(before), _feature_vector(after)
                record = {
                    "name": layer.name,
                    "event_index": event_index,
                    "input": None if before is None else _summary(before, full_vectors),
                    "output": None if after is None else _summary(after, full_vectors),
                }
                layer_records[position].append(record)
                if on_layer is not None:
                    event.append({"position": position, "input_l2": record["input"]["l2"] if before is not None else None,
                                  "output_l2": record["output"]["l2"] if after is not None else None})
            if on_layer is not None:
                on_layer(layer.name, event)
        return capture

    def probe_hook(probe):
        def capture(module, args, kwargs, output):
            event_index = next(events)
            for position in wanted:
                record = probe.read(Call(module, args, kwargs, output, position, length))
                probe_records[position].append({"name": probe.name, "kind": probe.kind,
                                                "event_index": event_index, "data": record})
        return capture

    try:
        for layer in layers:
            handles.append(layer.module.register_forward_hook(layer_hook(layer), with_kwargs=True))
        for probe in plan.probes:
            handles.append(probe.module.register_forward_hook(probe_hook(probe), with_kwargs=True))
        with torch.inference_mode():
            options = dict(forward_kwargs or {})
            result = model(input_ids, **options) if runner is None else runner(model, input_ids, options)
    finally:
        for handle in reversed(handles):
            handle.remove()

    reports = []
    for position in wanted:
        token_layers, token_probes = layer_records[position], probe_records[position]
        with torch.inference_mode():
            if logits_reader is None:
                logits = token_row(getattr(result, "logits", None), position, length)
            else:
                logits = _feature_vector(logits_reader(result, position, length))
        executed = {entry["name"] for entry in token_layers}
        observed = {entry["name"] for entry in token_probes}
        report = {
            "schema": "token-trace/1",
            "position": position,
            "token_id": input_ids[0, position].item(),
            "layers": token_layers,
            "probes": token_probes,
            "coverage": {
                "expected_layers": [layer.name for layer in layers],
                "unexecuted_layers": [layer.name for layer in layers if layer.name not in executed],
                "unmapped_layers": [entry["name"] for entry in token_layers if entry["output"] is None],
                "unmapped_layer_inputs": [entry["name"] for entry in token_layers if entry["input"] is None],
                "unexecuted_probes": [probe.name for probe in plan.probes if probe.name not in observed],
                "unmapped_probes": [entry["name"] for entry in token_probes if entry["data"] is None],
            },
        }
        if logits is not None:
            values, ids = logits.detach().float().topk(min(5, logits.numel()))
            report["prediction"] = [{"token_id": i, "logit": v}
                                    for i, v in zip(ids.tolist(), values.tolist())]
        reports.append(report)
    ids = input_ids[0].tolist()
    return {
        "schema": "token-trace/sequence/1",
        "kind": "local",
        "mode": "forward",
        "tokens": [{"position": pos, "id": token_id, "text": str(token_id)}
                   for pos, token_id in enumerate(ids)],
        "layer_names": [layer.name for layer in layers],
        "traces": reports,
    }
