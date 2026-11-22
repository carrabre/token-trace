from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
class Plan:
    layers: Sequence[tuple[str, nn.Module]]
    probes: Sequence[Probe] = field(default_factory=tuple)


def token_row(value: Any, position: int, length: int) -> Tensor | None:
    """Read from an explicitly batch-first [1, sequence, features] tensor."""
    if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[:2] != (1, length):
        return None
    return value[0, position]


def _vector(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64)


def _summary(value: Tensor, full_vectors: bool) -> dict:
    data = _vector(value)
    summary = {"size": data.numel(), "l2": data.norm().item()}
    if full_vectors:
        summary["values"] = data.tolist()
    return summary


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
    model: nn.Module,
    plan: Plan,
    input_ids: Tensor,
    position: int,
    *,
    full_vectors: bool = False,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict:
    """Run one forward pass and collect observations in invocation order."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, sequence] with at least one token")
    length = input_ids.shape[1]
    if position < 0:
        position += length
    if not 0 <= position < length:
        raise ValueError("token position out of range")
    if not plan.layers:
        raise ValueError("trace plan must list at least one layer")
    if len({id(module) for _, module in plan.layers}) != len(plan.layers):
        raise ValueError("list each layer module once; repeated calls are recorded in execution order")
    layer_records, probe_records, handles = [], [], []

    def layer_hook(name):
        def capture(module, args, kwargs, output):
            hidden = args[0] if args else kwargs.get("hidden_states")
            result = output[0] if isinstance(output, (tuple, list)) and output else output
            before = token_row(hidden, position, length)
            after = token_row(result, position, length)
            layer_records.append({
                "name": name,
                "input": None if before is None else _summary(before, full_vectors),
                "output": None if after is None else _summary(after, full_vectors),
            })
        return capture

    def probe_hook(probe):
        def capture(module, args, kwargs, output):
            record = probe.read(Call(module, args, kwargs, output, position, length))
            probe_records.append({"name": probe.name, "kind": probe.kind, "data": record})
        return capture

    try:
        for name, module in plan.layers:
            handles.append(module.register_forward_hook(layer_hook(name), with_kwargs=True))
        for probe in plan.probes:
            handles.append(probe.module.register_forward_hook(probe_hook(probe), with_kwargs=True))
        with torch.inference_mode():
            result = model(input_ids, **dict(forward_kwargs or {}))
    finally:
        for handle in reversed(handles):
            handle.remove()

    logits = getattr(result, "logits", None)
    logits = token_row(logits, position, length)
    executed = {entry["name"] for entry in layer_records}
    observed = {entry["name"] for entry in probe_records}
    report = {
        "schema": "token-trace/1",
        "position": position,
        "token_id": input_ids[0, position].item(),
        "layers": layer_records,
        "probes": probe_records,
        "coverage": {
            "expected_layers": [name for name, _ in plan.layers],
            "unexecuted_layers": [name for name, _ in plan.layers if name not in executed],
            "unmapped_layers": [entry["name"] for entry in layer_records if entry["output"] is None],
            "unexecuted_probes": [probe.name for probe in plan.probes if probe.name not in observed],
            "unmapped_probes": [entry["name"] for entry in probe_records if entry["data"] is None],
        },
    }
    if logits is not None:
        values, ids = logits.detach().float().topk(min(5, logits.numel()))
        report["prediction"] = [{"token_id": i, "logit": v}
                                for i, v in zip(ids.tolist(), values.tolist())]
    return report
