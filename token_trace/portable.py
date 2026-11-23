"""Conservative token-aligned probes for native PyTorch models."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from torch import nn

from .core import Plan, Probe, linear_terms, token_row

# Packed or reordered expert dispatch requires an explicit token-index mapping.
_EXPERT_COMPONENTS = {"expert", "experts", "shared_expert", "shared_experts"}
_STACK_TYPES = (nn.ModuleList, nn.Sequential)


@dataclass(frozen=True)
class Row:
    output: int
    input: int | None = None


def linear_probe(name, module, rows, *, top=5, all_terms=False, transposed=False):
    """Use for a module whose weight is accessible and token axis is preserved."""
    def read(call):
        value = call.args[0] if call.args else call.kwargs.get("input")
        signal = token_row(value, call.position, call.length)
        measured = token_row(call.output, call.position, call.length)
        if signal is None or measured is None:
            return None
        weight = module.weight.T if transposed else module.weight
        return {"rows": [
            linear_terms(signal, weight, row.output, bias=module.bias, measured=measured,
                         input_index=row.input, top=top, all_terms=all_terms)
            for row in rows
        ]}
    return Probe(name, module, read, "affine")


def embedding_probe(name, module, rows):
    for row in rows:
        if row.input is not None or not 0 <= row.output < module.embedding_dim:
            raise ValueError(f"{name}: select an embedding feature as MODULE:FEATURE")

    def read(call):
        ids = call.args[0] if call.args else call.kwargs.get("input")
        measured = token_row(call.output, call.position, call.length)
        if measured is None or getattr(ids, "shape", None) != (1, call.length):
            return None
        index = ids[0, call.position].item()
        return {"lookup_index": index, "features": [
            {"feature": row.output, "weight": module.weight[index, row.output].item(),
             "observed": measured[row.output].item()}
            for row in rows
        ]}

    return Probe(name, module, read, "embedding_lookup")



def discover_stacks(model: nn.Module) -> list[dict]:
    """Find structural candidates. Neither feeding order nor token axes are implied."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, _STACK_TYPES) and len(module):
            candidates.append({"path": name, "layer_count": len(module)})
    return candidates


def resolve_stacks(model, layer_paths):
    if isinstance(layer_paths, str):
        layer_paths = [layer_paths]
    if not layer_paths:
        candidates = discover_stacks(model)
        if len(candidates) != 1:
            choices = ", ".join(item["path"] for item in candidates) or "none"
            raise ValueError(f"Cannot choose a layer stack automatically (candidates: {choices}). "
                             "Supply layer paths or an explicit Plan.")
        layer_paths = [candidates[0]["path"]]
    layers = []
    for path in layer_paths:
        try:
            blocks = model.get_submodule(path)
        except AttributeError:
            raise ValueError(f"Unknown layer stack: {path}") from None
        if not isinstance(blocks, _STACK_TYPES) or not len(blocks):
            raise ValueError(f"{path!r} is not a nonempty ModuleList or Sequential")
        for name, child in blocks._modules.items():
            layers.append((f"{path}.{name}" if path else name, child))
    if len({id(module) for _, module in layers}) != len(layers):
        raise ValueError("Selected stacks share a layer module. Supply an explicit Plan "
                         "with one hook per module to record repeated calls.")
    return layers


def module_plan(model: nn.Module, *, layer_paths: Sequence[str] | str | None = None,
                rows: dict[str, Iterable[Row]] | None = None, top=5, all_terms=False,
                default_probes=True, probe_adapters=()):
    """Build a plan for one batch-first, position-preserving token stream.

    Each custom adapter receives (model, name, module, pending_rows) and the keyword
    options top, all_terms, and default_probes. A matching adapter consumes any
    handled row requests and returns a Probe, False to omit a supported probe,
    or None for generic fallback.
    When building a layers-only plan, pass default_probes=False and no rows.
    """
    if top < 0:
        raise ValueError("top must be nonnegative")
    layers = resolve_stacks(model, layer_paths)
    pending = {name: tuple(selection) for name, selection in (rows or {}).items()}
    adapters = tuple(probe_adapters)
    probes, skipped = [], []
    for name, module in model.named_modules():
        if not name:
            continue
        probe = None
        for adapter in adapters:
            probe = adapter(model, name, module, pending, top=top, all_terms=all_terms,
                            default_probes=default_probes)
            if probe is not None:
                break
        is_expert = bool(_EXPERT_COMPONENTS.intersection(name.split(".")))
        if probe is None and not is_expert:
            if type(module) in (nn.Linear, nn.Embedding):
                requested = pending.pop(name, (Row(0),) if default_probes else ())
                if requested and type(module) is nn.Linear:
                    probe = linear_probe(name, module, requested, top=top, all_terms=all_terms)
                elif requested:
                    probe = embedding_probe(name, module, requested)
        if probe is False:
            continue
        if probe is not None:
            probes.append(probe)
        elif any(True for _ in module.parameters(recurse=False)):
            if is_expert or type(module) not in (nn.Linear, nn.Embedding):
                skipped.append(name)
    if pending:
        raise ValueError("requested projections need an adapter or do not exist: " + ", ".join(pending))
    return Plan(layers, probes), skipped
