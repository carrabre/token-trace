"""Adapter for batch-first Hugging Face causal language models."""

from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from .core import Plan, Probe, linear_terms, token_row
from .mixtral import expert_probe, is_packed_mixtral

_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers",
                "model.decoder.layers", "transformer.blocks", "model.language_model.layers")
_PACKED = {"expert", "experts", "shared_expert", "shared_experts"}


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


def hf_plan(model, *, layer_path=None, rows: dict[str, Iterable[Row]] | None = None,
            top=5, all_terms=False):
    """Find the block stack and attach safe affine probes. Packed experts need an adapter."""
    from transformers.pytorch_utils import Conv1D

    choices = (layer_path,) if layer_path else _LAYER_PATHS
    expected = getattr(model.config, "num_hidden_layers", None)
    for path in choices:
        try:
            blocks = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks):
            break
    else:
        raise ValueError("block stack not found. Pass --layers PATH or supply a custom Plan")
    if expected is not None and len(blocks) != expected:
        raise ValueError(f"{path} has {len(blocks)} blocks, config declares {expected}")
    probes, skipped = [], []
    rows = dict(rows or {})
    for name, module in model.named_modules():
        if not name:
            continue
        if is_packed_mixtral(model, module):
            requests = rows.pop(name, [Row(0)])
            probes.append(expert_probe(name, module, requests,
                                       gate_rows=rows.pop(name + ".gate", []),
                                       up_rows=rows.pop(name + ".up", []),
                                       top=top, all_terms=all_terms))
            continue
        if type(module) not in (nn.Linear, Conv1D):
            if _PACKED.intersection(name.split(".")) or (
                list(module.parameters(recurse=False)) and not isinstance(module, nn.Embedding)
            ):
                skipped.append(name)
            continue
        if _PACKED.intersection(name.split(".")):
            skipped.append(name)
            continue
        requests = rows.pop(name, [Row(0)])
        probes.append(linear_probe(name, module, requests, top=top,
                                   all_terms=all_terms, transposed=type(module) is Conv1D))
    if rows:
        raise ValueError("requested projections need an adapter or do not exist: " + ", ".join(rows))
    layers = [(f"{path}.{i}", block) for i, block in enumerate(blocks)]
    return Plan(layers, probes), skipped
