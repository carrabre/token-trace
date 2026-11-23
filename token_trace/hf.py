"""Architecture hints and extra probes for Hugging Face causal language models."""

from torch import nn

from .mixtral import expert_probe, is_packed_mixtral
from .portable import Row, _EXPERT_COMPONENTS, embedding_probe, linear_probe, module_plan


_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers",
                "model.decoder.layers", "transformer.blocks", "model.language_model.layers")


def _hf_probe(model, name, module, rows, *, top, all_terms, default_probes):
    from transformers.pytorch_utils import Conv1D

    if is_packed_mixtral(model, module):
        down = rows.pop(name, (Row(0),) if default_probes else ())
        gate = rows.pop(name + ".gate", ())
        up = rows.pop(name + ".up", ())
        if down or gate or up:
            return expert_probe(name, module, down, gate_rows=gate, up_rows=up,
                                top=top, all_terms=all_terms)
        return False
    elif type(module) is Conv1D and not _EXPERT_COMPONENTS.intersection(name.split(".")):
        requested = rows.pop(name, (Row(0),) if default_probes else ())
        if requested:
            return linear_probe(name, module, requested, top=top, all_terms=all_terms, transposed=True)
        return False
    return None


def hf_plan(model, *, layer_path=None, rows=None, top=5, all_terms=False, default_probes=True):
    """Prefer tested stack paths, then delegate to portable discovery."""
    paths = layer_path
    if not paths:
        config = getattr(model, "config", None)
        expected = getattr(config, "num_hidden_layers", None)
        for path in _LAYER_PATHS:
            try:
                blocks = model.get_submodule(path)
            except AttributeError:
                continue
            if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks):
                if expected is None or expected == len(blocks):
                    paths = [path]
                    break
    return module_plan(model, layer_paths=paths, rows=rows, top=top, all_terms=all_terms,
                       default_probes=default_probes, probe_adapters=(_hf_probe,))
