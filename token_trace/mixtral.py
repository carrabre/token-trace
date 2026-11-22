"""Probe for the packed eager Mixtral layout used by Transformers 5."""

import torch
from torch import nn
from torch.nn import functional as F

from .core import Probe, linear_terms


def expert_probe(name, module, rows, *, gate_rows=(), up_rows=(), top=5, all_terms=False):
    """Record dispatch and replay each selected expert's affine/gating operations."""
    def terms(signal, weight, output, requests):
        results = []
        for row in requests:
            terms = linear_terms(signal, weight, row.output, input_index=row.input,
                                 top=top, all_terms=all_terms)
            replayed = output[row.output].item()
            terms.update(replayed=replayed, replayed_minus_reconstructed=replayed - terms["reconstructed"])
            results.append(terms)
        return results

    def read(call):
        inputs = call.args or tuple(call.kwargs.get(key) for key in (
            "hidden_states", "top_k_index", "top_k_weights",
        ))
        if len(inputs) < 3:
            return None
        hidden, choices, scores = inputs[:3]
        if not all(isinstance(value, torch.Tensor) and value.ndim == 2
                   and value.shape[0] == call.length
                   for value in (hidden, choices, scores, call.output)):
            return None
        if choices.shape != scores.shape:
            return None
        route = choices[call.position].tolist()
        scales = scores[call.position].tolist()
        if any(not 0 <= expert < module.down_proj.shape[0] for expert in route):
            return None
        signal = hidden[call.position]
        experts = []
        for expert, scale in zip(route, scales):
            gate_up = module.gate_up_proj[expert]
            down = module.down_proj[expert]
            gate, up = F.linear(signal, gate_up).chunk(2)
            gate_weights, up_weights = gate_up.chunk(2)
            intermediate = module.act_fn(gate) * up
            output = F.linear(intermediate, down)
            experts.append({
                "id": expert,
                "routing_weight": scale,
                "rows": terms(intermediate, down, output, rows),
                "gate_rows": terms(signal, gate_weights, gate, gate_rows),
                "up_rows": terms(signal, up_weights, up, up_rows),
                "scaled_outputs": [(output[row.output] * scale).item() for row in rows],
            })
        comparisons = []
        for i, row in enumerate(rows):
            actual = call.output[call.position, row.output].item()
            total = sum(expert["scaled_outputs"][i] for expert in experts)
            comparisons.append({"output": row.output, "measured": actual, "replayed": total,
                                "measured_minus_replayed": actual - total})
        return {"expert_intermediates": "replayed_from_recorded_dispatch_inputs",
                "experts": experts, "comparisons": comparisons}

    return Probe(name, module, read, "mixtral_expert_down")


def is_packed_mixtral(model, module):
    """Match the tested parameter layout before interpreting flattened token indices."""
    if getattr(model.config, "model_type", None) != "mixtral":
        return False
    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralExperts
    except ImportError:
        return False

    if type(module) is not MixtralExperts:
        return False
    gate_up, down = getattr(module, "gate_up_proj", None), getattr(module, "down_proj", None)
    return (isinstance(gate_up, nn.Parameter) and isinstance(down, nn.Parameter)
            and down.ndim == gate_up.ndim == 3
            and gate_up.shape == (down.shape[0], 2 * down.shape[2], down.shape[1]))
