"""Trace a token through two packed experts. Run with python -m examples.packed_moe."""
import json

import torch
from torch import nn
from torch.nn import functional as F

from token_trace import Plan, Probe, TracePoint, linear_terms, packed_token_row, trace


class PackedExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1., -2.], [.5, 4.]]) * scale)
        self.down = TracePoint()

    def forward(self, hidden, positions):
        result = F.linear(hidden, self.weight)
        return self.down(result, positions=positions, operand=hidden, weight=self.weight)


class RoutedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = TracePoint()
        self.experts = nn.ModuleList([PackedExpert(1), PackedExpert(2), PackedExpert(3)])

    def forward(self, hidden):
        routes = torch.tensor([[.25, .75], [0., 1.], [1., 0.]])
        self.router(routes)
        # Token 0 visits two experts in different packed positions. Expert 2 is idle.
        combined = torch.zeros_like(hidden[0])
        for expert, indices in enumerate(([2, 0], [0, 1])):
            positions = torch.tensor(indices)
            output = self.experts[expert](hidden[0, positions], positions)
            combined.index_add_(0, positions, output * routes[positions, expert, None])
        return combined.unsqueeze(0)


class RoutedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = RoutedLayer()

    def forward(self, input_ids):
        hidden = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
        return self.layer(hidden)


def read_down(call):
    positions = call.kwargs["positions"]
    signal = packed_token_row(call.kwargs["operand"], positions, call.position, call.length)
    value = packed_token_row(call.output, positions, call.position, call.length)
    if signal is None or value is None:
        return None
    return linear_terms(signal, call.kwargs["weight"], 0, measured=value, all_terms=True)



def make_plan(model):
    probes = [Probe(f"expert.{i}.down", expert.down, read_down, "expert_down")
              for i, expert in enumerate(model.layer.experts)]
    return Plan([("layer", model.layer)], probes)


if __name__ == "__main__":
    model = RoutedModel().eval()
    tokens = torch.tensor([[2, 3, 5]])
    print(json.dumps(trace(model, make_plan(model), tokens, 0, full_vectors=True), indent=2))
