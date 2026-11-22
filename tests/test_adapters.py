import unittest

import torch
from torch import nn

from token_trace import Layer, Plan, packed_token_row, trace
from examples.packed_moe import RoutedModel, make_plan


class AdapterTests(unittest.TestCase):
    def test_trace_dispatch_to_real_expert_operands(self):
        model = RoutedModel()
        plan = make_plan(model)
        tokens = torch.tensor([[2, 3, 5]])
        report = trace(model, plan, tokens, 0, full_vectors=True)
        self.assertEqual(report["layers"][0]["output"]["values"][0], -7.)
        self.assertEqual(report["coverage"]["unexecuted_probes"], ["expert.2.down"])
        for expected, probe in zip((-4., -8.), report["probes"]):
            self.assertEqual(probe["data"]["measured"], expected)
            self.assertEqual(probe["data"]["reconstructed"], expected)
            self.assertEqual([term["activation"] for term in probe["data"]["terms"]], [2., 3.])
        second = trace(model, plan, tokens, 2)
        self.assertEqual(second["coverage"]["unmapped_probes"], ["expert.1.down"])
        self.assertIsNone(second["probes"][1]["data"])

    def test_packed_token_map_matches_identity_instead_of_shape(self):
        hidden = torch.tensor([[9., 7.], [4., 3.], [0., 0.]])
        row = packed_token_row(hidden, torch.tensor([2, 0, -1]), 0, 3)
        self.assertEqual(row.tolist(), [4., 3.])
        for dtype in (torch.uint8, torch.int8):
            row = packed_token_row(hidden, torch.tensor([2, 0, 1], dtype=dtype), 0, 512)
            self.assertEqual(row.tolist(), [4., 3.])
        for bad_map in ([0, 0, 1], [0, 1, 3], [2, 1, -1], [0, 1], [0., 1., 2.]):
            self.assertIsNone(packed_token_row(hidden, torch.tensor(bad_map), 0, 3))

    def test_custom_layout_runner_and_logits_preserve_repeated_calls(self):
        class Block(nn.Module):
            def forward(self, *, activations):
                return {"updated": activations * 2}

        class Runtime(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward(self, *, tokens, delta):
                hidden = tokens.T.unsqueeze(-1).float() + delta
                first = self.block(activations=hidden)["updated"]
                second = self.block(activations=first)["updated"]
                return second.repeat(1, 1, 2)

        model = Runtime()

        def read_layer(call):
            return (call.kwargs["activations"][call.position, 0],
                    call.output["updated"][call.position, 0])

        report = trace(
            model, Plan([Layer("shared", model.block, read_layer)]),
            torch.tensor([[2, 5, 9]]), 1, full_vectors=True, forward_kwargs={"delta": 1},
            runner=lambda runtime, ids, options: runtime(tokens=ids, **options),
            logits_reader=lambda result, position, length: result[position, 0],
        )
        self.assertEqual([event["name"] for event in report["layers"]], ["shared", "shared"])
        self.assertEqual([event["input"]["values"] for event in report["layers"]], [[6.], [12.]])
        self.assertEqual(report["prediction"][0]["logit"], 24.)
        self.assertFalse(report["coverage"]["unmapped_layer_inputs"])
        self.assertFalse(model.block._forward_hooks)


if __name__ == "__main__":
    unittest.main()
