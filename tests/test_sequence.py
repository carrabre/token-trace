import unittest

import torch
from torch import nn

from token_trace import Plan, trace, trace_many
from token_trace.sequence import sentence_spans, tokens_for_sequence


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Identity()
        self.second = nn.Identity()
        self.calls = 0

    def forward(self, ids):
        self.calls += 1
        vectors = torch.stack([ids.float(), 2 * ids.float()], dim=-1)
        return self.second(self.first(vectors) + 1)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.model = CountingModel()
        self.plan = Plan([("first", self.model.first), ("second", self.model.second)])
        self.ids = torch.tensor([[4, 7, 9, 2]])

    def test_many_positions_share_one_forward_and_layer_events(self):
        events = []
        result = trace_many(self.model, self.plan, self.ids, [2, 0, -1],
                            full_vectors=True, on_layer=lambda name, values: events.append((name, values)))
        self.assertEqual(self.model.calls, 1)
        self.assertEqual([item["position"] for item in result["traces"]], [2, 0, 3])
        self.assertEqual([item["token_id"] for item in result["traces"]], [9, 4, 2])
        self.assertEqual(result["traces"][0]["layers"][1]["output"]["values"], [10., 19.])
        self.assertEqual([name for name, _ in events], ["first", "second"])
        self.assertEqual([item["position"] for item in events[1][1]], [2, 0, 3])
        self.assertFalse(self.model.first._forward_hooks)

    def test_all_tokens_and_one_token_schemas(self):
        result = trace_many(self.model, self.plan, self.ids)
        self.assertEqual(len(result["traces"]), 4)
        self.assertEqual(result["traces"][1], trace(self.model, self.plan, self.ids, 1))
        with self.assertRaisesRegex(ValueError, "unique"):
            trace_many(self.model, self.plan, self.ids, [-1, 3])

    def test_sentence_boundaries_do_not_split_three_js(self):
        text = "Three.js renders geometry. Materials control light.\nThen draw it."
        spans = sentence_spans(text)
        self.assertEqual([text[a:b] for a, b in spans],
                         ["Three.js renders geometry.", "Materials control light.", "Then draw it."])
        ids = [(10, "Three.js"), (11, " Materials"), (12, "Then"), (13, "continue")]
        tokens = tokens_for_sequence(ids, text,
                                     [[0, 8], [spans[1][0] - 1, spans[1][0] + 9], [len(text)-13, len(text)-9]],
                                     prompt_count=3)
        self.assertEqual([token.get("sentence") for token in tokens], [1, 2, 3, None])
        self.assertEqual(tokens[-1]["segment"], "generated")


if __name__ == "__main__":
    unittest.main()
