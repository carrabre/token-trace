import unittest

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM

from token_trace import Plan, linear_terms, trace
from token_trace.hf import Row, hf_plan


class TraceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.ids = torch.tensor([[4, 9, 7, 11]])

    def run_trace(self, model, target, module, *, input_index=1, **kwargs):
        model.eval()
        plan, _ = hf_plan(model, rows={module: [Row(2, input_index)]}, all_terms=True)
        return trace(model, plan, self.ids, target, forward_kwargs={"use_cache": False},
                     full_vectors=True, **kwargs)

    def check_dense(self, model, module, dims):
        report = self.run_trace(model, 2, module)
        self.assertEqual(report["token_id"], 7)
        self.assertEqual(len(report["layers"]), 2)
        self.assertEqual(report["layers"][0]["output"]["size"], 16)
        self.assertFalse(report["coverage"]["unmapped_layers"])
        self.assertFalse(report["coverage"]["unmapped_probes"])
        row = next(record["data"]["rows"][0] for record in report["probes"] if record["name"] == module)
        self.assertEqual(len(row["terms"]), dims)
        self.assertEqual(row["selected_term"], row["terms"][1])
        self.assertAlmostEqual(row["reconstructed"], row["measured"], places=6)
        with torch.no_grad():
            logits = model(self.ids, use_cache=False).logits[0, 2]
        self.assertEqual(report["prediction"][0]["token_id"], int(logits.argmax()))
        self.assertFalse(model.get_submodule(module)._forward_hooks)

    def test_gpt2_transposed_weights(self):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_head=4, n_layer=2, bos_token_id=0, eos_token_id=1))
        self.check_dense(model, "transformer.h.0.mlp.c_fc", 16)

    def test_llama_linear_weights(self):
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                                intermediate_size=24, num_attention_heads=4,
                                num_key_value_heads=2, num_hidden_layers=2))
        self.check_dense(model, "model.layers.1.mlp.down_proj", 24)

    def test_mixtral_recorded_dispatch_and_weight_replay(self):
        model = MixtralForCausalLM(MixtralConfig(vocab_size=32, hidden_size=16,
                                  intermediate_size=24, num_attention_heads=4, num_key_value_heads=2,
                                  num_hidden_layers=2, num_local_experts=4, num_experts_per_tok=2))
        model.eval()
        name = "model.layers.1.mlp.experts"
        plan, _ = hf_plan(model, rows={name: [Row(2, 1)], name + ".gate": [Row(3, 4)],
                                       name + ".up": [Row(3, 4)]}, all_terms=True)
        report = trace(model, plan, self.ids, -1, forward_kwargs={"use_cache": False})
        self.assertEqual(report["position"], 3)
        self.assertEqual(len(report["layers"]), 2)
        self.assertFalse(report["coverage"]["unmapped_layers"])
        expert_records = [p["data"] for p in report["probes"] if p["kind"] == "mixtral_expert_down"]
        self.assertEqual(len(expert_records), 2)
        for record in expert_records:
            experts = record["experts"]
            self.assertEqual(len({e["id"] for e in experts}), 2)
            self.assertAlmostEqual(sum(e["routing_weight"] for e in experts), 1, places=6)
            self.assertAlmostEqual(record["comparisons"][0]["measured_minus_replayed"], 0, places=6)
        row = expert_records[1]["experts"][0]["rows"][0]
        self.assertEqual(len(row["terms"]), 24)
        self.assertEqual(row["selected_term"], row["terms"][1])
        for kind in ("gate_rows", "up_rows"):
            selected = expert_records[1]["experts"][0][kind][0]
            self.assertEqual(len(selected["terms"]), 16)
            self.assertEqual(selected["selected_term"], selected["terms"][4])
            self.assertAlmostEqual(selected["replayed_minus_reconstructed"], 0, places=6)

    def test_hook_cleanup_on_forward_failure(self):
        class Broken(nn.Sequential):
            def forward(self, x):
                raise RuntimeError("broken")
        model = Broken(nn.Identity())
        with self.assertRaisesRegex(RuntimeError, "broken"):
            trace(model, Plan([("0", model[0])]), self.ids, 0)
        self.assertFalse(model[0]._forward_hooks)

    def test_exact_product_and_bias(self):
        row = linear_terms(torch.tensor([2., -3.]), torch.tensor([[.5, 4.]]), 0,
                           bias=torch.tensor([2.]), input_index=1, top=1)
        self.assertEqual(row["selected_term"], {"input": 1, "activation": -3.,
                                                  "weight": 4., "product": -12.})
        self.assertEqual(row["reconstructed"], -9.)


if __name__ == "__main__":
    unittest.main()
