import unittest

import torch
from torch import nn

from token_trace import Layer, Plan, Probe, trace_many
from token_trace.inventory import inspect_model, parameter_value
from token_trace.portable import Row, discover_stacks, module_plan


class Mixer(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.projection = nn.Linear(1, 1, bias=False)
        self.projection.weight.detach().fill_(gain)

    def forward(self, hidden):
        return self.projection(hidden)


class Runtime(nn.Module):
    """A model with no Hugging Face config or conventional layer names."""

    def __init__(self, second_stack=False):
        super().__init__()
        self.trunk = nn.Module()
        self.trunk.stages = nn.ModuleList([Mixer(2), Mixer(3)])
        if second_stack:
            self.refinement = nn.ModuleList([Mixer(5), Mixer(7)])

    def forward(self, input_ids):
        hidden = input_ids.unsqueeze(-1).float()
        for block in self.trunk.stages:
            hidden = block(hidden)
        for block in getattr(self, "refinement", ()):
            hidden = block(hidden)
        return hidden


class PortableTests(unittest.TestCase):
    def test_discovers_unique_custom_stack_and_affine_terms(self):
        model = Runtime()
        self.assertEqual([(item["path"], item["layer_count"]) for item in discover_stacks(model)],
                         [("trunk.stages", 2)])
        self.assertFalse(hasattr(model, "config"))
        plan, _ = module_plan(
            model, default_probes=False, rows={"trunk.stages.1.projection": [Row(0, 0)]},
        )
        report = trace_many(model, plan, torch.tensor([[2, 5, 9]]), [0, 2], full_vectors=True)
        for token, value in zip(report["traces"], (2, 9)):
            self.assertEqual([event["name"] for event in token["layers"]],
                             ["trunk.stages.0", "trunk.stages.1"])
            self.assertEqual(token["layers"][1]["output"]["values"], [6.0 * value])
            self.assertEqual(len(token["probes"]), 1)
            layer, probe = token["layers"][1], token["probes"][0]
            self.assertLess(token["layers"][0]["event_index"], probe["event_index"])
            self.assertLess(probe["event_index"], layer["event_index"])
            row = probe["data"]["rows"][0]
            self.assertEqual(row["selected_term"]["activation"], 2.0 * value)
            self.assertEqual(row["selected_term"]["weight"], 3.0)
            self.assertEqual(row["selected_term"]["product"], 6.0 * value)
            self.assertEqual(row["reconstructed"], row["measured"])

    def test_competing_stacks_require_explicit_selection(self):
        model = Runtime(second_stack=True)
        self.assertEqual({item["path"] for item in discover_stacks(model)}, {"trunk.stages", "refinement"})
        with self.assertRaises(ValueError) as rejected:
            module_plan(model)
        self.assertIn("trunk.stages", str(rejected.exception))
        self.assertIn("refinement", str(rejected.exception))
        plan, _ = module_plan(model, layer_paths=["refinement", "trunk.stages"], default_probes=False)
        report = trace_many(model, plan, torch.tensor([[2, 5, 9]]), [1], full_vectors=True)["traces"][0]
        self.assertEqual([event["name"] for event in report["layers"]],
                         ["trunk.stages.0", "trunk.stages.1", "refinement.0", "refinement.1"])
        self.assertEqual(report["layers"][-1]["output"]["values"], [1050.0])
        self.assertEqual(report["probes"], [])

    def test_positional_identity_for_different_sequence_axes_is_explicit(self):
        class RuntimeWithTwoAxes(nn.Module):
            def __init__(self):
                super().__init__()
                self.encode = Mixer(2)
                self.decode = Mixer(3)

            def forward(self, *, source, target):
                memory = self.encode(source.unsqueeze(-1).float())
                return self.decode(target.unsqueeze(-1).float() + memory[:, :1])

        source = torch.tensor([[9, 8, 7]])
        target = torch.tensor([[2, 5]])
        model = RuntimeWithTwoAxes()
        plan = Plan(
            [
                Layer("source", model.encode, lambda call: (None, None)),
                ("target", model.decode),
            ],
            [Probe("decode.input", model.decode, lambda call: {
                "value": call.args[0][0, call.position, 0].item(),
            })],
        )
        report = trace_many(
            model, plan, target, [1], full_vectors=True,
            runner=lambda runtime, ids, _: runtime(source=source, target=ids),
            logits_reader=lambda result, position, _: result[0, position],
        )["traces"][0]
        self.assertEqual(report["token_id"], 5)
        self.assertEqual(report["coverage"]["unmapped_layers"], ["source"])
        self.assertEqual(report["layers"][1]["input"]["values"], [23.0])
        self.assertEqual(report["layers"][1]["output"]["values"], [69.0])
        self.assertGreater(report["probes"][0]["event_index"], report["layers"][0]["event_index"])
        self.assertEqual(report["prediction"][0]["logit"], 69.0)


    def test_custom_probe_can_opt_into_nonstandard_projection_semantics(self):
        class ShiftedLinear(nn.Linear):
            def forward(self, hidden):
                return super().forward(hidden) + 11

        model = Runtime()
        name = "trunk.stages.0.projection"
        model.trunk.stages[0].projection = ShiftedLinear(1, 1, bias=False)
        model.trunk.stages[0].projection.weight.detach().fill_(2)
        requests = {name: [Row(0)]}

        with self.assertRaisesRegex(ValueError, "need an adapter"):
            module_plan(model, rows=requests)

        def adapter(runtime, path, module, pending, **options):
            if type(module) is not ShiftedLinear:
                return None
            selected = pending.pop(path, ())
            if not selected:
                return False

            def read(call):
                return {"observed": call.output[0, call.position, selected[0].output].item()}

            return Probe(path, module, read, "shifted")

        plan, _ = module_plan(model, rows=requests, default_probes=False, probe_adapters=[adapter])
        report = trace_many(model, plan, torch.tensor([[2, 5, 9]]), [1])["traces"][0]
        self.assertEqual(len(report["probes"]), 1)
        self.assertEqual(report["probes"][0]["kind"], "shifted")
        self.assertEqual(report["probes"][0]["data"]["observed"], 21.0)
        self.assertEqual(requests, {name: [Row(0)]})

    def test_t5_decoder_tracks_its_own_positions_and_logits(self):
        from transformers import T5Config, T5ForConditionalGeneration

        model = T5ForConditionalGeneration(T5Config(
            vocab_size=24, d_model=8, d_ff=16, d_kv=4, num_heads=2,
            num_layers=2, decoder_start_token_id=0,
        )).eval()
        paths = {item["path"] for item in discover_stacks(model)}
        self.assertTrue({"encoder.block", "decoder.block"}.issubset(paths))
        name = "decoder.block.1.layer.2.DenseReluDense.wo"
        plan, _ = module_plan(
            model, layer_paths=["decoder.block"], rows={name: [Row(1, 0)]}, default_probes=False,
        )
        source = torch.tensor([[3, 7, 9]])
        target = torch.tensor([[0, 5]])

        def run(runtime, ids, options):
            return runtime(input_ids=source, decoder_input_ids=ids, **options)

        report = trace_many(
            model, plan, target, [1], runner=run, forward_kwargs={"use_cache": False},
        )["traces"][0]
        weights = report["probes"][0]["data"]["rows"][0]
        self.assertAlmostEqual(weights["reconstructed"], weights["measured"], places=6)
        self.assertEqual(report["token_id"], 5)
        self.assertEqual([event["name"] for event in report["layers"]], ["decoder.block.0", "decoder.block.1"])
        self.assertEqual(report["coverage"]["unmapped_layers"], [])
        with torch.inference_mode():
            logits = run(model, target, {"use_cache": False}).logits[0, 1]
        self.assertEqual(report["prediction"][0]["token_id"], int(logits.argmax()))
        self.assertAlmostEqual(report["prediction"][0]["logit"], logits.max().item(), places=6)

    def test_inventory_retains_aliases_persistence_and_scalar_coordinates(self):
        model = nn.Module()
        model.input = nn.Linear(2, 2, bias=False)
        model.reused = model.input
        model.output = nn.Linear(2, 2, bias=False)
        model.output.weight = model.input.weight
        model.input.weight.detach().copy_(torch.tensor([[1., 2.], [3., 4.]]))
        model.temperature = nn.Parameter(torch.tensor(0.5))
        model.register_buffer("persistent", torch.tensor([7, 8]))
        model.register_buffer("temporary", torch.tensor([9]), persistent=False)

        topology = inspect_model(model)
        self.assertEqual(topology["schema"], "token-trace/model/1")
        modules = {item["path"]: item for item in topology["modules"]}
        self.assertEqual(modules["reused"]["alias_of"], "input")
        parameters = {item["name"]: item for item in topology["parameters"]}
        self.assertIsNone(parameters["input.weight"]["alias_of"])
        self.assertEqual(parameters["reused.weight"]["alias_of"], "input.weight")
        self.assertEqual(parameters["output.weight"]["alias_of"], "input.weight")
        buffers = {item["name"]: item for item in topology["buffers"]}
        self.assertTrue(buffers["persistent"]["persistent"])
        self.assertFalse(buffers["temporary"]["persistent"])
        scalar = parameter_value(model, "output.weight", (1, 0))
        self.assertEqual(scalar["coordinate"], [1, 0])
        self.assertEqual(scalar["value"], 3.)
        self.assertEqual(parameter_value(model, "temperature", ())["value"], 0.5)
        for invalid in ((-1, 0), (2, 0), (1,)):
            with self.subTest(coordinate=invalid), self.assertRaises(ValueError):
                parameter_value(model, "input.weight", invalid)
        with self.assertRaises(ValueError):
            parameter_value(model, "persistent", (0,))


if __name__ == "__main__":
    unittest.main()
