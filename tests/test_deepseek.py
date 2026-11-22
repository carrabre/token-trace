import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest

from token_trace.deepseek import (
    CaptureStore, EXPERTS, LAYERS, NEURONS, REVISION, WIDTH, WeightTerm,
    decode_operands, trace_capture,
)


def bf16(value):
    bits, = struct.unpack("<I", struct.pack("<f", value))
    return struct.pack("<H", bits >> 16)


def operands_file():
    hidden = bytearray(LAYERS * WIDTH * 2)
    offset = 2 * (24 * WIDTH + 32)
    hidden[offset:offset + 2] = bf16(0.171875)
    pre_quant = bytearray(NEURONS * 2)
    w2_input = bytearray(NEURONS * 2)
    w2_input[2 * 1076:2 * 1077] = bf16(0.4375)
    entry = struct.pack("<hhf", 24, 34, 0.25) + pre_quant + w2_input
    return struct.pack("<4sHH", b"WMAP", LAYERS, WIDTH) + hidden + entry


class DeepSeekCaptureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "cases").mkdir()
        (self.root / "runs").mkdir()
        self.root.joinpath("research.json").write_text(json.dumps({
            "checkpoint_revision": REVISION, "real_model_measurements": True,
            "requests": [{"request_id": "case", "caseFile": "case.json.gz"}],
        }), encoding="utf-8")
        payload = operands_file()
        self.root.joinpath("runs/input.bin.gz").write_bytes(gzip.compress(payload))
        ids = [34, 2, 3, 4, 5, EXPERTS - 1]
        weights = [0.25] * len(ids)
        self.case = {
            "request_id": "case", "prompt_ids": [100], "pieces": [" THREE", ".B"],
            "generated_ids": [94175, 7342],
            "tokens": {"1": {"file": "runs/input.bin.gz", "records": 1, "uncompressed_bytes": len(payload)}},
            "routes": [{"ids": [ids] * 3, "weights": [weights] * 3} for _ in range(LAYERS)],
        }
        self.root.joinpath("cases/case.json.gz").write_bytes(gzip.compress(json.dumps(self.case).encode()))
        self.w2 = WeightTerm(24, 34, 1076, "w2", 647)
        self.w1 = WeightTerm(24, 34, 1076, "w1", 32)
        self.w3 = WeightTerm(24, 34, 1076, "w3", 32)
        self.unused = WeightTerm(24, 0, 1076, "w2", 647)
        self.weight_file = self.root / "terms.json"
        self.weight_file.write_text(json.dumps({"values": [
            {"tensor": term.tensor, "coordinate": term.coordinate, "revision": REVISION, "value": value}
            for term, value in [(self.w2, -0.0234375), (self.w1, 0.015625), (self.w3, -0.015625), (self.unused, 1.0)]
        ]}), encoding="utf-8")
        self.store = CaptureStore(self.root, None, [self.weight_file])

    def test_join_routes_tokens_and_exact_weight_coordinates(self):
        trace = trace_capture(self.store, self.store.case("case"), 1, [self.w2, self.w1, self.w3], strongest=True)
        self.assertEqual(trace["token"], {"id": 94175, "piece": " THREE"})
        self.assertEqual(trace["next_recorded_token"], {"id": 7342, "piece": ".B"})
        self.assertEqual([layer["stage"] for layer in trace["layers"]], ["encoder"] * 20 + ["decoder"] * 20)
        self.assertTrue(all(len(layer["experts"]) == 7 for layer in trace["layers"]))
        self.assertEqual(trace["layers"][24]["experts"][0]["strongest_w2_input"],
                         {"neuron": 1076, "w2_input": 0.4375})
        self.assertEqual(trace["coverage"]["loaded_expert_records"], 1)
        self.assertEqual([term["local_contribution"] for term in trace["weight_terms"]],
                         [-0.01025390625, 0.002685546875, -0.002685546875])
        self.assertEqual([term["coefficient"]["coordinate"] for term in trace["weight_terms"]],
                         [[647, 1076], [1076, 32], [1076, 32]])

    def test_distinguishes_missing_operand_from_unselected_expert(self):
        trace = trace_capture(self.store, self.case, 2, [self.w2, self.unused])
        self.assertIsNone(trace["next_recorded_token"])
        self.assertIsNone(trace["weight_terms"][0]["local_contribution"])
        self.assertEqual(trace["weight_terms"][0]["coverage"], "operand_not_recorded")
        self.assertEqual(trace["weight_terms"][1]["local_contribution"], 0)
        self.assertEqual(trace["weight_terms"][1]["coverage"], "expert_not_routed")

    def test_report_fails_when_recorded_dispatch_does_not_match_route(self):
        self.case["routes"][24]["weights"][1] = [0.5] * 6
        with self.assertRaisesRegex(ValueError, "routing weight disagrees"):
            trace_capture(self.store, self.case, 1, [self.w2])

    def test_route_only_does_not_read_or_require_operand_file(self):
        self.root.joinpath("runs/input.bin.gz").unlink()
        trace = trace_capture(self.store, self.case, 1, route_only=True)
        self.assertEqual(trace["coverage"]["loaded_expert_records"], 0)
        self.assertEqual(trace["coverage"]["routing_layers"], LAYERS)

    def test_rejects_bad_headers_and_truncated_expert_record(self):
        with self.assertRaisesRegex(ValueError, "WMAP header"):
            decode_operands(b"other")
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_operands(operands_file()[:-1])

    def test_offline_weight_file_must_match_checkpoint(self):
        record = {"tensor": self.w2.tensor, "coordinate": self.w2.coordinate, "revision": "other", "value": 1}
        self.weight_file.write_text(json.dumps(record), encoding="utf-8")
        store = CaptureStore(self.root, None, [self.weight_file])
        with self.assertRaisesRegex(ValueError, "revision did not match"):
            trace_capture(store, self.case, 1, [self.w2])

    def test_shared_expert_tensor_path(self):
        term = WeightTerm(24, -1, 1076, "w2", 647)
        self.assertEqual(term.tensor, "layers.24.ffn.shared_experts.w2.weight")


if __name__ == "__main__":
    unittest.main()
