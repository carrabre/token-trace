"""Contract tests for agent-facing CLI streams and schemas."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from token_trace import cli


def invoke(args, stdin=""):
    out, err = StringIO(), StringIO()
    with patch.object(sys, "stdin", StringIO(stdin)), redirect_stdout(out), redirect_stderr(err):
        try:
            cli.main(args)
        except SystemExit as error:
            code = error.code
        else:
            code = 0
    return code, out.getvalue(), err.getvalue()


def fake_session(run):
    session = ModuleType("token_trace.session")
    session.model_sequence = run
    session.parse_rows = lambda rows: rows
    return session


def sequence():
    return {"schema": "token-trace/sequence/1", "traces": [
        {"schema": "token-trace/1", "position": 0, "token_id": 42, "layers": [], "probes": []}],
        "tokens": [{"position": 0, "id": 42, "text": "café"}], "model": "fixture", "revision": None}


class AutomationTests(unittest.TestCase):
    def test_discovery_skips_runtime(self):
        with patch.dict(sys.modules, {"token_trace.session": None}):
            code, output, error = invoke(["--describe", "--compact", "--json-errors"])
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(len(output.splitlines()), 1)
        contract = json.loads(output)
        self.assertEqual(contract["schema"], "token-trace/describe/1")
        self.assertEqual(contract["commands"]["token-trace"]["schemas"]["single_position"], "token-trace/1")
        self.assertIn("token-trace-deepseek", contract["commands"])

    def test_parser_errors_stay_off_stdout(self):
        cases = [
            ["--unknown", "--json-errors"],
            ["--json-errors"],
            ["checkpoint", "--all-tokens", "--position", "0", "--json-errors"],
            ["checkpoint", "prompt", "--depth", "selected", "--json-errors"],
        ]
        for args in cases:
            with self.subTest(args=args):
                code, output, error = invoke(args)
                self.assertEqual((code, output), (2, ""))
                diagnostic = json.loads(error)
                self.assertEqual(diagnostic["schema"], "token-trace/error/1")
                self.assertEqual(diagnostic["event"], "error")
                self.assertEqual(diagnostic["code"], "invalid_arguments")
                self.assertTrue(diagnostic["message"])

    def test_stdout_and_progress_are_separate(self):
        def run(model, text, **options):
            self.assertEqual((model, text, options["positions"]), ("fixture", "café", None))
            print("dependency warning", file=sys.stderr)
            print("dependency chatter")
            emit = options["emit"]
            emit("status", message="reading weights")
            emit("sequence", tokens=[42], layers=["a", "b"], mode="forward", input=text)
            emit("layer", name="a", values=[{"position": 0, "output_l2": 2}])
            return sequence()

        with patch.dict(sys.modules, {"token_trace.session": fake_session(run)}):
            code, output, error = invoke(["fixture", "--text-file", "-", "--all-tokens",
                                          "--compact", "--progress", "--json-errors"], "café")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output), sequence())
        records = list(map(json.loads, error.splitlines()))
        self.assertEqual([r["event"] for r in records], ["status", "sequence", "layer"])
        self.assertTrue(all(r["schema"] == "token-trace/progress/1" for r in records))
        self.assertEqual(records[1]["token_count"], 1)
        self.assertNotIn("input", records[1])

    def test_file_output_keeps_legacy_single_token_schema(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            with patch.dict(sys.modules, {"token_trace.session": fake_session(lambda *a, **k: sequence())}):
                code, output, error = invoke(["fixture", "café", "--output", str(path)])
            self.assertEqual((code, output, error), (0, "", ""))
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["schema"], "token-trace/1")
            self.assertEqual(result["tokens"], [{"id": 42, "text": "café"}])

    def test_runtime_failure_can_follow_progress(self):
        def fail(*args, **options):
            options["emit"]("status", message="loading")
            raise RuntimeError("unsupported device")

        with patch.dict(sys.modules, {"token_trace.session": fake_session(fail)}):
            code, output, error = invoke(["fixture", "text", "--progress", "--json-errors"])
        self.assertEqual((code, output), (2, ""))
        progress, failure = map(json.loads, error.splitlines())
        self.assertEqual(progress["event"], "status")
        self.assertEqual(failure["code"], "trace_failed")
        self.assertEqual(failure["message"], "unsupported device")


if __name__ == "__main__":
    unittest.main()
