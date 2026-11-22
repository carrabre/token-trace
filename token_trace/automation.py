"""Automation-friendly serialization shared by the console commands."""

import argparse
import json
import sys
from pathlib import Path


ERROR_SCHEMA = "token-trace/error/1"
PROGRESS_SCHEMA = "token-trace/progress/1"


class AgentParser(argparse.ArgumentParser):
    """Emit structured diagnostics even when argument parsing fails."""

    def __init__(self, *args, argv=None, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)
        self.argv = list(sys.argv[1:] if argv is None else argv)
        boundary = self.argv.index("--") if "--" in self.argv else len(self.argv)
        self.json_errors = "--json-errors" in self.argv[:boundary]

    def parse_args(self, args=None, namespace=None):
        return super().parse_args(self.argv if args is None else args, namespace)

    def error(self, message, *, code="invalid_arguments"):
        if not self.json_errors:
            super().error(message)
        record = {"schema": ERROR_SCHEMA, "event": "error", "code": code, "message": message}
        sys.stderr.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        raise SystemExit(2)


def add_output_options(parser, *, progress=False):
    parser.add_argument("--compact", action="store_true", help="emit one compact JSON document")
    parser.add_argument("--json-errors", action="store_true", help="report errors as JSON on stderr")
    if progress:
        parser.add_argument("--progress", action="store_true", help="emit progress as JSON Lines on stderr")


def write_json(data, output: Path | None = None, *, compact=False):
    options = {"separators": (",", ":")} if compact else {"indent": 2}
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, **options) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text, encoding="utf-8")


def progress_emitter(stream):
    """Create an emitter independent of the model's redirected diagnostic streams."""
    def emit(event, **data):
        if event == "sequence":
            data = {"token_count": len(data["tokens"]), "layers": data["layers"],
                    "mode": data["mode"]}
        record = {"schema": PROGRESS_SCHEMA, "event": event, **data}
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()

    return emit
