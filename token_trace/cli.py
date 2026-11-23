from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import sys

from .automation import AgentParser, add_output_options, progress_emitter, write_json


def describe():
    """Describe the navigation and output contract without importing the model runtime."""
    from .deepseek import describe_capture

    try:
        release = version("token-trace")
    except PackageNotFoundError:
        release = None
    return {
        "schema": "token-trace/describe/1",
        "package": "token-trace",
        "version": release,
        "commands": {
            "token-trace": {
                "purpose": "Inventory a local Hugging Face checkpoint or trace tokens in a causal language model.",
                "inputs": ["checkpoint ID or local directory", "positional text", "--text-file UTF-8_PATH_OR_-"],
                "operations": {
                    "inspect": ["--describe", "--inspect-model", "--coefficient PARAMETER:INDEX[,INDEX...] (repeatable)"],
                    "positions": ["--position N (repeatable)", "--all-tokens"],
                    "capture_depth": ["--depth layers", "--depth selected", "--depth weights"],
                    "weight_selection": ["--row MODULE:OUTPUT[:INPUT] (repeatable)", "--top N", "--all-terms"],
                    "capture_vectors": ["--full-vectors"],
                    "generation_replay": ["--generate N"],
                    "model": ["--revision REVISION", "--device DEVICE", "--layers PATH (repeatable)", "--local-files-only"],
                },
                "defaults": {"position": -1, "depth": "weights", "top": 5, "device": "cpu", "generate": 0},
                "schemas": {"single_position": "token-trace/1", "multiple_or_all_positions": "token-trace/sequence/1",
                            "model_inventory": "token-trace/model/1"},
                "io_flags": ["--output PATH", "--compact", "--json-errors", "--progress"],
                "examples": [
                    ["token-trace", "CHECKPOINT", "--text-file", "-", "--all-tokens", "--depth", "layers",
                     "--local-files-only", "--compact", "--json-errors"],
                    ["token-trace", "CHECKPOINT", "A mesh has geometry. Materials control shading.",
                     "--position", "1", "--position", "-1", "--depth", "selected",
                     "--row", "transformer.h.0.mlp.c_fc:2:1", "--progress"],
                ],
            },
            "token-trace-deepseek": describe_capture(),
        },
        "reports": {
            "token-trace/model/1": {"modules": "registered paths", "parameters": "registered learned tensors",
                                    "buffers": "registered non-parameter tensors",
                                    "layer_stack_candidates": "structural candidates", "coefficients": "selected scalar values"},
            "token-trace/1": {"position": "zero-based in full model input", "trace": ["layers", "probes", "coverage", "prediction?"]},
            "token-trace/sequence/1": {"tokens": "full sequence of token records", "traces": "one report per selected position"},
            "token-trace/deepseek-capture/1": {"tokens": "recorded answer traces at selected absolute positions"},
        },
        "io": {
            "stdout": "One JSON result unless --output is used.",
            "error": {"flag": "--json-errors", "stream": "stderr", "schema": "token-trace/error/1",
                      "exit_code": 2, "fields": ["schema", "event", "code", "message"]},
            "progress": {"flag": "--progress", "stream": "stderr", "encoding": "JSON Lines",
                         "schema": "token-trace/progress/1",
                         "events": ["status", "generation", "sequence", "layer"]},
        },
    }


def main(argv=None):
    parser = AgentParser(description="Trace tokens in a local Hugging Face forward pass", argv=argv)
    parser.add_argument("--describe", action="store_true", help="emit the commands and JSON schemas without loading a model")
    parser.add_argument("model", nargs="?", help="local checkpoint directory or Hugging Face model ID")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("text", nargs="?", help="prompt or multi-sentence passage")
    source.add_argument("--text-file", type=Path, help="read UTF-8 text from a file or '-' for stdin")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--position", type=int, action="append", help="repeat to select token positions")
    scope.add_argument("--all-tokens", action="store_true", help="trace every token in one forward pass")
    parser.add_argument("--layers", action="append", help="repeat to select dotted paths to token-aligned block stacks")
    parser.add_argument("--inspect-model", action="store_true", help="inventory a checkpoint without a forward pass or prompt")
    parser.add_argument("--coefficient", action="append", default=[], metavar="PARAMETER:INDEX[,INDEX...]",
                        help="sample one runtime parameter coordinate during model inspection")
    parser.add_argument("--revision", help="pin a Hub checkpoint revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top", type=int, default=5, help="number of contribution terms per row")
    parser.add_argument("--row", action="append", default=[], metavar="MODULE:OUTPUT[:INPUT]",
                        help="repeat to select an affine row, weight term, or embedding feature")
    parser.add_argument("--depth", choices=("layers", "selected", "weights"), default="weights")
    parser.add_argument("--generate", type=int, default=0, help="generate up to this many tokens, then replay the sequence")
    parser.add_argument("--all-terms", action="store_true", help="emit every coefficient in each selected row")
    parser.add_argument("--full-vectors", action="store_true", help="include selected tokens' layer vectors")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    add_output_options(parser, progress=True)
    args = parser.parse_args()
    if args.describe:
        try:
            write_json(describe(), args.output, compact=args.compact)
        except OSError as error:
            parser.error(str(error), code="output_failed")
        return
    if not args.model:
        parser.error("a checkpoint is required unless --describe is used")
    if args.inspect_model and (args.text_file is not None or args.text is not None):
        parser.error("--inspect-model does not accept text")
    if not args.inspect_model and args.text_file is None and args.text is None:
        parser.error("supply text or --text-file")
    if args.coefficient and not args.inspect_model:
        parser.error("--coefficient requires --inspect-model")
    if args.top < 0:
        parser.error("--top must be nonnegative")
    if not 0 <= args.generate <= 128:
        parser.error("--generate must be between 0 and 128")
    if not args.inspect_model and args.depth == "selected" and not args.row:
        parser.error("--depth selected requires --row")

    positions = None if args.all_tokens else args.position or [-1]
    emit = progress_emitter(sys.stderr) if args.progress else None
    try:
        if args.text_file == Path("-"):
            text = sys.stdin.read()
        elif args.text_file is not None:
            text = args.text_file.read_text(encoding="utf-8")
        else:
            text = args.text

        def run():
            if args.inspect_model:
                from .session import inspect_checkpoint, parse_coefficients

                return inspect_checkpoint(args.model, revision=args.revision, device=args.device,
                                          local_only=args.local_files_only,
                                          coefficients=tuple(parse_coefficients(args.coefficient)))

            from .session import model_sequence, parse_rows

            return model_sequence(
                args.model, text, positions=positions, revision=args.revision,
                device=args.device, local_only=args.local_files_only, layer_path=args.layers,
                rows=parse_rows(args.row), top=args.top, all_terms=args.all_terms,
                full_vectors=args.full_vectors, depth=args.depth, generate=args.generate, emit=emit,
            )

        if args.progress or args.json_errors:
            # Keep dependency diagnostics separate from the command's machine-readable events.
            with open(os.devnull, "w", encoding="utf-8") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
                data = run()
        else:
            with redirect_stdout(sys.stderr):
                data = run()
    except ImportError:
        parser.error("install Hugging Face support with pip install '.[hf]'", code="missing_dependency")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error), code="inspect_failed" if args.inspect_model else "trace_failed")

    if not args.inspect_model and not args.all_tokens and len(positions) == 1:
        # Retain the one-token CLI schema and token fields.
        single = data["traces"][0]
        single.update(model=data["model"], revision=data["revision"],
                      tokens=[{"id": token["id"], "text": token["text"]} for token in data["tokens"]])
        if args.generate:
            single.update(generated=data["generated"], mode=data["mode"])
        data = single
    try:
        write_json(data, args.output, compact=args.compact)
    except (OSError, ValueError) as error:
        parser.error(str(error), code="output_failed")


if __name__ == "__main__":
    main()
