"""Replay published DeepSeek V4.1 Flash 1017 observations without running the model."""

import argparse
from dataclasses import dataclass
import gzip
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
from urllib.parse import urlencode
from urllib.request import urlopen

from .automation import AgentParser, add_output_options, write_json
from .sequence import tokens_for_sequence


REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DATA_URL = "https://neuron.carrabre.com/data/weights"
WEIGHT_API = "https://neuron.carrabre.com/api/weights"
FEATURED_CASE = "wm-00-3d"
LAYERS = 40
ENCODER_LAYERS = 20
EXPERTS = 384
ROUTED_PER_LAYER = 6
WIDTH = 5120
NEURONS = 2304
MAX_BYTES = 16 * 1024 * 1024


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _read(source):
    if isinstance(source, Path):
        with source.open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
    else:
        with urlopen(source, timeout=30) as stream:
            data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"file exceeds {MAX_BYTES} bytes")
    if data.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"uncompressed file exceeds {MAX_BYTES} bytes")
    return data


def _resource(root, key):
    segments = key.split("/")
    if any(part in ("", ".", "..") for part in segments) or any(c in key for c in "\\%?#:"):
        raise ValueError("capture paths must be relative paths within the data root")
    return root / PurePosixPath(key) if isinstance(root, Path) else root.rstrip("/") + "/" + key


def _bf16(data, offset):
    bits, = struct.unpack_from("<H", data, offset)
    return _bf16_bits(bits)


def _bf16_bits(bits):
    value, = struct.unpack("<f", struct.pack("<I", bits << 16))
    if not math.isfinite(value):
        raise ValueError("nonfinite bfloat16 operand")
    return value


@dataclass(frozen=True)
class ExpertRecord:
    routing_weight: float
    w2_input_offset: int


@dataclass(frozen=True)
class Operands:
    data: bytes
    records: dict[tuple[int, int], ExpertRecord]

    def ffn_input(self, layer, feature):
        return _bf16(self.data, 8 + 2 * (layer * WIDTH + feature))

    def w2_input(self, layer, expert, neuron):
        return _bf16(self.data, self.records[layer, expert].w2_input_offset + 2 * neuron)

    def strongest_w2(self, layer, expert):
        offset = self.records[layer, expert].w2_input_offset
        bits = struct.unpack_from(f"<{NEURONS}H", self.data, offset)
        if any((value & 0x7F80) == 0x7F80 for value in bits):
            raise ValueError("nonfinite bfloat16 operand")
        index = max(range(NEURONS), key=lambda i: bits[i] & 0x7FFF)
        return {"neuron": index, "w2_input": _bf16_bits(bits[index])}


def decode_operands(data):
    """Decode the uncompressed, little-endian WMAP file used by the explorer."""
    if len(data) < 8 or struct.unpack_from("<4sHH", data) != (b"WMAP", LAYERS, WIDTH):
        raise ValueError("unexpected DeepSeek 1017 WMAP header")
    offset = 8 + 2 * LAYERS * WIDTH
    record_bytes = 8 + 4 * NEURONS
    if len(data) < offset or (len(data) - offset) % record_bytes:
        raise ValueError("truncated DeepSeek WMAP operands")
    records = {}
    while offset < len(data):
        layer, expert, routing_weight = struct.unpack_from("<hhf", data, offset)
        key = (layer, expert)
        if (not 0 <= layer < LAYERS or not -1 <= expert < EXPERTS
                or not _finite(routing_weight) or routing_weight <= 0 or key in records):
            raise ValueError("invalid or duplicate DeepSeek WMAP expert record")
        records[key] = ExpertRecord(routing_weight, offset + 8 + 2 * NEURONS)
        offset += record_bytes
    return Operands(data, records)


@dataclass(frozen=True)
class WeightTerm:
    layer: int
    expert: int
    neuron: int
    projection: str
    feature: int

    def __post_init__(self):
        if not 0 <= self.layer < LAYERS or not -1 <= self.expert < EXPERTS:
            raise ValueError("expected layer 0–39 and expert -1 (shared) or 0–383")
        if not 0 <= self.neuron < NEURONS or not 0 <= self.feature < WIDTH:
            raise ValueError("expected neuron 0–2303 and feature 0–5119")
        if self.projection not in ("w1", "w2", "w3"):
            raise ValueError("projection must be w1, w2, or w3")

    @property
    def tensor(self):
        expert = "shared_experts" if self.expert == -1 else f"experts.{self.expert}"
        return f"layers.{self.layer}.ffn.{expert}.{self.projection}.weight"

    @property
    def coordinate(self):
        if self.projection == "w2":
            return [self.feature, self.neuron]
        return [self.neuron, self.feature]


def _term_arg(value):
    try:
        layer, expert, neuron, projection, feature = value.split(":")
        expert = -1 if expert == "shared" else int(expert)
        return WeightTerm(int(layer), expert, int(neuron), projection.lower(), int(feature))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LAYER:EXPERT:NEURON:W1|W2|W3:FEATURE") from error


class CaptureStore:
    """Read the explorer's public or local weight-research archive."""

    def __init__(self, root=DATA_URL, weight_api=WEIGHT_API, weight_files=()):
        self.root = root if isinstance(root, Path) else (
            root if root.startswith(("https://", "http://")) else Path(root)
        )
        self.weight_api = weight_api
        self.saved_weights = {}

        for file in weight_files:
            payload = json.loads(_read(Path(file)))
            if not isinstance(payload, dict):
                raise ValueError("weight JSON must contain a coefficient or an Explorer tile")
            for record in payload.get("values", [payload]):
                if not isinstance(record, dict):
                    raise ValueError("offline weights require full coefficient records from a tile")
                key = (record["tensor"], tuple(record["coordinate"]))
                self.saved_weights[key] = record

        self.catalog = self.get_json("research.json")
        if self.catalog.get("checkpoint_revision") != REVISION:
            raise ValueError("capture catalog is not the pinned DeepSeek 1017 revision")
        if self.catalog.get("real_model_measurements") is not True:
            raise ValueError("capture catalog does not identify real model measurements")

    def get_json(self, key):
        return json.loads(_read(_resource(self.root, key)))

    def coefficient(self, term):
        row, col = term.coordinate
        key = (term.tensor, (row, col))
        record = self.saved_weights.get(key)
        if record is None and self.weight_api:
            params = urlencode({"mode": "tile", "tensor": term.tensor, "row": row, "col": col,
                                "rows": 1, "cols": 1})
            tile = json.loads(_read(f"{self.weight_api}?{params}"))
            records = tile.get("values", [])
            if len(records) != 1:
                raise ValueError("weight lookup did not return exactly one coefficient")
            record = records[0]
        if record is None:
            raise ValueError(f"missing local coefficient for {term.tensor} {term.coordinate}")
        if (record.get("tensor") != term.tensor or record.get("coordinate") != term.coordinate
                or record.get("revision") != REVISION or not _finite(record.get("value"))):
            raise ValueError("coefficient value, address, or revision did not match the trace")
        self.saved_weights[key] = record
        return record

    def case(self, request):
        entry = next((x for x in self.catalog["requests"] if x["request_id"] == request), None)
        if entry is None:
            raise ValueError(f"unknown capture {request!r}; use --list-cases")
        case = self.get_json("cases/" + entry["caseFile"])
        if case.get("request_id") != request:
            raise ValueError("case file has a different request ID")
        return case

    def captured_operands(self, descriptor):
        data = _read(_resource(self.root, descriptor["file"]))
        if len(data) != descriptor["uncompressed_bytes"]:
            raise ValueError("operand length disagrees with the capture manifest")
        operands = decode_operands(data)
        if len(operands.records) != descriptor["records"]:
            raise ValueError("operand record count disagrees with the capture manifest")
        return operands


def _route_list(case, position):
    routes = case["routes"]
    if len(routes) != LAYERS:
        raise ValueError("expected 40 routing layers in DeepSeek 1017")
    result = []
    for route in routes:
        ids, weights = route["ids"][position], route["weights"][position]
        if (len(ids) != ROUTED_PER_LAYER or len(weights) != ROUTED_PER_LAYER
                or any(type(i) is not int or not 0 <= i < EXPERTS for i in ids)
                or len(set(ids)) != ROUTED_PER_LAYER or any(not _finite(w) or w <= 0 for w in weights)):
            raise ValueError("invalid DeepSeek 1017 routing data at this token position")
        result.append(dict(zip(ids, weights, strict=True)))
    return result


def _contribution(store, term, route, operands):
    weight = store.coefficient(term)
    result = {
        "layer": term.layer, "expert": term.expert, "neuron": term.neuron,
        "projection": term.projection, "feature": term.feature, "coefficient": weight,
    }
    if term.expert != -1 and term.expert not in route:
        return {**result, "coverage": "expert_not_routed", "local_contribution": 0.0}
    if operands is None or (term.layer, term.expert) not in operands.records:
        return {**result, "coverage": "operand_not_recorded", "local_contribution": None}
    if term.projection == "w2":
        value = operands.w2_input(term.layer, term.expert, term.neuron)
        source = "recorded_w2_input"
    else:
        value = operands.ffn_input(term.layer, term.feature)
        source = "recorded_ffn_input"
    return {**result, "coverage": "recorded_operand", "activation_source": source,
            "activation": value, "local_contribution": value * weight["value"]}


def trace_capture(store, case, position, terms=(), *, route_only=False, strongest=False, neuron_grid=None):
    """Join the recorded route and operands for one generated answer token."""
    prompt_count = len(case["prompt_ids"])
    generated, pieces = case["generated_ids"], case["pieces"]
    index = position - prompt_count
    if len(generated) != len(pieces) or not 0 <= index < len(generated):
        raise ValueError("position must identify a generated answer token in the saved capture")
    descriptor = case["tokens"].get(str(position))
    layers = _route_list(case, position)
    operands = store.captured_operands(descriptor) if descriptor and not route_only else None
    if operands is not None:
        for (layer, expert), record in operands.records.items():
            route = 1.0 if expert == -1 else layers[layer].get(expert)
            if route is None or not math.isclose(route, record.routing_weight, rel_tol=1e-6):
                raise ValueError(f"operand routing weight disagrees with layer {layer}, expert {expert}")
    timeline = []
    for layer, route in enumerate(layers):
        experts = [{"expert": i, "routing_weight": w} for i, w in route.items()]
        experts.append({"expert": -1, "routing_weight": 1.0, "shared": True})
        for expert in experts:
            identifier = expert["expert"]
            recorded = operands is not None and (layer, identifier) in operands.records
            expert["w2_inputs_recorded"] = recorded
            if recorded and strongest:
                expert["strongest_w2_input"] = operands.strongest_w2(layer, identifier)
        timeline.append({"layer": layer, "stage": "encoder" if layer < ENCODER_LAYERS else "decoder",
                         "attention": "sliding_window" if layer < 2 else "csa2_with_sliding_window",
                         "experts": experts})
    grid = None
    if neuron_grid is not None and operands is not None:
        layer, expert = neuron_grid
        if (layer, expert) in operands.records:
            grid = {"layer": layer, "expert": expert, "side": 48,
                    "w2_inputs": [operands.w2_input(layer, expert, index) for index in range(NEURONS)]}
    following = ({"id": generated[index + 1], "piece": pieces[index + 1]}
                 if index + 1 < len(generated) else None)
    return {
        "position": position, "answer_token_index": index, "token": {"id": generated[index], "piece": pieces[index]},
        "next_recorded_token": following,
        "context": {"before": "".join(pieces[max(0, index - 8):index]),
                    "after": "".join(pieces[index + 1:index + 9])},
        "embedding_width": WIDTH,
        "layers": timeline,
        **({"neuron_grid": grid} if grid else {}),
        "weight_terms": [_contribution(store, term, layers[term.layer], operands) for term in terms],
        "coverage": {"routing_layers": len(timeline), "operand_file": descriptor["file"] if descriptor else None,
                     "loaded_expert_records": len(operands.records) if operands else 0,
                     "missing_detail": ["attention internals", "vocabulary logits", "effects of intervening"]},
    }


def replay_sequence(store, case, positions, terms=(), *, route_only=False, strongest=False, neuron_grid=None):
    """Return a sequence report from saved routes, loading operands only for selected positions."""
    pieces = case["pieces"]
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    start = len(case["prompt_ids"])
    detailed = sorted(map(int, case["tokens"]))
    detailed_set = set(detailed)
    tokens = tokens_for_sequence(zip(case["generated_ids"], pieces, strict=True), "".join(pieces), offsets)
    for token in tokens:
        token["position"] += start
        token["segment"] = "recorded answer"
        token["details"] = token["position"] in detailed_set
    selected = list(positions)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select at least one unique answer position")
    return {
        "schema": "token-trace/sequence/1", "kind": "capture", "mode": "replay",
        "model": "DeepSeek V4.1 Flash 1017", "revision": REVISION,
        "request_id": case["request_id"], "title": case["title"], "input": case["prompt"],
        "generated": case["answer"], "tokens": tokens, "detailed_positions": detailed,
        "layer_names": [str(layer) for layer in range(LAYERS)],
        "capture_layout": case.get("capture_layout"), "parity_checked": case.get("parity_checked"),
        "restore_verified": case.get("restore_verified"),
        "traces": [trace_capture(store, case, position, terms, route_only=route_only,
                                strongest=strongest, neuron_grid=neuron_grid) for position in selected],
    }


def _grid_arg(value):
    try:
        layer, expert = value.split(":")
        layer, expert = int(layer), -1 if expert == "shared" else int(expert)
        if not 0 <= layer < LAYERS or not -1 <= expert < EXPERTS:
            raise ValueError
        return layer, expert
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LAYER:EXPERT with layer 0–39 and expert 0–383 or shared") from error


def describe_capture():
    return {
        "purpose": "Read saved DeepSeek 1017 routes and join recorded operands to saved coefficients.",
        "operations": {
            "inspect": ["--describe", "--list-cases"],
            "positions": ["--case REQUEST", "--position N (repeatable)", "--all-tokens", "--recorded-only"],
            "measurements": ["--route-only", "--strongest", "--neuron-grid LAYER:EXPERT",
                             "--term LAYER:EXPERT:NEURON:W1|W2|W3:FEATURE (repeatable)"],
            "offline": ["--offline", "--data-root PATH", "--weight-file PATH (repeatable)"],
            "report": ["--report-format capture", "--report-format sequence"],
        },
        "schemas": {"capture": "token-trace/deepseek-capture/1",
                    "sequence": "token-trace/sequence/1", "cases": "JSON array"},
        "io_flags": ["--output PATH", "--compact", "--json-errors"],
        "examples": [
            ["token-trace-deepseek", "--case", "wm-00-3d", "--all-tokens", "--route-only",
             "--report-format", "sequence", "--compact"],
            ["token-trace-deepseek", "--position", "135", "--position", "171",
             "--term", "24:34:1076:w2:647", "--neuron-grid", "24:34", "--json-errors"],
        ],
    }


def main(argv=None):
    parser = AgentParser(description=__doc__, argv=argv)
    parser.add_argument("--describe", action="store_true", help="emit command capabilities without fetching a capture")
    parser.add_argument("--data-root", default=DATA_URL, help="URL or directory containing research.json")
    parser.add_argument("--weight-api", default=WEIGHT_API, help="Explorer coefficient endpoint")
    parser.add_argument("--weight-file", type=Path, action="append", default=[], help="saved coefficient or tile JSON")
    parser.add_argument("--offline", action="store_true", help="use local capture and coefficient files only")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case", default=FEATURED_CASE)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--position", type=int, action="append", help="absolute token position, repeat to compare")
    scope.add_argument("--all-tokens", action="store_true", help="trace every answer token in the request")
    scope.add_argument("--recorded-only", action="store_true", help="trace every answer position with saved operands")
    parser.add_argument("--term", type=_term_arg, action="append", default=[])
    parser.add_argument("--route-only", action="store_true", help="skip WMAP operands")
    parser.add_argument("--strongest", action="store_true", help="include strongest recorded neuron per expert")
    parser.add_argument("--neuron-grid", type=_grid_arg, metavar="LAYER:EXPERT",
                        help="include every recorded W2 input for one expert")
    parser.add_argument("--report-format", choices=("capture", "sequence"), default="capture",
                        help="legacy capture schema or shared sequence schema")
    parser.add_argument("-o", "--output", type=Path)
    add_output_options(parser)
    args = parser.parse_args()
    if args.describe:
        from .cli import describe

        try:
            write_json(describe(), args.output, compact=args.compact)
        except OSError as error:
            parser.error(str(error), code="output_failed")
        return
    if args.offline and args.data_root.startswith(("http://", "https://")):
        parser.error("--offline needs a local --data-root")
    if args.route_only and (args.term or args.strongest or args.neuron_grid):
        parser.error("--route-only cannot be combined with --term, --strongest, or --neuron-grid")
    if args.neuron_grid and not args.position:
        parser.error("--neuron-grid requires at least one explicit --position")

    try:
        store = CaptureStore(args.data_root, None if args.offline else args.weight_api, args.weight_file)
        if args.list_cases:
            report = store.catalog["requests"]
        else:
            case = store.case(args.case)
            detailed = sorted(map(int, case["tokens"]))
            start = len(case["prompt_ids"])
            default = 171 if args.case == FEATURED_CASE else next(iter(detailed), start)
            if args.all_tokens:
                positions = range(start, start + len(case["generated_ids"]))
            elif args.recorded_only:
                positions = detailed
            else:
                positions = args.position or [default]
            sequence = replay_sequence(store, case, positions, args.term, route_only=args.route_only,
                                       strongest=args.strongest, neuron_grid=args.neuron_grid)
            if args.report_format == "sequence":
                report = sequence
            else:
                report = {
                    "schema": "token-trace/deepseek-capture/1",
                    "checkpoint_revision": REVISION,
                    "request_id": args.case, "title": sequence["title"],
                    "prompt": sequence["input"], "answer": sequence["generated"],
                    "capture_layout": sequence["capture_layout"], "parity_checked": sequence["parity_checked"],
                    "restore_verified": sequence["restore_verified"], "detailed_positions": detailed,
                    "tokens": sequence["traces"],
                }
    except (ValueError, KeyError, IndexError, TypeError, OSError, EOFError) as error:
        parser.error(str(error), code="capture_failed")
    try:
        write_json(report, args.output, compact=args.compact)
    except (OSError, ValueError) as error:
        parser.error(str(error), code="output_failed")


if __name__ == "__main__":
    main()
