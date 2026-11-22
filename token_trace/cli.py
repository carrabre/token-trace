import argparse
import json
from pathlib import Path

import torch

from .core import trace
from .hf import Row, hf_plan


def main():
    parser = argparse.ArgumentParser(description="Trace one token in a local Hugging Face forward pass")
    parser.add_argument("model", help="local checkpoint directory or Hugging Face model ID")
    parser.add_argument("text")
    parser.add_argument("--position", type=int, default=-1, help="zero-based token position, default last")
    parser.add_argument("--layers", help="explicit dotted path to the block list")
    parser.add_argument("--revision", help="pin a Hub checkpoint revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top", type=int, default=5, help="number of contribution terms per row")
    parser.add_argument("--row", action="append", default=[], metavar="MODULE:OUTPUT[:INPUT]",
                        help="repeat to select an affine row, weight term, or embedding feature")
    parser.add_argument("--all-terms", action="store_true", help="emit every coefficient in each selected row")
    parser.add_argument("--full-vectors", action="store_true", help="include selected token's layer vectors")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        parser.error("install Hugging Face support with pip install '.[hf]'")
    rows = {}
    for spec in args.row:
        name, *coordinates = spec.split(":")
        if not name or len(coordinates) not in (1, 2):
            parser.error("use --row MODULE:OUTPUT[:INPUT]")
        try:
            pair = [int(index) for index in coordinates]
        except ValueError:
            parser.error("weight coordinates must be integers")
        rows.setdefault(name, []).append(Row(*pair))

    options = {"revision": args.revision, "local_files_only": args.local_files_only,
               "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModelForCausalLM.from_pretrained(args.model, **options).to(args.device).eval()
    inputs = tokenizer(args.text, return_tensors="pt").to(args.device)
    ids = inputs.pop("input_ids")
    try:
        plan, skipped = hf_plan(model, layer_path=args.layers, rows=rows,
                                top=args.top, all_terms=args.all_terms)
        data = trace(model, plan, ids, args.position, full_vectors=args.full_vectors,
                     forward_kwargs={**inputs, "use_cache": False})
    except ValueError as error:
        parser.error(str(error))

    def piece(token_id):
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

    data.update(model=args.model, revision=args.revision,
                tokens=[{"id": i, "text": piece(i)} for i in ids[0].tolist()])
    for prediction in data.get("prediction", []):
        prediction["text"] = piece(prediction["token_id"])
    data["coverage"]["modules_without_affine_adapter"] = skipped
    output = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
