"""Local-model sequence workflow for the CLI and Python callers."""
from functools import lru_cache

import torch

from .core import Plan, trace_many
from .hf import Row, hf_plan
from .sequence import tokens_for_sequence


def parse_rows(specs):
    rows = {}
    for spec in specs:
        name, *coordinates = spec.split(":")
        if not name or len(coordinates) not in (1, 2):
            raise ValueError("Use MODULE:OUTPUT[:INPUT] for each weight selection.")
        try:
            pair = [int(index) for index in coordinates]
        except ValueError:
            raise ValueError("Weight coordinates must be integers.") from None
        rows.setdefault(name, []).append(Row(*pair))
    return rows


@lru_cache(maxsize=1)
def load_model(name, revision, device, local_only):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    options = {"revision": revision, "local_files_only": local_only, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(name, **options)
    model = AutoModelForCausalLM.from_pretrained(name, **options).to(device).eval()
    return tokenizer, model


class GenerationObserver:
    """Receive one greedy generation's new IDs without retokenizing its text."""

    def __init__(self, tokenizer, emit):
        self.tokenizer = tokenizer
        self.emit = emit
        self.prompt_seen = False
        self.ids = []

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.ids.extend(value.reshape(-1).tolist())
        self.emit("generation", text=self.tokenizer.decode(self.ids, skip_special_tokens=True),
                  count=len(self.ids))

    def end(self):
        pass


def model_sequence(name, text, *, positions=None, revision=None, device="cpu", local_only=False,
                   layer_path=None, rows=None, top=5, all_terms=False, full_vectors=False,
                   depth="weights", generate=0, max_tokens=None, emit=None):
    """Trace all selected positions in context in one forward pass."""
    if not text.strip():
        raise ValueError("Enter a prompt or passage.")
    if depth not in {"layers", "selected", "weights"}:
        raise ValueError("Unknown trace depth.")
    if not 0 <= generate <= 128:
        raise ValueError("Generate between 0 and 128 new tokens.")
    if emit is None:
        def emit(*args, **kwargs):
            pass

    emit("status", message="Loading tokenizer and model")
    tokenizer, model = load_model(name, revision, device, local_only)
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=tokenizer.is_fast)
    offsets_tensor = encoded.pop("offset_mapping", None)
    offsets = offsets_tensor[0].tolist() if offsets_tensor is not None else None
    # AutoModelForCausalLM expects only the model-facing inputs named by its tokenizer.
    encoded = encoded.to(device)
    prompt_count = int(encoded["input_ids"].shape[1])
    if max_tokens is not None and prompt_count + generate > max_tokens:
        raise ValueError(f"Request would exceed the workspace limit of {max_tokens} tokens.")
    if depth == "selected" and not rows:
        raise ValueError("Add at least one MODULE:OUTPUT[:INPUT] selection for Selected weights.")
    plan, skipped = hf_plan(model, layer_path=layer_path, rows=rows, top=top, all_terms=all_terms)
    if depth != "weights":
        requested = {name for name in rows or {}}
        probes = [probe for probe in plan.probes
                  if any(key in requested for key in (probe.name, probe.name + ".gate", probe.name + ".up"))]
        plan = Plan(plan.layers, probes if depth == "selected" else [])

    def piece(token_id):
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

    if generate:
        emit("status", message="Generating answer")
        observer = GenerationObserver(tokenizer, emit)
        with torch.inference_mode():
            ids = model.generate(**encoded, max_new_tokens=generate, do_sample=False, streamer=observer)
        generated = ids[0, prompt_count:].tolist()
        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        # Preserve the original token IDs and replay the full context. Padding is unused.
        options = {"attention_mask": torch.ones_like(ids), "use_cache": False}
    else:
        ids = encoded.pop("input_ids")
        output_text = None
        options = {**encoded, "use_cache": False}
    tokens = tokens_for_sequence([(i, piece(i)) for i in ids[0].tolist()], text, offsets,
                                 prompt_count, tokenizer.all_special_ids)
    layer_names = [layer.name if hasattr(layer, "name") else layer[0] for layer in plan.layers]
    emit("sequence", tokens=tokens, layers=layer_names, input=text, generated=output_text,
         mode="replay" if generate else "forward")
    emit("status", message="Tracing the generated sequence" if generate else "Tracing the input sequence")
    report = trace_many(
        model, plan, ids, positions, full_vectors=full_vectors, forward_kwargs=options,
        on_layer=lambda name, values: emit("layer", name=name, values=values),
    )
    for trace in report["traces"]:
        for prediction in trace.get("prediction", []):
            prediction["text"] = piece(prediction["token_id"])
        trace["coverage"]["modules_without_affine_adapter"] = skipped
    report.update(model=name, revision=revision or getattr(model.config, "_commit_hash", None),
                  kind="local", mode="replay" if generate else "forward", input=text,
                  generated=output_text, tokens=tokens, layer_names=layer_names)
    return report
