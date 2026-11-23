"""Local-model sequence workflow for the CLI and Python callers."""
from functools import lru_cache, wraps
from threading import RLock

import torch

from .core import trace_many
from .hf import Row, hf_plan
from .sequence import tokens_for_sequence

_SESSION_LOCK = RLock()


def _serialized(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        with _SESSION_LOCK:
            return fn(*args, **kwargs)
    return run


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


def inspect_checkpoint(name, *, revision=None, device="cpu", local_only=False, coefficients=()):
    """Inspect a locally loaded HF topology without training data, inputs or a tokenizer."""
    import transformers
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    from .inventory import inspect_model, parameter_value
    from .portable import discover_stacks

    options = {"revision": revision, "local_files_only": local_only, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(name, **options)
    # Respect installed, built-in task heads declared by the checkpoint.
    loader = None
    for architecture in config.architectures or ():
        candidate = getattr(transformers, architecture, None)
        if isinstance(candidate, type) and issubclass(candidate, transformers.PreTrainedModel):
            loader = candidate
            break
    if loader is None:
        task = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
        loader = task if type(config) in task._model_mapping else AutoModel
    model = loader.from_pretrained(name, config=config, **options).to(device)
    report = inspect_model(model)
    report.update(model=name, revision=getattr(config, "_commit_hash", None) or revision,
                  requested_revision=revision, layer_stack_candidates=discover_stacks(model),
                  coefficients=[parameter_value(model, key, indices) for key, indices in coefficients])
    return report


def parse_coefficients(specs):
    for spec in specs:
        name, separator, indices = spec.partition(":")
        if not name or not separator:
            raise ValueError("Use PARAMETER:INDEX[,INDEX...] for each coefficient.")
        try:
            coordinate = tuple(map(int, indices.split(","))) if indices else ()
        except ValueError:
            raise ValueError("Coefficient coordinates must be integers.") from None
        yield name, coordinate


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


@_serialized
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
    plan, skipped = hf_plan(model, layer_path=layer_path, rows=rows if depth != "layers" else None,
                            top=top, all_terms=all_terms, default_probes=depth == "weights")

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
