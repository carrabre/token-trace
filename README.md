# Token Trace

Trace one token or a full passage through an instrumentable model. The CLI emits JSON for layer activations, supported weight projections, expert dispatch and next-token logits. Use the Python API to attach probes to a custom runtime, or analyze published DeepSeek captures on a CPU.

| Task | Entry point |
| --- | --- |
| Map a checkpoint before running a trace | [Inspect the model](#inspect-a-model) |
| Trace a local checkpoint or a Hugging Face model | [Local model commands](#trace-a-model) |
| Explore a saved DeepSeek answer or a specific weight | [DeepSeek replay commands](#replay-deepseek) |
| Instrument a runtime or a packed expert | [Python adapters](#add-an-adapter) |
| Verify the published activation-swap findings | [Offline score verifier](examples/interventions/README.md) |

## Install

Use Python 3.10 or later in a virtual environment. From the repository root:

```sh
python -m pip install -e '.[hf]'
```

Install with `python -m pip install -e .` for the Python core and DeepSeek replay without the optional Transformers dependency. The local-model CLI defaults to CPU and loads Hugging Face checkpoints with remote custom code disabled.

## Inspect a model

Start with a checkpoint inventory to find module and tensor names, plausible layer stacks, and coordinate bounds. This loads the model without running inference and requires no prompt.

```sh
token-trace hf-internal-testing/tiny-random-gpt2 --inspect-model \
  --local-files-only --compact -o model.json
```

The `token-trace/model/1` report describes the registered PyTorch module hierarchy, parameters, buffers, and candidate layer stacks. Use `layer_stack_candidates` to find paths for `--layers`. Tensor entries include shapes and dtypes. Where a module or tensor object has several registered names, its later entries point to its first registered name via `alias_of`. Functional operations inside a module and token-specific execution require a forward trace or a runtime probe.

To read a particular coefficient, repeat `--coefficient PARAMETER:INDEX[,INDEX...]` with the parameter name and its tensor coordinates. Use `PARAMETER:` for a scalar. Read the shape in the inventory before choosing an address.

```sh
token-trace hf-internal-testing/tiny-random-gpt2 --inspect-model \
  --coefficient transformer.h.0.mlp.c_fc.weight:0,0 --local-files-only
```

Read the returned address and value in `coefficients`. Parameter coordinates use the loaded tensor's axis order. Trace rows use logical projection output and input indices. For Hugging Face GPT-2's `Conv1D`, the loaded weight axes are `[input, output]`. The scalar is read at the runtime dtype. Quantized, sharded, or otherwise inaccessible representations can require a reader that knows their format.

For an already loaded PyTorch model, use `inspect_model(model)` and `parameter_value(model, name, coordinate)` from `token_trace.inventory`. Use an empty coordinate tuple for a rank-zero parameter.

## Automate a trace

Discover supported commands and report schemas without loading a model. The descriptor uses `token-trace/describe/1` and supports the same output flags as a trace.

```sh
token-trace --describe --compact
token-trace-deepseek --describe --compact
```

Both commands write the JSON result to stdout by default. Use `--compact` for single-line JSON. With `--output PATH`, the result goes to the file and stdout is empty.

With `--json-errors`, a failure exits with status 2 and writes a `token-trace/error/1` object to stderr containing `event: "error"`, `code`, and `message`. For local-model traces, `--progress` sends one JSON object per line to stderr with schema `token-trace/progress/1` and event `status`, `generation`, `sequence`, or `layer`. A caller using both flags can distinguish progress and errors by the schema. Capture stdout and stderr separately.

```sh
printf '%s' 'First sentence. Second sentence.' |
  token-trace hf-internal-testing/tiny-random-gpt2 --text-file - \
    --all-tokens --depth layers --local-files-only --compact --json-errors
```

## Trace a model

Use a local checkpoint directory or a Hugging Face model ID. In the examples, `hf-internal-testing/tiny-random-gpt2` is a small test checkpoint. `--local-files-only` uses an existing model directory or cache. Omit that option when downloading is wanted, and pass `--revision` to pin a Hub revision.

**Whole passage, layer summaries**

```sh
token-trace hf-internal-testing/tiny-random-gpt2 "Three.js draws a mesh. Its material controls shading." \
  --all-tokens --depth layers --local-files-only -o passage.json
```

**Selected positions and one exact weight**

```sh
token-trace hf-internal-testing/tiny-random-gpt2 "A Three.js mesh combines geometry and material." \
  --position 1 --position -1 --depth selected \
  --row transformer.h.0.mlp.c_fc:2:1 --local-files-only -o selected.json
```

**A passage from a file and a generated continuation**

```sh
token-trace hf-internal-testing/tiny-random-gpt2 --text-file passage.txt \
  --generate 4 --all-tokens --depth layers --local-files-only --progress -o continuation.json
```

- Positions are zero-based in the model's token sequence. Negative positions count from the end. Repeated `--position` traces the selected tokens in one pass. `--all-tokens` observes every position in one pass. The default selects the last token.
- `--depth layers` avoids weight probes. `--depth selected` uses explicit `--row` selections. `--depth weights` observes supported projections with output row zero by default.
- `--row MODULE:OUTPUT[:INPUT]` selects an output coordinate and optionally an exact input coefficient. Repeat it for other rows or modules. For an embedding, use `--row MODULE:FEATURE` to read the saved value for the token's lookup index.
- `--top` limits reported terms by magnitude and defaults to five. `--all-terms` emits all input-weight terms in each requested row. `--full-vectors` includes layer input and output vectors. Output grows with the number of positions, probes, rows and terms selected.
- `--generate N` first generates up to N tokens greedily, then traces the exact prompt-plus-continuation IDs in one forward pass. `--device` selects the execution device. Repeat `--layers PATH` to specify the block stacks when discovery is ambiguous or when multiple stacks share the selected token axis.

### Read the JSON

| Schema | Contents |
| --- | --- |
| `token-trace/model/1` | Static module and tensor inventory and requested checkpoint coefficients. |
| `token-trace/1` | Single local position with `layers`, `probes`, `coverage`, and top next-token logits when available. The CLI uses this for one position. |
| `token-trace/sequence/1` | A `tokens` index and per-position `traces` for the chosen scope. Local CLI reports include model, revision, input and execution mode. DeepSeek's sequence view indexes the recorded answer, retains archive provenance, and stores capture records in `traces`. |
| `token-trace/deepseek-capture/1` | Saved DeepSeek request with absolute answer positions, recorded routes, following recorded tokens, and requested weight terms. The capture catalog command returns an array. |

Join local records by `position`. Within a forward pass, use `event_index` to merge layer and probe records in their observed hook order, including repeated module calls. A record for position i scores the token after i when the model exposes next-token logits. `tokens` entries in sequence reports mark input and generated tokens. Input entries may include tokenizer offsets and sentence numbers. Sentence grouping uses punctuation and newline heuristics for analysis while the model receives the complete sequence.

Read `coverage` alongside results. It reports missing layers, unexecuted probes, and observations whose token axes could not be mapped. Modules without an affine adapter are listed separately. Dense terms use an observed input times a saved coefficient. Where available, the report adds bias and compares the reconstruction with the measured scalar. Precision and accumulation can cause small differences.

Weight support includes batch-first PyTorch `nn.Linear`, Hugging Face `Conv1D`, standard embedding lookups, and the supported packed Mixtral projections. Mixtral records dispatch and routing multipliers and replays its gate, up, and down projections from the observed input. The JSON identifies replayed operands and compares the replayed mixture with the recorded output. Use a row such as `model.layers.0.mlp.experts:2:1` for the down projection, or append `.gate` or `.up` to the module name for the input projections.

A scalar product measures a local computation. A separate controlled intervention is required to measure an effect on the final prediction.

## Replay DeepSeek

The capture adapter reads saved DeepSeek V4.1 Flash 1017 records and resolves coefficients for the pinned checkpoint. Replay uses existing records and needs no GPU or checkpoint download.

**Discover requests and read the full answer's routes**

```sh
token-trace-deepseek --list-cases
token-trace-deepseek --case wm-00-3d --all-tokens --route-only \
  --report-format sequence --compact -o deepseek-routes.json
```

**Read exact terms across two token positions**

```sh
token-trace-deepseek --case wm-00-3d --position 135 --position 171 \
  --term 24:34:1076:w2:647 --term 24:34:1076:w1:32 --term 24:34:1076:w3:32 \
  --strongest -o deepseek-weights.json
```

The term format is `LAYER:EXPERT:NEURON:W1|W2|W3:FEATURE`. Use `shared` or `-1` for the shared expert. W1 and W3 use `[neuron, input feature]` checkpoint coordinates. W2 uses `[output feature, neuron]`. At layer 24, expert 34, W2 coordinate `[647, 1076]` contributes `+0.00732421875` at position 135 and `−0.01025390625` at position 171. The W2 input already incorporates the routing multiplier.

Use `--recorded-only` to select every saved answer position with detailed operands. Use `--position 171 --neuron-grid 24:34` to retrieve all 2,304 recorded W2 inputs for layer 24, expert 34 at that position. The grid is available for selected positions and is incompatible with route-only runs. Add `--report-format sequence` for a token index alongside the per-position capture records. The default retains the capture report.

The archive reports router choices for saved answer positions. Detailed operands exist for selected positions, collected from a teacher-forced replay of the recorded answer. The result distinguishes a missing operand from an inactive expert. Attention internals, vocabulary logits, and causal effects have separate evidence requirements. Exact continuation tokens in this format come from the archive.

Use the [offline capture walkthrough](examples/deepseek-1017.md#use-an-offline-copy) to assemble replay files. Run with `--offline --data-root capture` and pass saved coefficient records via repeatable `--weight-file`. See the [DeepSeek method](docs/deepseek-method.md) for replay provenance, quantization, and coordinate mapping.

## Add an adapter

For a custom PyTorch model with a shared batch-first token axis, use the framework directly. If the model has a unique discoverable block stack, `module_plan(model)` can pick it. For a known stack, pass its dotted module path explicitly.

```python
from token_trace import trace_many
from token_trace.portable import Row, module_plan

plan, skipped = module_plan(
    model,
    layer_paths=("blocks",),
    rows={"blocks.0.proj": [Row(output=0, input=1)]},
    default_probes=False,
)
report = trace_many(model.eval(), plan, input_ids, positions=[0, -1])
```

Use `discover_stacks(model)` to list candidate paths. For additional module types with known layouts, pass `probe_adapters` to `module_plan`. Each adapter receives `(model, name, module, pending_rows)` and the keyword options `top`, `all_terms`, and `default_probes`. A matching adapter consumes its handled row requests and returns a `Probe` or `False` to skip the probe. Returning `None` lets the next adapter or the generic probes try the module. [Hugging Face's adapters](token_trace/hf.py) implement this contract.

For a model whose execution or layout needs custom alignment, build a plan with specific layer readers and probes.

```python
from token_trace import Plan, Probe, linear_terms, token_row, trace_many

def read_projection(call):
    signal = token_row(call.args[0], call.position, call.length)
    output = token_row(call.output, call.position, call.length)
    if signal is None or output is None:
        return None
    return linear_terms(signal, call.module.weight, 0,
                        bias=call.module.bias, measured=output)

plan = Plan(
    [(f"blocks.{i}", block) for i, block in enumerate(model.blocks)],
    [Probe("blocks.0.proj", model.blocks[0].proj, read_projection)],
)
report = trace_many(model.eval(), plan, input_ids, positions=[0, -1])
```

The hooks run in one uncompiled PyTorch forward pass with batch size one. Use one trace at a time per model instance. The runtime removes hooks when the pass finishes or raises an error. `on_layer` receives a layer name and token summaries in execution order.

Use `Layer` with a custom reader for nonstandard input and output layouts. Use `runner` and `logits_reader` for alternate forward signatures and result formats. Use `TracePoint` to expose values and metadata near functional projections and `packed_token_row` to join a sequence position to a dispatched expert row. Packed token positions can differ from the original sequence order. Run the [packed expert example](examples/packed_moe.py) with `python -m examples.packed_moe`.

| Runtime | How to trace it |
| --- | --- |
| Hugging Face causal model with an accessible block stack | Start with model inspection and trace layer summaries. Add exact weight terms for supported projections. |
| Custom PyTorch model whose layers share a batch-first token axis | Use `module_plan` or pick explicit layers and probes. For recurrent models, repeated calls appear as separate events when the probes can identify the token at each call. |
| Encoder–decoder, multimodal, or packed expert model with multiple token layouts | Supply custom readers using the runtime's alignment or dispatch metadata. Choose the token axis the trace represents and map other axes explicitly. |
| Fused or quantized operation | Supply a probe or runtime tap exposing the operands and a reader that understands the weight format. |
| Hosted API or another execution engine | Export or adapt traces at the runtime boundary when it exposes the required execution data. The PyTorch hooks require access to the running model. |

## Validate

```sh
python -m unittest discover -s tests -v
python -m examples.packed_moe
python examples/interventions/replay_scores.py
```

These checks run on CPU. The tests instantiate small models and exercise multiple positions in one forward pass, token mapping, affine and expert replay, and hook cleanup. The intervention example checks published results using saved logits.
