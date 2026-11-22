# Token Trace

Follow one token through a model you can run and inspect. Token Trace records its layer activations, identifies the projections that ran, and shows how individual saved weights contributed to selected outputs. The command-line tool runs one forward pass and writes JSON.

## Choose a starting point

| Goal | Start here |
| --- | --- |
| Trace a local Hugging Face model | Run the CLI below to inspect embeddings, layer activations, projection weights, and next-token logits. |
| Reproduce the DeepSeek walkthrough | Run the saved-capture examples below. They use the same pinned weights and recorded operands as Neuron Flow. |
| Instrument a different runtime | Use the adapter API below. A worked packed-expert adapter is in [examples/packed_moe.py](examples/packed_moe.py). |
| Check the activation-swap findings | Run the [recorded score verifier](examples/interventions/README.md) on a CPU. |

## Run a trace

Install Python 3.10 or later and create a virtual environment. From this repository:

```bash
python -m pip install '.[hf]'

token-trace hf-internal-testing/tiny-random-gpt2 "A Three.js Mesh combines geometry and material." \
  --position -1 --row transformer.h.0.mlp.c_fc:2:1 --all-terms -o trace.json
```

The example model is a tiny GPT-2 checkpoint for testing. Use a model ID or a local checkpoint directory for research. Pass `--revision` to pin a Hub revision and `--local-files-only` to restrict loading to local files. The CLI loads checkpoints with remote custom code disabled. It defaults to CPU.

- `--position` selects a zero-based token position. Negative positions count from the end. The report lists the token IDs and their decoded pieces.
- `--row MODULE:OUTPUT[:INPUT]` selects an output coordinate and optionally one exact weight. Repeat it to inspect other rows or modules. Each supported projection defaults to output coordinate 0. For a standard embedding, use `--row MODULE:FEATURE` to inspect the saved value for the token’s actual lookup index.
- `--all-terms` includes every input-weight term for each selected output. The default returns the five largest terms by magnitude. Use `--full-vectors` to include the selected token's layer input and output vectors.
- `--layers PATH` names a block list when discovery cannot identify it. For example, `--layers model.layers`.

## Read the report

The report lists blocks in execution order, observations from projection and expert probes, and the top next-token logits at the chosen position when available. A forward pass includes the context supplied in the input. For an autoregressive model, the logits at position (i) score the token after position (i).

A weight term is the observed input multiplied by the corresponding saved coefficient. The report adds bias, compares the reconstructed scalar with the model's output, and records the numerical difference. Precision and accumulation order can produce small differences. The term measures a local computation. Measuring an effect on the final prediction requires a separate intervention.

Dense support covers batch-first PyTorch `nn.Linear` and Hugging Face `Conv1D` projections. The packed Mixtral adapter records selected experts and routing weights and replays the gate, up, and down projections from the observed dispatch input. It compares the replayed mixture with the recorded mixture output. Use `--row model.layers.0.mlp.experts:2:1` for the down projection, or add `.gate` or `.up` after `experts` for either input projection. Inner operands for packed experts are identified as replayed in JSON.

Coverage reports missing layers, probes that never ran, and observations whose token axis the adapter could not map. A projection may run for every token in a sequence, while an expert may run for a subset. Adapters map the token to the runtime dispatch row before interpreting a packed tensor.

## Reproduce the DeepSeek trace

After installation, run this public-capture example from the repository root:

```bash
token-trace-deepseek --position 135 --position 171 \
  --term 24:34:1076:w2:647 --term 24:34:1076:w1:32 --term 24:34:1076:w3:32 \
  --strongest -o deepseek-trace.json
```

See [more DeepSeek examples](examples/deepseek-1017.md) for listing all saved requests, offline replay, and routing-only positions. This command reads the published traces and checkpoint coefficient API. The positions are absolute in the saved sequence. Position 171 is token `94175` (` THREE`), followed in the recorded answer by `7342` (`.B`), the start of `.BoxGeometry`. The report identifies the 20 causal encoder layers and 20 decoder layers, six selected experts and one shared expert per layer, and the requested W1, W3, and W2 contributions. W2 coefficient `[647, 1076]` in layer 24, expert 34 contributes `+0.00732421875` at position 135 and `−0.01025390625` at position 171.

Read the [DeepSeek tracing method](docs/deepseek-method.md) for the original capture pipeline, quantization details, stored signals, and the exact layer-to-weight mapping. The saved request was replayed with the answer tokens supplied. Use the explicit adapter API for a fresh forward pass. Replaying a saved trace requires no checkpoint download or GPU.

## Add an adapter

The Python API takes an explicit layer list and any probes that can interpret a runtime call:

```python
from token_trace import Plan, Probe, linear_terms, token_row, trace

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
report = trace(model.eval(), plan, input_ids, position=-1)
```

For a custom MoE, observe its router and map the selected sequence position to the physical row inside the expert with `packed_token_row`. `TracePoint` is a passthrough module you can place next to functional projections or fused operations to expose values and dispatch indices. The [worked adapter](examples/packed_moe.py) shows the same token reaching two experts at different packed rows, and verifies the exact local contributions. Run it from the repository root with `python -m examples.packed_moe`.

For nonstandard layouts, use `Layer` with a custom reader for token input and output vectors. The `runner` and `logits_reader` arguments to `trace` let an adapter specify a custom forward signature and result format. At opaque or quantized kernel boundaries, document which operands were captured and which were reconstructed. Hosted APIs that only return tokens or logits cannot supply internal traces. The built-in adapter does not run the original DeepSeek 1017 runtime or trace arbitrary fused attention, quantization kernels, or encoder–decoder token axes.

The hooks run during a single, uncompiled PyTorch forward pass with a batch size of one. Use one trace at a time per model instance. The library removes all hooks when the run finishes or raises an error.

## Tests

```bash
python -m pip install -e '.[hf]'
python -m unittest discover -s tests -v
```

The tests instantiate small GPT-2, Llama, and Mixtral models locally. They check layer and token coverage, affine reconstruction, router dispatch, expert replay, and hook cleanup.
