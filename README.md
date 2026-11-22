# Token Trace

Follow one token through a model you can run and inspect. Token Trace records its layer activations, identifies the projections that ran, and shows how individual saved weights contributed to selected outputs. The command-line tool runs one forward pass and writes JSON.

## Run a trace

Install Python 3.10 or later and create a virtual environment. From this repository:

```bash
python -m pip install '.[hf]'

token-trace hf-internal-testing/tiny-random-gpt2 "A Three.js Mesh combines geometry and material." \
  --position -1 --row transformer.h.0.mlp.c_fc:2:1 --all-terms -o trace.json
```

The example model is a tiny GPT-2 checkpoint for testing. Use a model ID or a local checkpoint directory for research. Pass `--revision` to pin a Hub revision and `--local-files-only` to restrict loading to local files. The CLI loads checkpoints with remote custom code disabled. It defaults to CPU.

- `--position` selects a zero-based token position. Negative positions count from the end. The report lists the token IDs and their decoded pieces.
- `--row MODULE:OUTPUT[:INPUT]` selects an output coordinate and optionally one exact weight. Repeat it to inspect other rows or modules. Each supported projection defaults to output coordinate 0.
- `--all-terms` includes every input-weight term for each selected output. The default returns the five largest terms by magnitude. Use `--full-vectors` to include the selected token's layer input and output vectors.
- `--layers PATH` names a block list when discovery cannot identify it. For example, `--layers model.layers`.

## Read the report

The report lists blocks in execution order, observations from projection and expert probes, and the top next-token logits at the chosen position when available. A forward pass includes the context supplied in the input. For an autoregressive model, the logits at position (i) score the token after position (i).

A weight term is the observed input multiplied by the corresponding saved coefficient. The report adds bias, compares the reconstructed scalar with the model's output, and records the numerical difference. Precision and accumulation order can produce small differences. The term measures a local computation. Measuring an effect on the final prediction requires a separate intervention.

Dense support covers batch-first PyTorch `nn.Linear` and Hugging Face `Conv1D` projections. The packed Mixtral adapter records selected experts and routing weights and replays the gate, up, and down projections from the observed dispatch input. It compares the replayed mixture with the recorded mixture output. Use `--row model.layers.0.mlp.experts:2:1` for the down projection, or add `.gate` or `.up` after `experts` for either input projection. Inner operands for packed experts are identified as replayed in JSON.

Coverage reports missing layers, probes that never ran, and observations whose token axis the adapter could not map. A projection may run for every token in a sequence, while an expert may run for a subset. Adapters map the token to the runtime dispatch row before interpreting a packed tensor.

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

For a custom MoE, observe its router and map the selected position through the dispatch indices before reading expert inputs. For a fused or quantized operation, use an exposed hook point or add one to the runtime, and record whether a term was measured or recomputed. Hosted APIs that only return tokens or logits cannot supply internal traces. The built-in adapter does not run the original DeepSeek 1017 runtime or trace arbitrary fused attention, quantization kernels, or encoder–decoder token axes.

The hooks run during a single, uncompiled PyTorch forward pass with a batch size of one. Use one trace at a time per model instance. The library removes all hooks when the run finishes or raises an error.

## Tests

```bash
python -m pip install -e '.[hf]'
python -m unittest discover -s tests -v
```

The tests instantiate small GPT-2, Llama, and Mixtral models locally. They check layer and token coverage, affine reconstruction, router dispatch, expert replay, and hook cleanup.
