# Replay actual DeepSeek V4.1 Flash 1017 captures

These examples read the public [Neuron Explorer](https://neuron.carrabre.com/token-path) archive and the weight values it decodes from the pinned [DeepSeek 1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170). You can run them on a laptop after installing Token Trace. They read existing records and need no GPU.

## Follow the ` THREE` token

```sh
python -m token_trace.deepseek --list-cases
python -m token_trace.deepseek --case wm-00-3d --position 171 --strongest \
  --term 24:34:1076:w2:647 \
  --term 24:34:1076:w1:32 \
  --term 24:34:1076:w3:32 \
  --output trace.json
```

`trace.json` shows the saved request, prompt, and answer, then the selected token and its surrounding text. This position is token ID **94175** with piece ` THREE` from the nested `new THREE.BoxGeometry` call. Its following recorded piece is `.B`, ID **7342**. See `tokens[0].layers` for the **20 causal encoder layers and 20 causal decoder layers**, and each layer's six routed experts and one shared expert. With `--strongest`, each expert includes the index and signed value of the largest recorded W2 input by magnitude.

The `--term` format is `LAYER:EXPERT:NEURON:PROJECTION:FEATURE`. In the shared expert, use `shared` or `-1` for the expert field. In W2, the matrix coordinate is `[output_feature, neuron]`. In W1 and W3, it is `[neuron, input_feature]`. The report includes the original weight address and checkpoint revision.

At **layer 24, expert 34, neuron 1076**, these commands reproduce:

| Projection | Coordinate | Captured input | Saved weight | Local contribution |
| --- | --- | ---: | ---: | ---: |
| W1 gate | `[1076, 32]` | 0.171875 | 0.015625 | 0.002685546875 |
| W3 up | `[1076, 32]` | 0.171875 | −0.015625 | −0.002685546875 |
| W2 down | `[647, 1076]` | 0.4375 | −0.0234375 | −0.01025390625 |

The W1 and W3 scalar products are terms in two pre-activation sums. W2 uses the captured input at the down projection, after the captured routing scale and input quantization, so the product needs no second multiplication by the router weight. These are local terms in one forward computation.

## Compare two tokens at one connection

```sh
python -m token_trace.deepseek --case wm-00-3d \
  --position 135 --position 171 --position 301 \
  --term 24:34:1076:w2:647 --output trace.json
```

| Position | Piece | W2 input | Local contribution |
| --- | --- | ---: | ---: |
| 135 | ` =` | −0.3125 | +0.00732421875 |
| 171 | ` THREE` | +0.4375 | −0.01025390625 |
| 301 | End-of-sentence marker | Expert 34 was not routed | 0 |

The same saved weight has opposite-signed contributions at positions 135 and 171. Position 301 shows the separate `expert_not_routed` state. Position 172 shows the route without detailed operands. Use `--route-only` to skip the binary files entirely.

The archive saves router choices and multipliers for each answer position. Dense operands come from selected positions in a teacher-forced replay. The report marks missing operands separately from inactive experts. Vocabulary logits and interventions require separate evidence.

## Use an offline copy

To save the featured case for repeatable offline work:

```sh
origin=https://neuron.carrabre.com/data/weights
mkdir -p capture/cases capture/runs/wm-20261115-c
curl -fsS "$origin/research.json" -o capture/research.json
curl -fsS "$origin/cases/wm-20261115-c-wm-00-3d.json.gz" \
  -o capture/cases/wm-20261115-c-wm-00-3d.json.gz
curl -fsS "$origin/runs/wm-20261115-c/wm-00-3d-171.bin.gz" \
  -o capture/runs/wm-20261115-c/wm-00-3d-171.bin.gz
curl -fsS 'https://neuron.carrabre.com/api/weights?mode=tile&tensor=layers.24.ffn.experts.34.w2.weight&row=647&col=1076&rows=1&cols=1' \
  -o capture/w2.json

python -m token_trace.deepseek --offline --data-root capture \
  --position 171 --term 24:34:1076:w2:647 --weight-file capture/w2.json \
  --output trace.json
```

For more terms, save their Explorer coefficient or tile JSON and repeat `--weight-file`. Keep the catalog's relative case and operand directory layout.
