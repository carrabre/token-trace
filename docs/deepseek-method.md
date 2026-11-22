# How we traced DeepSeek V4.1 Flash

This is the procedure behind the [Token Path explorer](https://neuron.carrabre.com/token-path). It joins three kinds of evidence from the same model revision. The checkpoint identifies tensors and saved coefficients. A recorded forward pass identifies where a token went and the operands it used. Separate interventions measure the change in an output score under a controlled edit.

## 1. Establish the model and the token

Pin the model, tokenizer and inference implementation together. The runs used [DeepSeek V4.1 Flash 1017 at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).

| Part | What to trace |
| --- | --- |
| Token and embedding | Record the exact token ID, decoded piece including leading spaces, and position in the full prompt-plus-answer sequence. The text hidden width is 5,120. |
| Causal encoder, layers 0–19 | Follow the token through 20 transformer blocks. Attention uses causal context. Engram lookups contribute at zero-based layers 1 and 14. |
| Causal decoder, layers 20–39 | Continue through 20 blocks. The decoder global KV cache is derived from the final encoder representations. Record the actual execution mode since optimized prompt processing can bypass most decoder computation for prompt positions. |
| MoE feed-forward path | At each traced backbone block, the router supplies six selected expert IDs out of 384 and their multipliers. One shared expert is also evaluated. Each expert has 2,304 SwiGLU channels. |
| Prediction | After the last block, the model combines its residual streams, normalizes, and projects to vocabulary logits. A stored continuation identifies the next token in the recorded answer. A logits probe is needed to recover the scores for that position. |

DeepSeek calls the two portions its **Causal Encoder–Decoder** architecture. The three DSpark speculative layers and the vision tower are separate from this 40-layer text backbone. The token path in this example shows the recorded backbone MoE route. Attention maps, cache reads, vision execution and speculative draft execution have their own instrumentation requirements. [DeepSeek technical report, sections 2.1–2.4](https://arxiv.org/html/2609.19969v1#S2) · [Pinned inference code and config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/inference)

## 2. Capture the forward pass

The weight study generated answers for 32 prompts, then replayed each prompt plus its generated answer using a full-sequence, teacher-forced prefill. Routing was saved across all answer positions, with detailed operands captured for 32 positions per request. The browser data retains eight detailed positions per request, or 256 positions total. Stepwise decoding can use a different kernel, batch shape or cache state. Re-run the observation in the original execution mode if exact generation-time parity is required.

1. Save the token IDs, prompt length, answer pieces, sequence positions, and checkpoint revision. In the published case data, an absolute position `p` within the answer indexes `generated_ids[p - len(prompt_ids)]` and `pieces[p - len(prompt_ids)]`.
2. Observe each backbone block's MoE gate output. Save the aligned top-k expert IDs and actual multipliers per token. The router's logical token row is separate from a dispatched expert's batch row.
3. At the expert boundary, map the model token to its local dispatch row using the actual selected IDs and dispatch order. In the reference runtime, this meant enumerating `where(selected_ids == expert_id)` for each routed expert and selecting the wanted positions among the resulting rows. Different implementations can permute, pad, group or duplicate those rows. Key the join by layer, token position and expert ID, and keep a single copy of each shared-expert observation when merging distributed results.
4. Capture the gate and up outputs at `w1` and `w3`. Before `w2`, save the operand after SwiGLU, the router multiplication for routed experts, and conversion back to the runtime dtype. Capture or reconstruct the post-quantizer operand with the same runtime quantizer and scale layout. The pinned capture reran the reference activation quantizer on selected rows, checked that the dequantized values could be encoded losslessly in BF16, and compared dot products with the observed `w2` output.
5. Restore the original runtime when tracing ends. Compare observed and unobserved forwards with identical inputs. The original study obtained bit-identical last logits for both parity requests across all four ranks. Validation additionally checked 98,301 W2 reconstructions and 196,602 W1/W3 reconstructions. Those checks sampled coordinates and operands; the saved traces cover the executed routes.

The archived browser format starts with a gzip-compressed `WMAP` payload, followed by little-endian layer count and model width and one BF16 input vector per layer. Then it stores one record per observed expert, using signed 16-bit layer and expert IDs, an FP32 routing multiplier, and two 2,304-element BF16 vectors for the weighted pre-quantizer input and quantized W2 operand. Shared expert ID is `-1` in this format. A fully measured position has 40 × 7 = 280 records. For the upstream implementation, see `tools/gpu/weight_research.py` (capture), `tools/gpu/weight_research_pack.py` (serialization), `site/lib/weights/data.js` (browser decoder), and `site/scripts/verify_weight_data.mjs` (pack integrity and alignment). See [the upstream Weight Atlas results](https://github.com/mintdotgg/intervention-lab/blob/d1e1fce730ab9e9c3359e97f3b83eee53c142b86/docs/deepseek_1017/WEIGHT_MAP_RESULTS.md) for validator details.

## 3. Follow one token down to one weight

Use the public saved case `wm-00-3d` from run `wm-20261115-c`. Absolute position **171** is token ID **94175**, decoded as ` THREE` with its leading space. The following recorded piece is `.B`, ID **7342**, in the start of `.BoxGeometry`. Position **135** in the same request is ID **438**, decoded as ` =`. Both route through expert 34 at layer 24.

| Projection | Checkpoint coordinates | Local multiply |
| --- | --- | --- |
| W1 gate | `[neuron, input_feature]` | Quantized layer input at `input_feature` × W1 coefficient. Sum over the input features to get the gate projection for that neuron. |
| W3 up | `[neuron, input_feature]` | Quantized layer input at `input_feature` × W3 coefficient. Sum over the input features to get the up projection. |
| W2 down | `[output_feature, neuron]` | Quantized W2 operand at `neuron` × W2 coefficient. Sum over the 2,304 contributing channels to get one expert output feature, subject to kernel accumulation and rounding. |

For this checkpoint, the expert clamps gate from above and up on both sides, then computes `SiLU(gate) × up`. Its routing multiplier enters before W2, so the saved post-quantizer W2 operand already incorporates routing. Read the decoded checkpoint coefficient using its original packed code and scale. Applying the router multiplier again would change the term.

At layer **24**, expert **34**, neuron **1,076**, output feature **647**, the same coefficient in `layers.24.ffn.experts.34.w2.weight` has address **441811674570** and value **−0.0234375**:

| Absolute position | Token piece | Captured W2 operand | Operand × coefficient |
| --- | --- | ---: | ---: |
| 135 | ` =` | −0.3125 | +0.00732421875 |
| 171 | ` THREE` | +0.4375 | −0.01025390625 |

The coefficient stays fixed. The operand and the signed local contribution change. The public trace offers routes for a wider set of positions than it offers operands. If a position lacks an operand file, or a requested expert was inactive at that position, report that status explicitly.

The following downloads can be replayed without running DeepSeek or downloading its checkpoint:

```bash
base=https://neuron.carrabre.com
curl -fsS "$base/data/weights/cases/wm-20261115-c-wm-00-3d.json.gz" -o case.json.gz
curl -fsS "$base/data/weights/runs/wm-20261115-c/wm-00-3d-135.bin.gz" -o token-135.bin.gz
curl -fsS "$base/data/weights/runs/wm-20261115-c/wm-00-3d-171.bin.gz" -o token-171.bin.gz
curl -fsS "$base/api/weights?mode=address&id=441811674570" -o weight-address.json
curl -fsS "$base/api/weights?mode=neuron&tensor=layers.24.ffn.experts.34.w2.weight&neuron=1076" -o weights.json
```

Check the address response for the pinned revision and coordinate `[647, 1076]`. The last response contains the decoded output weight column for neuron 1076, so check `values[647]`. The ID is specific to this checkpoint's tensor inventory and coordinate ordering.

The opening scene in the launch video uses another saved request, `saved-v1-mesh`, and the combined routing/activity from all 24 answer tokens about what a Three.js mesh combines. Its activity cannot be assigned to a single token without selecting one token from that recording. [Full-answer replay](https://neuron.carrabre.com/live?show=saved-v1-mesh) · [Featured single-token replay](https://neuron.carrabre.com/token-path)

## 4. Separate local contribution from an output change

The Weight Atlas indexed 48,496 parameter tensors. To test whether strong local contributions predicted 3D-specific effects, the study selected eight W2 coefficients on discovery prompts, paired them with same-expert, same-output-row controls, and measured edits on 16 held-out prompts. Each edit temporarily zeroed one FP4 coefficient with its scale and neighboring nibble preserved. Across 256 candidate and control edits, the measured changes in fixed continuation log probability were small and did not establish 3D-specific weights. Baseline scores were restored exactly after the edits for all 16 prompts. [Weight Atlas results](https://github.com/mintdotgg/intervention-lab/blob/d1e1fce730ab9e9c3359e97f3b83eee53c142b86/docs/deepseek_1017/WEIGHT_MAP_RESULTS.md)

A separate coordinate-space experiment tested whether activation replacement could change an interpretable binary decision:

1. Pair prompts that ask for `localToWorld` and `worldToLocal`. Present each pair in both A/B answer orders. Pin the chat wrapper and tokenization, verify the candidate letters are single-token continuations, and measure **pre-sampling logits** for each answer. Score by method to account for swapped letters.
2. Establish the unmodified baseline before looking for a target layer. The shading pilot favored `flatShading: true` for smooth and faceted prompts and failed its baseline. The early coordinate wording also failed to separate the directions. Explicit source and destination frames, plus a requirement to preserve the same physical point, gave the revised coordinate prompts a baseline of 44/44 correct comparisons across six discovery pairs and five held-out pairs.
3. Save donor block-output residuals at aligned instruction and suffix positions. For each patch, keep receiver token IDs fixed, clone its residual stream, replace the selected token's residual with the donor's, preserve the other block return values, and let the forward pass finish. DeepSeek's block output separates the residual stream from its hyper-connection pre-mix state. The residual contains four parallel streams of width 5,120 per token, so this full-vector patch replaces 20,480 scalar coordinates together. A self-patch should reproduce the baseline. In other runtimes, verify the axis, position, dtype, device and any auxiliary state before patching.
4. Choose the layer and position on discovery only and evaluate once on held-out data. Compare with same-prompt patches, color-only donor changes, and a patch at the other position. The selected patch was zero-based **layer 38** at the **last prompt token before the answer**. The full stream-vector replacement flipped **20/20 held-out choices** in the expected direction and exceeded all three control effects. Replacing only the 128 coordinates selected by discovery donor differences flipped **0/20**. The latter produced a median signed shift of **+0.088**, versus **+4.453** for the full vector and **+0.030** pooled across random coordinate groups of equal size.

These results describe the tested prompts and patch site. They do not identify a single responsible weight or establish a general 3D capability change. The coordinate experiment is distinct from the token-path demonstration, which uses saved routing and projection operands to explain computations. The [offline intervention examples](../examples/interventions/README.md) contain the result data and a checker for these comparisons.

## 5. Port the method to another runtime

- Start with one local, uncompiled forward and a known tokenization. Provide the model's actual block list in execution order. Compare a baseline forward against the traced forward and check your hooks are removed on failure.
- Use Token Trace's batch-first row mapping for ordinary dense projections. The pinned DeepSeek residual has `[batch, sequence, 4, 5120]` layout and needs its own layer reader; an expert linear call receives only the rows routed to that expert. At attention or MoE boundaries that flatten, cache, remap, or fuse tokens, use a custom probe and expose the actual position-to-row map. Document whether the observed input is a whole hidden state, a local shard, an expert dispatch row, or a replayed operand.
- For quantized or fused weights, record where the activation is quantized and where checkpoint weights are decoded. If exact internal operands are inaccessible, report a replay separately and give its measured reconstruction error. Verify matmul orientation, bias, scaling, routing and accumulation against at least one known output coordinate.
- Keep execution observations, reconstructed scalar terms and interventions distinct in the report. Report unexecuted and unmapped sites, save representative comparisons and retain the exact environment, model revision, prompt IDs and execution mode. Full token-level weight coverage requires adapters for every relevant weighted operation.
