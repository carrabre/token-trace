# Replay the DeepSeek intervention results

Run from the repository root:

```bash
python examples/interventions/replay_scores.py
```

The checked-in fixture contains the authored prompt cases, baseline candidate logits, held-out control comparisons, selected-coordinate and random-subset comparisons, and each source result's SHA-256. The verifier recomputes decisions from logits. This example reanalyzes captured results on a CPU. A fresh activation swap requires the pinned checkpoint and its instrumented runtime.

| Study | Observed result |
| --- | --- |
| Three.js `flatShading` | The model preferred `true` even for smooth-surface prompts. The baseline gate stopped the swap experiment. |
| Initial coordinate-space prompts | `localToWorld` was preferred on 4 of 22 local-to-world comparisons. The baseline gate stopped the swap experiment. |
| Explicit source and destination frames | The expected comparison won 44/44. Replacing the full layer-38 suffix activation with one from the opposite task flipped 20/20 held-out comparisons. Replacing 128 selected coordinates flipped 0/20. |

`outward` means `localToWorld`, and `inward` means `worldToLocal`. Both answer orders are included. Continue only after the model distinguishes the baseline cases. Measure the candidate against the self-patch, color-change, and other-position controls before interpreting the effect. The unit of comparison is a fixed pair of candidate answers for each prompt.
