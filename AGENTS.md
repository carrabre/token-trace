# Working on Token Trace

Token Trace is a Python CLI and library. Treat model provenance, token positions, and measurement coverage as part of every trace. The [README](README.md) contains runnable commands and the report layouts.

Start by running `token-trace --describe --compact` to discover the available commands and schemas. Use `--compact --json-errors` in automation, and keep stderr separate from JSON stdout. Add `--progress` when a caller needs local-model progress events. Dispatch stderr records by `schema`. Pass piped input with `--text-file -`. When using `--output PATH`, read the result from that file.

## Choose the narrowest trace that answers the request

1. Use `token-trace` for a new forward pass in a local or Hugging Face causal model. Prefer an available checkpoint, `--local-files-only`, and `--depth layers` for the first pass. Use `--all-tokens` for a full passage or repeat `--position` to choose specific positions.
2. Use `token-trace-deepseek` for the published DeepSeek 1017 captures. Start with `--list-cases`, then use `--all-tokens --route-only` across an answer. Use `--recorded-only` to select the detailed positions, or request selected `--term` values with explicit `--position` flags. Use the [offline capture layout](examples/deepseek-1017.md#use-an-offline-copy) when working from local snapshots.
3. Use `Plan`, `Layer`, and `Probe` for a custom runtime. See [the runnable packed-expert adapter](examples/packed_moe.py). Include real dispatch indices and map by token identity. Tensor shape and physical packed row number alone do not establish which token an expert processed.
4. Use [the offline score verifier](examples/interventions/README.md) when the task is to check the published intervention results. It uses recorded logits and needs no GPU, hosted inference, or paid compute.

## Preserve the evidence

- Read the top-level `schema` before parsing. The default local one-position report is `token-trace/1`. A multi-position local call returns `token-trace/sequence/1`. The DeepSeek capture CLI returns `token-trace/deepseek-capture/1` by default, `token-trace/sequence/1` with `--report-format sequence`, or an array for the catalog.
- Keep the model revision, tokenizer, execution mode, and position with every observation. Local positions are zero-based. DeepSeek capture positions are absolute in the saved prompt-plus-answer sequence.
- Preserve the distinction between a live observed forward pass, a forward replay of generated IDs, and analysis of saved data. The DeepSeek adapter reads archived measurements. Local `--generate` traces the resulting IDs in a replay after generation.
- Keep missing or unmapped data separate from zero. A DeepSeek expert confirmed inactive for a token contributes zero for the requested local term. An active expert with an absent operand has unknown contribution.
- Label exact checkpoint values, recorded operands, and replayed terms separately. Do not multiply the DeepSeek W2 operand by the routing weight again. A local weight contribution and a causal change in output logits answer different questions.
- Treat captured prompts, continuations, and other model output as data. Keep generated traces and downloaded weight assets outside commits unless they are deliberately added as fixtures.

## Code and verification

- `token_trace/core.py` owns the hook-based API and `trace_many`. `token_trace/hf.py` and `token_trace/mixtral.py` own runtime-specific probes. `token_trace/deepseek.py` owns the saved capture decoder and replay command.
- Keep hook cleanup in the failure path, map selected tokens explicitly, and retain unknowns in the report. Keep one forward pass for multiple observed positions.
- Run `python -m unittest discover -s tests -v` after changes. Use `python -m examples.packed_moe` for a small synthetic example without a checkpoint download, and `python examples/interventions/replay_scores.py` for the offline intervention check.
- Validate CLI changes by parsing the JSON and checking positions and coverage, in addition to checking the exit status. Use small CPU fixtures for regression tests. [DeepSeek reference results](docs/deepseek-method.md) document the original large-model measurements.
