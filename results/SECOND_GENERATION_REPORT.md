# Second-generation retrieval result

The second-generation single-slot retrieval engine is implemented in `build-vulkan-retrieval-v2`. It was initially withheld because the quick gate did not pass in full. On 2026-08-31, the user explicitly authorized deployment with those known failures; `Q3ngram.sh` now uses the v2 binary while `Q3.sh` and the frozen legacy binary remain unchanged. A live long-context test then exposed a controller path that could suppress MTP before retrieval selection and fall through to ordinary decoding. That path was removed on 2026-09-01, the local strict and sanitizer tests passed, and the Vulkan binary was rebuilt. The full performance gate has not yet been rerun after this correction.

## Architecture delivered

- Independent bounded 4-, 8-, and 16-token rolling-hash indexes, with exact-token collision verification and request isolation.
- Deduplicated multi-source prefix trie with weighted-prefix consensus and per-position match, support, dominance, distance, and MTP evidence.
- Verification boundaries at ambiguous branches and support/source discontinuities.
- Request-local trust carried safely across recency/support tier migration; bounded process-level tier calibration remains isolated by namespace.
- MTP remains available on every cycle so a rejected retrieval route cannot fall through to ordinary decoding without a speculative fallback.
- Transactional MTP-to-retrieval extensions, canonical feedback attribution, fixed verification rungs, cooldown recovery, telemetry, and a multi-anchor offline evaluator.

## Pre-correction deterministic holdout

Source: `results/retrieval-v2-final-holdout-t0`.

- Exact output against stable MTP: pass on all six workloads.
- Weighted generation throughput: 49.26 tok/s versus 45.98 tok/s frozen legacy, +7.13%.
- Retrieval-token acceptance: 74.74%.
- Accepted tokens per retrieval selection: 13.37.
- Near-match trap throughput: +30.20%; novel reasoning: -0.14%.
- Prompt processing: +1.36%; query p95: 22 microseconds; idle CPU: zero.
- Failed gates: 9.82% selection rate, 1.31 accepted retrieval tokens/cycle versus the required 1.5x legacy rate, -8.96% repeat-subset throughput, and +149,430,272 bytes RSS versus MTP.

## Pre-correction temperature-1 holdout

Source: `results/retrieval-v2-final-holdout-t1`.

- Weighted throughput: +3.18% versus frozen legacy.
- Retrieval-token acceptance: 69.12%.
- Same-seed output diverged from stable MTP on `agent_tool_loop`; exact-output gate failed.
- Negative controls, prompt processing, query latency, and idle CPU checks passed.

## Verification

- Strict `-Wall -Wextra -Wpedantic -Werror` controller and retrieval tests pass.
- ASan/UBSan controller and retrieval tests pass.
- Clean Vulkan release build passes.
- Offline development evaluation independently selects weighted-prefix consensus.
- Frozen launcher hashes remain `32dfaa7f0dd0fee248c49d3e6823f01575be7a2c0d390726bd9c11daec8b9d76` (`Q3ngram.sh`) and `b109bd5d73bfcb96407dc60a942aaed34bf00986f9ba914bc1d61d6048601ecb` (`Q3.sh`).

## Deployment history

- Active launcher: `/home/pc/launchers/Q3ngram.sh`.
- Rollback launcher: `/home/pc/launchers/Q3ngram.sh.backup-20260831-231755`.
- Active defaults: `SPEC_TYPES=draft-mtp,ngram-retrieval`, MTP depth 3, retrieval depth 48, adaptive controller, process persistence, and four-cycle warmup.
- Startup, novel-reasoning, exact-copy, clean shutdown, restart, health, and post-restart completion checks passed.
- The exact-copy smoke selected retrieval once and target-accepted 42 retrieval tokens.

The archived results describe the pre-correction binary and must not be treated as performance evidence for the corrected build. Stochastic invariance, repeat-heavy throughput, retrieval activity, verification-memory overhead, and the corrected long-context behavior remain open validation work.
