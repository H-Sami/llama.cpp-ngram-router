# llama.cpp n-gram router

This fork adds an adaptive router for using MTP and llama.cpp's n-gram speculators together.

MTP is always present. It provides the baseline draft and the fallback when the router does not have enough evidence to trust an n-gram result. The router never replaces target-model verification, so an n-gram guess cannot bypass the model.

## Why this exists

MTP is useful across most text, while n-gram drafting can be much faster when the model repeats or closely follows text already seen in the context. No single n-gram method works best everywhere.

The router lets several methods propose drafts, measures how they behave, and chooses the draft and length that are most likely to save time for the current context.

The n-gram methods do not merge into one larger n-gram algorithm, and they do not vote on the next token. Each method searches for reusable text in its own way and submits a candidate. The router compares those candidates and selects one producer, or one useful prefix, for target-model verification. It can also join an MTP prefix to an n-gram continuation when that combination looks stronger than either draft by itself.

This gives the system more chances to find a useful match. One method may work well for an exact repeated block, while another may handle several possible continuations better. They are competing options under one controller, and methods that are not useful for the current text are simply ignored.

## How it works

For each decoding cycle:

1. MTP creates its normal draft.
2. Enabled n-gram producers can create their own drafts.
3. An n-gram producer can also extend the MTP draft. This produces one candidate with an MTP prefix and an n-gram suffix.
4. The router compares the candidates using previous acceptance rates, proposal time, target verification time, context size, batch shape, and active slot count.
5. The target model verifies the selected candidate.
6. The router records which token positions were accepted and updates its estimates.

The router learns acceptance per token position, not just one average for an entire producer. It can choose a shorter prefix when the beginning of a draft is reliable but the tail is not. Recent results matter more than old results, which lets the router react when the workload changes.

If a long n-gram draft performs no better than the available MTP draft, that challenger enters a temporary cooldown. Useful challengers are allowed back in, so a bad match does not permanently disable a producer.

## Available producers

| Type | Role |
| --- | --- |
| `draft-mtp` | Baseline draft and safe fallback |
| `ngram-cache` | Reuses continuations stored from earlier tokens |
| `ngram-simple` | Looks for a matching token sequence and copies its continuation |
| `ngram-mod` | Uses llama.cpp's modified n-gram matching strategy |
| `ngram-map-k` | Tracks a continuation and its acceptance history |
| `ngram-map-k4v` | Tracks several possible continuations for the same key |

When the controller is enabled, it also creates a small set of fixed parameter variants for the enabled n-gram types. These variants are separate choices, so the router can learn that a strict or long configuration works in one situation while a shorter configuration works elsewhere.

## Controller modes

| Mode | Behavior |
| --- | --- |
| `off` | Uses llama.cpp's fixed speculative producer order |
| `shadow` | Collects all candidates and feedback while keeping fixed selection behavior |
| `adaptive` | Learns online and chooses the producer and prefix length |
| `replay` | Replays producer and prefix decisions from a file |

Learning can last for one request or for the lifetime of the server process. Process learning is held in memory and is not written to disk.

Requests can provide `speculative_controller_namespace` in the server API. This keeps process-level learning separate between workloads or tenants. Namespace storage is bounded, inactive entries are removed first, and a request falls back to local learning if every resident namespace is busy.

## Example

Build llama.cpp with Vulkan:

```bash
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

Run the adaptive controller with every supported n-gram producer:

```bash
./build/bin/llama-server \
  -m /path/to/model.gguf \
  --spec-type draft-mtp,ngram-cache,ngram-simple,ngram-mod,ngram-map-k,ngram-map-k4v \
  --spec-controller adaptive \
  --spec-controller-persistence process
```

For the tested two-slot setup, use a shared verification budget:

```bash
  --parallel 2 \
  --spec-controller-global-max-verify 8
```

The shared budget limits how many speculative tokens can be verified across active slots at once. Every active slot receives minimum service first, then remaining tokens go to the prefixes with the strongest confidence.

Useful controls include:

| Option | Purpose |
| --- | --- |
| `--spec-controller-margin` | Required advantage over MTP or ordinary decoding |
| `--spec-controller-decay` | How quickly old observations lose weight |
| `--spec-controller-warmup` | MTP learning cycles before full adaptation |
| `--spec-controller-max-verify` | Per-request draft verification limit |
| `--spec-controller-global-max-verify` | Shared verification limit across slots |
| `--spec-controller-max-namespaces` | Maximum process-learning namespaces |
| `--spec-controller-trace` | Write decisions and feedback as JSON Lines |

Run `llama-server --help` for producer-specific n-gram settings.

## Current scope

The implementation has been developed and tested with Qwen3.8 27B using its MTP support on Vulkan. One-slot and two-slot server use are covered. Two-slot testing uses a shared verification budget of 8 tokens.

Other models, backends, and larger slot counts have not been validated yet. This is experimental work, not an upstream llama.cpp feature.

Based on [llama.cpp](https://github.com/ggml-org/llama.cpp).
