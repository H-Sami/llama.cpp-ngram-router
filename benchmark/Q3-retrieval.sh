#!/usr/bin/env bash

set -Eeuo pipefail

# Adaptive MTP and n-gram version of Q3.sh.
# The original /home/pc/launchers/Q3.sh is not modified.

MODEL="/home/pc/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
MMPROJ="/home/pc/models/mmproj-F16.gguf"
CHAT_TEMPLATE="/home/pc/models/chat_template.jinja"
LLAMA_SERVER="${LLAMA_SERVER:-/home/pc/Work/llama.cpp-retrieval/build-vulkan-retrieval-v2/bin/llama-server}"

CTX_SIZE="${CTX_SIZE:-81920}"
N_PARALLEL="${N_PARALLEL:-1}"
SERVER_PORT="${SERVER_PORT:-8080}"
POLL="${POLL:-0}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-8}"
CHECKPOINT_MIN_STEP="${CHECKPOINT_MIN_STEP:-8192}"

SPEC_TYPES="${SPEC_TYPES:-draft-mtp,ngram-retrieval}"
MTP_N_MAX="${MTP_N_MAX:-3}"
NGRAM_RETRIEVAL_MAX="${NGRAM_RETRIEVAL_MAX:-48}"

NGRAM_CACHE_MAX="${NGRAM_CACHE_MAX:-8}"
NGRAM_MATCH="${NGRAM_MATCH:-32}"
NGRAM_MIN="${NGRAM_MIN:-16}"
NGRAM_MAX="${NGRAM_MAX:-32}"
SIMPLE_N="${SIMPLE_N:-12}"
SIMPLE_M="${SIMPLE_M:-48}"
SIMPLE_MIN_HITS="${SIMPLE_MIN_HITS:-1}"
MAP_K_N="${MAP_K_N:-12}"
MAP_K_M="${MAP_K_M:-48}"
MAP_K_MIN_HITS="${MAP_K_MIN_HITS:-1}"
MAP_K4V_N="${MAP_K4V_N:-12}"
MAP_K4V_M="${MAP_K4V_M:-48}"
MAP_K4V_MIN_HITS="${MAP_K4V_MIN_HITS:-1}"

SPEC_CONTROLLER="${SPEC_CONTROLLER:-adaptive}"
CONTROLLER_PERSISTENCE="${CONTROLLER_PERSISTENCE:-process}"
CONTROLLER_WARMUP="${CONTROLLER_WARMUP:-4}"
CONTROLLER_MAX_VERIFY="${CONTROLLER_MAX_VERIFY:-0}"
CONTROLLER_GLOBAL_MAX_VERIFY="${CONTROLLER_GLOBAL_MAX_VERIFY-}"
CONTROLLER_MAX_NAMESPACES="${CONTROLLER_MAX_NAMESPACES:-64}"
CONTROLLER_MARGIN="${CONTROLLER_MARGIN:-0.05}"
CONTROLLER_DECAY="${CONTROLLER_DECAY:-0.98}"
CONTROLLER_TRACE="${CONTROLLER_TRACE:-}"
BENCH_TRACE="${BENCH_TRACE:-0}"

die() {
    printf 'Q3ngram.sh: %s\n' "$*" >&2
    exit 2
}

for integer_var in MTP_N_MAX NGRAM_RETRIEVAL_MAX NGRAM_CACHE_MAX NGRAM_MATCH NGRAM_MIN NGRAM_MAX \
        SIMPLE_N SIMPLE_M SIMPLE_MIN_HITS \
        MAP_K_N MAP_K_M MAP_K_MIN_HITS \
        MAP_K4V_N MAP_K4V_M MAP_K4V_MIN_HITS \
        CONTROLLER_WARMUP CONTROLLER_MAX_VERIFY CONTROLLER_MAX_NAMESPACES \
        N_PARALLEL SERVER_PORT POLL CTX_CHECKPOINTS CHECKPOINT_MIN_STEP; do
    value="${!integer_var}"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$integer_var must be a non-negative integer (got '$value')"
done

for positive_var in MTP_N_MAX NGRAM_CACHE_MAX NGRAM_MATCH NGRAM_MIN NGRAM_MAX \
        SIMPLE_N SIMPLE_M SIMPLE_MIN_HITS \
        MAP_K_N MAP_K_M MAP_K_MIN_HITS \
        MAP_K4V_N MAP_K4V_M MAP_K4V_MIN_HITS N_PARALLEL; do
    (( ${!positive_var} > 0 )) || die "$positive_var must be positive"
done

(( NGRAM_CACHE_MAX <= 1024 )) || die "NGRAM_CACHE_MAX must not exceed 1024"
(( NGRAM_RETRIEVAL_MAX >= 8 && NGRAM_RETRIEVAL_MAX <= 48 )) || \
    die "NGRAM_RETRIEVAL_MAX must be between 8 and 48"
(( CONTROLLER_MAX_NAMESPACES > 0 && CONTROLLER_MAX_NAMESPACES <= 4096 )) || \
    die "CONTROLLER_MAX_NAMESPACES must be between 1 and 4096"
(( NGRAM_MIN <= NGRAM_MAX )) || die "NGRAM_MIN must be less than or equal to NGRAM_MAX"
(( SERVER_PORT > 0 && SERVER_PORT <= 65535 )) || die "SERVER_PORT must be between 1 and 65535"
(( CTX_CHECKPOINTS > 0 && CTX_CHECKPOINTS <= 64 )) || \
    die "CTX_CHECKPOINTS must be between 1 and 64"
[[ "$BENCH_TRACE" == 0 || "$BENCH_TRACE" == 1 ]] || die "BENCH_TRACE must be 0 or 1"

case "$SPEC_CONTROLLER" in
    off|shadow|adaptive|replay) ;;
    *) die "SPEC_CONTROLLER must be off, shadow, adaptive, or replay" ;;
esac

case "$CONTROLLER_PERSISTENCE" in
    request|process) ;;
    *) die "CONTROLLER_PERSISTENCE must be request or process" ;;
esac

if [[ -z "$CONTROLLER_GLOBAL_MAX_VERIFY" ]]; then
    if [[ "$SPEC_CONTROLLER" == adaptive ]] && (( N_PARALLEL == 2 )); then
        CONTROLLER_GLOBAL_MAX_VERIFY=8
    elif [[ "$SPEC_CONTROLLER" == adaptive ]] && (( N_PARALLEL > 2 )); then
        die "adaptive mode above two slots has not been validated"
    else
        CONTROLLER_GLOBAL_MAX_VERIFY=0
    fi
elif [[ ! "$CONTROLLER_GLOBAL_MAX_VERIFY" =~ ^[0-9]+$ ]]; then
    die "CONTROLLER_GLOBAL_MAX_VERIFY must be a non-negative integer"
fi

IFS=',' read -r -a spec_types <<< "$SPEC_TYPES"
(( ${#spec_types[@]} > 0 )) || die "SPEC_TYPES must not be empty"
declare -A seen_spec_types=()
for spec_type in "${spec_types[@]}"; do
    case "$spec_type" in
        none|draft-mtp|ngram-cache|ngram-simple|ngram-mod|ngram-map-k|ngram-map-k4v|ngram-retrieval) ;;
        *) die "unsupported SPEC_TYPES entry '$spec_type'" ;;
    esac
    [[ -z "${seen_spec_types[$spec_type]:-}" ]] || die "duplicate SPEC_TYPES entry '$spec_type'"
    seen_spec_types[$spec_type]=1
done

if (( ${#spec_types[@]} > 1 )) && [[ -n "${seen_spec_types[none]:-}" ]]; then
    die "SPEC_TYPES=none cannot be combined with another implementation"
fi
if [[ "$SPEC_CONTROLLER" != off && -z "${seen_spec_types[draft-mtp]:-}" ]]; then
    die "controller modes require draft-mtp in SPEC_TYPES"
fi

for required in "$MODEL" "$MMPROJ" "$CHAT_TEMPLATE" "$LLAMA_SERVER"; do
    [[ -e "$required" ]] || die "missing required file: $required"
done

ulimit -l unlimited 2>/dev/null || true

# Hide the RTX 2060 and expose only the RX 9060 XT. After filtering, the RX is
# Vulkan0 inside llama.cpp.
export CUDA_VISIBLE_DEVICES=-1
export GGML_VK_VISIBLE_DEVICES=1
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

args=(
    -m "$MODEL"
    --host 127.0.0.1
    --port "$SERVER_PORT"
    --webui
    --device Vulkan0
    -sm none
    -ngl all
    --fit off
    --no-host
    --mmproj "$MMPROJ"
    --mmproj-device none
    --no-mmproj-offload
    -fa on
    -c "$CTX_SIZE"
    -np "$N_PARALLEL"
    --kv-unified
    -ctk q8_0
    -ctv q8_0
    -b 1024
    -ub 512
    -t 2
    -tb 8
    --poll "$POLL"
    --poll-batch 1
    --ctx-checkpoints "$CTX_CHECKPOINTS"
    --checkpoint-min-step "$CHECKPOINT_MIN_STEP"
    --cache-ram 8192
    --cache-idle-slots
    --jinja
    --chat-template-file "$CHAT_TEMPLATE"
    --reasoning on
    --temp 1.0
    --top-p 0.95
    --top-k 20
    --min-p 0.0
    --presence-penalty 0.0
    --repeat-penalty 1.0
    --spec-type "$SPEC_TYPES"
    --spec-draft-device Vulkan0
    --spec-draft-ngl all
    -ctkd f16
    -ctvd f16
    --spec-draft-n-max "$MTP_N_MAX"
    --spec-draft-p-min 0
    --spec-draft-backend-sampling
    --spec-draft-threads 1
    --spec-draft-threads-batch 1
    --spec-draft-poll 1
    --spec-draft-poll-batch 0
    --spec-ngram-cache-n-max "$NGRAM_CACHE_MAX"
    --spec-ngram-mod-n-match "$NGRAM_MATCH"
    --spec-ngram-mod-n-min "$NGRAM_MIN"
    --spec-ngram-mod-n-max "$NGRAM_MAX"
    --spec-ngram-simple-size-n "$SIMPLE_N"
    --spec-ngram-simple-size-m "$SIMPLE_M"
    --spec-ngram-simple-min-hits "$SIMPLE_MIN_HITS"
    --spec-ngram-map-k-size-n "$MAP_K_N"
    --spec-ngram-map-k-size-m "$MAP_K_M"
    --spec-ngram-map-k-min-hits "$MAP_K_MIN_HITS"
    --spec-ngram-map-k4v-size-n "$MAP_K4V_N"
    --spec-ngram-map-k4v-size-m "$MAP_K4V_M"
    --spec-ngram-map-k4v-min-hits "$MAP_K4V_MIN_HITS"
    --metrics
)

if [[ -n "${seen_spec_types[ngram-retrieval]:-}" ]]; then
    args+=(--spec-ngram-retrieval-n-max "$NGRAM_RETRIEVAL_MAX")
fi

if [[ "$SPEC_CONTROLLER" != off ]]; then
    args+=(
        --spec-controller "$SPEC_CONTROLLER"
        --spec-controller-persistence "$CONTROLLER_PERSISTENCE"
        --spec-controller-warmup "$CONTROLLER_WARMUP"
        --spec-controller-max-verify "$CONTROLLER_MAX_VERIFY"
        --spec-controller-global-max-verify "$CONTROLLER_GLOBAL_MAX_VERIFY"
        --spec-controller-max-namespaces "$CONTROLLER_MAX_NAMESPACES"
        --spec-controller-margin "$CONTROLLER_MARGIN"
        --spec-controller-decay "$CONTROLLER_DECAY"
    )
    if [[ -n "$CONTROLLER_TRACE" ]]; then
        mkdir -p "$(dirname "$CONTROLLER_TRACE")"
        args+=(--spec-controller-trace "$CONTROLLER_TRACE")
    fi
fi

if [[ "$BENCH_TRACE" == 1 ]]; then
    args+=(--verbose)
fi

exec "$LLAMA_SERVER" "${args[@]}"
