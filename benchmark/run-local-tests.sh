#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${BUILD_DIR:-$root_dir/build-local-retrieval-tests}"
mkdir -p "$build_dir"

cxx="${CXX:-c++}"
flags=(-std=c++17 -Wall -Wextra -Wpedantic -Werror -I"$root_dir/common" -I"$root_dir/include" -I"$root_dir/ggml/include")
if [[ "${SANITIZE:-0}" == 1 ]]; then
    flags+=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined)
else
    flags+=(-O2)
fi

"$cxx" "${flags[@]}" "$root_dir/benchmark/test-ngram-retrieval.cpp" "$root_dir/common/ngram-retrieval.cpp" -o "$build_dir/test-ngram-retrieval"
"$cxx" "${flags[@]}" "$root_dir/benchmark/test-controller.cpp" "$root_dir/common/speculative-controller.cpp" -o "$build_dir/test-controller"
if [[ "${SANITIZE:-0}" == 1 ]]; then
    ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$build_dir/test-ngram-retrieval"
    ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$build_dir/test-controller"
else
    "$build_dir/test-ngram-retrieval"
    "$build_dir/test-controller"
fi
