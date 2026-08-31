#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


POLICY_SYMBOLS = {
    "longest-recency": "COMMON_NGRAM_RETRIEVAL_LONGEST_RECENT",
    "majority-prefix": "COMMON_NGRAM_RETRIEVAL_MAJORITY_PREFIX",
    "weighted-prefix": "COMMON_NGRAM_RETRIEVAL_WEIGHTED_PREFIX",
}


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generation_rate(records):
    tokens = sum(item["timings"]["predicted_n"] for item in records)
    milliseconds = sum(item["timings"]["predicted_ms"] for item in records)
    return 1000.0 * tokens / milliseconds


def retrieval_metric(records, needle):
    return sum(
        value
        for item in records
        for name, value in item.get("metrics_delta", {}).items()
        if needle in name and 'type="ngram-retrieval"' in name
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select the fastest development policy subject to retrieval accuracy")
    parser.add_argument("--result", action="append", required=True, metavar="POLICY=RESULTS_JSONL")
    parser.add_argument("--minimum-acceptance", type=float, default=0.70)
    parser.add_argument("--policy-header", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    policies = {}
    for item in args.result:
        policy, separator, raw_path = item.partition("=")
        if not separator or policy not in POLICY_SYMBOLS:
            parser.error(f"invalid --result {item!r}")
        records = load_records(Path(raw_path))
        selected = retrieval_metric(records, "producer_selected_tokens_total")
        accepted = retrieval_metric(records, "producer_accepted_tokens_total")
        policies[policy] = {
            "generation_tokens_per_second": generation_rate(records),
            "retrieval_acceptance": accepted / max(1.0, selected),
            "selected_tokens": selected,
            "accepted_tokens": accepted,
        }

    eligible = [
        policy for policy, values in policies.items()
        if values["retrieval_acceptance"] >= args.minimum_acceptance
    ]
    if not eligible:
        raise SystemExit("no policy met the minimum retrieval acceptance")
    selected_policy = max(eligible, key=lambda policy: policies[policy]["generation_tokens_per_second"])
    report = {
        "selected_policy": selected_policy,
        "minimum_acceptance": args.minimum_acceptance,
        "policies": policies,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    args.policy_header.write_text(
        "#pragma once\n\n"
        "// Generated from measured development throughput by benchmark/select-retrieval-policy.py.\n"
        "constexpr common_ngram_retrieval_policy COMMON_NGRAM_RETRIEVAL_DEVELOPMENT_POLICY =\n"
        f"    {POLICY_SYMBOLS[selected_policy]};\n"
    )


if __name__ == "__main__":
    main()
