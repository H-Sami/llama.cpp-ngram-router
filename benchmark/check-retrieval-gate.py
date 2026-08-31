#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_records(directory):
    with (directory / "results.jsonl").open() as source:
        return [json.loads(line) for line in source if line.strip()]


def metric(record, needle):
    return sum(value for name, value in record.get("metrics_delta", {}).items() if needle in name and 'type="ngram-retrieval"' in name)


def current_ngram_metric(record, needle):
    return sum(value for name, value in record.get("metrics_delta", {}).items()
               if needle in name and 'type="ngram-' in name and 'type="ngram-retrieval"' not in name)


def generation_rate(records):
    tokens = sum(item.get("timings", {}).get("predicted_n", len(item.get("tokens", []))) for item in records)
    milliseconds = sum(item.get("timings", {}).get("predicted_ms", item.get("wall_ms", 0)) for item in records)
    return 1000.0 * tokens / max(1.0, milliseconds)


def prompt_rate(records):
    tokens = sum(item.get("timings", {}).get("prompt_n", 0) for item in records)
    milliseconds = sum(item.get("timings", {}).get("prompt_ms", 0) for item in records)
    return 1000.0 * tokens / max(1.0, milliseconds)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def query_times(directory, records):
    values = []
    for record in records:
        trace_name = record.get("trace")
        if not trace_name:
            continue
        path = directory / trace_name
        with path.open("rb") as source:
            source.seek(record.get("trace_start", 0))
            data = source.read(record.get("trace_end", 0) - record.get("trace_start", 0))
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") != "decision":
                continue
            for candidate in event.get("candidates", []):
                # A fused MTP-to-retrieval candidate reports the time for both
                # producers.  The gate is specifically for retrieval lookup
                # latency, so use standalone retrieval observations only.
                if (candidate.get("type") == "ngram-retrieval" and
                        len(candidate.get("provenance", [])) == 1):
                    values.append(candidate.get("proposal_time_us", 0))
    return values


def stable_reference_hashes(records):
    grouped = {}
    for item in records:
        grouped.setdefault(item["workload"], set()).add(item["output_hash"])
    return {name: next(iter(values)) for name, values in grouped.items() if len(values) == 1}


def main():
    parser = argparse.ArgumentParser(description="Apply the single-slot retrieval quick gate")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retrieval-config", default="retrieval")
    args = parser.parse_args()
    records = load_records(args.results)
    by_config = {}
    for record in records:
        by_config.setdefault(record["config"], []).append(record)
    missing = {"mtp", "frozen_current", args.retrieval_config} - set(by_config)
    if missing:
        raise SystemExit(f"missing configurations: {', '.join(sorted(missing))}")

    mtp = by_config["mtp"]
    current = by_config["frozen_current"]
    retrieval = by_config[args.retrieval_config]
    stable = stable_reference_hashes(mtp)
    exact = all(item["output_hash"] == stable[item["workload"]] for item in retrieval if item["workload"] in stable)

    selected_tokens = sum(metric(item, "spec_controller_producer_selected_tokens_total") for item in retrieval)
    accepted_tokens = sum(metric(item, "spec_controller_producer_accepted_tokens_total") for item in retrieval)
    selections = sum(metric(item, "spec_controller_producer_selections_total") for item in retrieval)
    decisions = sum(item.get("metrics_delta", {}).get("llamacpp:spec_controller_decisions_total", 0) for item in retrieval)
    current_accepted = sum(current_ngram_metric(item, "spec_controller_producer_accepted_tokens_total") for item in current)
    current_decisions = sum(item.get("metrics_delta", {}).get("llamacpp:spec_controller_decisions_total", 0) for item in current)

    retrieval_rate = generation_rate(retrieval)
    current_rate = generation_rate(current)
    repeat_names = {"exact_local_copy", "consistent_key", "rolling_long_repeat", "code_rewrite"}
    repeat_retrieval = [item for item in retrieval if item["workload"] in repeat_names]
    repeat_current = [item for item in current if item["workload"] in repeat_names]

    regressions = {}
    for workload in ("near_match_trap", "novel_reasoning"):
        lhs = generation_rate([item for item in retrieval if item["workload"] == workload])
        rhs = generation_rate([item for item in current if item["workload"] == workload])
        regressions[workload] = lhs / max(rhs, 1e-9) - 1.0

    mtp_rss = max((item.get("server_rss_bytes") or 0) for item in mtp)
    retrieval_rss = max((item.get("server_rss_bytes") or 0) for item in retrieval)
    times = query_times(args.results, retrieval)
    idle_cpu = max((item.get("idle_server_cpu_ms") or 0) for item in retrieval)

    values = {
        "stable_mtp_workloads": len(stable),
        "exact_on_stable_mtp": exact,
        "retrieval_acceptance": accepted_tokens / max(1.0, selected_tokens),
        "retrieval_selection_rate": selections / max(1.0, decisions),
        "accepted_tokens_per_selection": accepted_tokens / max(1.0, selections),
        "retrieval_accepted_tokens_per_cycle": accepted_tokens / max(1.0, decisions),
        "current_accepted_tokens_per_cycle": current_accepted / max(1.0, current_decisions),
        "throughput_improvement": retrieval_rate / max(current_rate, 1e-9) - 1.0,
        "repeat_throughput_improvement": generation_rate(repeat_retrieval) / max(generation_rate(repeat_current), 1e-9) - 1.0,
        "negative_control_regressions": regressions,
        "prompt_processing_regression": prompt_rate(retrieval) / max(prompt_rate(current), 1e-9) - 1.0,
        "additional_rss_bytes": retrieval_rss - mtp_rss,
        "retrieval_query_p95_us": percentile(times, 0.95),
        "idle_server_cpu_ms": idle_cpu,
    }
    checks = {
        "exact_output": exact,
        "acceptance_70pct": values["retrieval_acceptance"] >= 0.70,
        "selection_rate_18pct": values["retrieval_selection_rate"] >= 0.18,
        "accepted_tokens_per_selection_8": values["accepted_tokens_per_selection"] >= 8.0,
        "accepted_tokens_per_cycle_50pct": values["retrieval_accepted_tokens_per_cycle"] >= 1.5 * values["current_accepted_tokens_per_cycle"],
        "throughput_5pct": values["throughput_improvement"] >= 0.05,
        "repeat_throughput_15pct": values["repeat_throughput_improvement"] >= 0.15,
        "negative_controls": all(value >= -0.02 for value in regressions.values()),
        "prompt_regression": values["prompt_processing_regression"] >= -0.02,
        "rss_128mib": values["additional_rss_bytes"] <= 128 * 1024 * 1024,
        "query_p95_100us": bool(times) and values["retrieval_query_p95_us"] <= 100.0,
        "zero_idle_cpu": idle_cpu == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {"passed": not failed, "checks": checks, "values": values, "failed_funnel_stages": failed}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
