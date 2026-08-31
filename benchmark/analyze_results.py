#!/usr/bin/env python3

import argparse
import collections
import json
from pathlib import Path
import statistics


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def read_trace_segment(path, start, end):
    if not path.exists() or end <= start:
        return []
    with path.open("rb") as source:
        source.seek(start)
        raw = source.read(end - start)
    events = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def producer_definitions(path):
    definitions = {}
    if not path.exists():
        return definitions
    for event in load_jsonl(path):
        if event.get("event") != "producers":
            continue
        for producer in event["producers"]:
            definitions[producer["producer_id"]] = producer
    return definitions


def candidate_label(candidate, definitions):
    provenance = candidate.get("provenance", [])
    if len(provenance) > 1:
        names = []
        for span in provenance:
            definition = definitions.get(span["producer_id"], {})
            names.append(definition.get("configuration_name") or definition.get("type") or str(span["producer_id"]))
        return "fused:" + "+".join(names)
    definition = definitions.get(candidate["producer_id"], {})
    return definition.get("configuration_name") or definition.get("type") or candidate["type"]


def trace_summary(events, definitions):
    pending = collections.defaultdict(collections.deque)
    coverage = collections.Counter()
    candidate_calls = collections.Counter()
    candidate_tokens = collections.Counter()
    proposal_us = collections.Counter()
    selected = collections.Counter()
    selected_proposed = collections.Counter()
    selected_accepted = collections.Counter()
    cycles = 0

    for event in events:
        event_type = event.get("event")
        if event_type == "decision":
            cycles += 1
            pending[event["seq_id"]].append(event)
            seen = set()
            for candidate in event["candidates"]:
                label = candidate_label(candidate, definitions)
                seen.add(label)
                candidate_calls[label] += 1
                candidate_tokens[label] += len(candidate["tokens"])
                proposal_us[label] += candidate.get("proposal_time_us", 0)
            coverage.update(seen)
        elif event_type == "feedback":
            queue = pending[event["seq_id"]]
            if not queue:
                continue
            decision = queue.popleft()
            selected_index = decision["selected_index"]
            if selected_index < 0:
                label = "ordinary"
            else:
                label = candidate_label(decision["candidates"][selected_index], definitions)
            selected[label] += 1
            selected_proposed[label] += decision["prefix_length"]
            selected_accepted[label] += event["accepted_length"]

    return {
        "cycles": cycles,
        "coverage_cycles": dict(coverage),
        "candidate_calls": dict(candidate_calls),
        "candidate_tokens": dict(candidate_tokens),
        "proposal_time_us": dict(proposal_us),
        "selected_cycles": dict(selected),
        "selected_proposed_tokens": dict(selected_proposed),
        "selected_accepted_tokens": dict(selected_accepted),
    }


def metric(record, name, default=0.0):
    return float(record.get("metrics_delta", {}).get(name, default))


def median(values):
    return statistics.median(values) if values else 0.0


def aggregate(records):
    groups = collections.defaultdict(list)
    for record in records:
        groups[(record["config"], record["workload"])].append(record)

    rows = []
    for (config, workload), group in sorted(groups.items()):
        rates = [float(item.get("timings", {}).get("predicted_per_second", 0.0)) for item in group]
        wall = [float(item["wall_ms"]) for item in group]
        draft = sum(metric(item, "llamacpp:spec_decode_num_draft_tokens_total") for item in group)
        accepted = sum(metric(item, "llamacpp:spec_decode_num_accepted_tokens_total") for item in group)
        rows.append({
            "config": config,
            "workload": workload,
            "runs": len(group),
            "median_tokens_per_second": median(rates),
            "min_tokens_per_second": min(rates) if rates else 0.0,
            "max_tokens_per_second": max(rates) if rates else 0.0,
            "median_wall_ms": median(wall),
            "draft_tokens": draft,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / draft if draft else 0.0,
        })
    return rows


def weighted_rate(records):
    tokens = sum(float(item.get("timings", {}).get("predicted_n", 0.0)) for item in records)
    milliseconds = sum(float(item.get("timings", {}).get("predicted_ms", 0.0)) for item in records)
    return 1000.0 * tokens / milliseconds if milliseconds else 0.0


def aggregate_configs(records, server_scope):
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["config"]].append(record)
    result = []
    for config, group in sorted(groups.items()):
        cold = group if server_scope == "request" else [item for item in group if item["repetition"] == 0]
        warm = [] if server_scope == "request" else [item for item in group if item["repetition"] > 0]
        result.append({
            "config": config,
            "runs": len(group),
            "weighted_tokens_per_second": weighted_rate(group),
            "cold_tokens_per_second": weighted_rate(cold),
            "warm_tokens_per_second": weighted_rate(warm),
        })
    return result


def output_stability(records):
    groups = collections.defaultdict(set)
    for record in records:
        groups[(record["config"], record["workload"], record.get("request_seed"))].add(record["output_hash"])
    return [
        {
            "config": config,
            "workload": workload,
            "request_seed": seed,
            "distinct_output_hashes": len(hashes),
        }
        for (config, workload, seed), hashes in sorted(groups.items())
        if len(hashes) > 1
    ]


def render_markdown(rows, config_rows, correctness, instability):
    lines = [
        "# Benchmark summary",
        "",
        f"Correctness: **{'PASS' if correctness.get('passed') else 'FAIL'}** "
        f"({len(correctness.get('mismatches', []))} mismatches)",
        "",
        "## Overall throughput",
        "",
        "| Configuration | Runs | Weighted tok/s | Cold tok/s | Warm tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in config_rows:
        lines.append(
            f"| {row['config']} | {row['runs']} | {row['weighted_tokens_per_second']:.2f} | "
            f"{row['cold_tokens_per_second']:.2f} | {row['warm_tokens_per_second']:.2f} |"
        )
    lines.extend([
        "",
        f"Within-configuration unstable workload groups: **{len(instability)}**",
        "",
        "## Per-workload throughput",
        "",
        "| Configuration | Workload | Runs | Median tok/s | Range tok/s | Acceptance |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['workload']} | {row['runs']} | "
            f"{row['median_tokens_per_second']:.2f} | "
            f"{row['min_tokens_per_second']:.2f}-{row['max_tokens_per_second']:.2f} | "
            f"{row['acceptance_rate']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark result and trace artifacts")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    records = load_jsonl(args.result_dir / "results.jsonl")
    manifest = json.loads((args.result_dir / "manifest.json").read_text(encoding="utf-8"))
    correctness = json.loads((args.result_dir / "correctness.json").read_text(encoding="utf-8"))
    rows = aggregate(records)
    server_scope = manifest.get("server_scope", "config")
    config_rows = aggregate_configs(records, server_scope)
    instability = output_stability(records)

    trace_runs = []
    definition_cache = {}
    for record in records:
        if not record.get("trace"):
            continue
        trace_path = args.result_dir / record["trace"]
        definitions = definition_cache.setdefault(trace_path, producer_definitions(trace_path))
        events = read_trace_segment(trace_path, record["trace_start"], record["trace_end"])
        trace_runs.append({
            "config": record["config"],
            "workload": record["workload"],
            "repetition": record["repetition"],
            "summary": trace_summary(events, definitions),
        })

    report = {
        "correctness": correctness,
        "server_scope": server_scope,
        "within_config_instability": instability,
        "config_aggregates": config_rows,
        "groups": rows,
        "trace_runs": trace_runs,
    }
    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_markdown(rows, config_rows, correctness, instability)
    print(rendered_markdown, end="")

    json_out = args.json_out or args.result_dir / "summary.json"
    markdown_out = args.markdown_out or args.result_dir / "summary.md"
    json_out.write_text(rendered_json, encoding="utf-8")
    markdown_out.write_text(rendered_markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
