#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


DEVELOPMENT = {
    "exact_local_copy", "consistent_key", "ambiguous_key", "fusion_bridge",
    "regime_change", "circuit_recovery", "cooldown_recovery_long",
}
HOLDOUT = {
    "rolling_long_repeat", "agent_tool_loop", "code_rewrite", "near_match_trap",
    "novel_reasoning", "json_schema",
}
POLICIES = ("longest-recency", "majority-prefix", "weighted-prefix")
ANCHOR_SIZES = (4, 8, 16)


def anchor_hash(tokens):
    value = 1469598103934665603
    for token in tokens:
        for byte in int(token & 0xFFFFFFFF).to_bytes(4, "little"):
            value ^= byte
            value = value * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return value


def build_indexes(tokens):
    indexes = []
    for length in ANCHOR_SIZES:
        postings = defaultdict(lambda: deque(maxlen=32))
        for position in range(max(0, len(tokens) - length + 1)):
            postings[anchor_hash(tokens[position:position + length])].append(position)
        indexes.append(postings)
    return indexes


def retrieve(tokens, policy, n_max):
    if len(tokens) < 5:
        return [], []
    indexes = build_indexes(tokens)
    unique = {}
    for length, postings in reversed(list(zip(ANCHOR_SIZES, indexes))):
        if len(tokens) < length + 1:
            continue
        current = len(tokens) - length
        anchor = tokens[current:current + length]
        for source in postings.get(anchor_hash(anchor), ()):
            if source >= current or tokens[source:source + length] != anchor:
                continue
            backward = 0
            while (backward + length < 48 and backward < source and backward < current and
                   tokens[source - backward - 1] == tokens[current - backward - 1]):
                backward += 1
            continuation_begin = source + length
            continuation = tokens[continuation_begin:min(current, continuation_begin + n_max)]
            match = (continuation, length + backward, len(tokens) - continuation_begin)
            if continuation and (continuation_begin not in unique or
                                 unique[continuation_begin][1] < match[1]):
                unique[continuation_begin] = match
    matches = list(unique.values())
    if not matches:
        return [], []

    nodes = [{"children": {}, "support": 0, "weight": 0, "nearest": 1 << 60, "match": 0}]
    for continuation, match_length, distance in matches:
        node = 0
        for token in continuation:
            child = nodes[node]["children"].get(token)
            if child is None:
                child = len(nodes)
                nodes[node]["children"][token] = child
                nodes.append({"children": {}, "support": 0, "weight": 0, "nearest": 1 << 60, "match": 0})
            node = child
            nodes[node]["support"] += 1
            nodes[node]["weight"] += match_length
            nodes[node]["nearest"] = min(nodes[node]["nearest"], distance)
            nodes[node]["match"] = max(nodes[node]["match"], match_length)

    candidate = []
    confidence = []
    node = 0
    while nodes[node]["children"] and len(candidate) < n_max:
        children = list(nodes[node]["children"].items())
        if policy == "longest-recency":
            key = lambda item: (nodes[item[1]]["match"], -nodes[item[1]]["nearest"], -item[0])
        elif policy == "majority-prefix":
            key = lambda item: (nodes[item[1]]["support"], -nodes[item[1]]["nearest"], -item[0])
        else:
            key = lambda item: (nodes[item[1]]["weight"], -nodes[item[1]]["nearest"], -item[0])
        token, node = max(children, key=key)
        total = sum(nodes[index]["support"] for _, index in children)
        if confidence:
            previous = confidence[-1]
            previous_support, previous_distance, previous_match = previous[1:]
            source_jump = nodes[node]["nearest"] > previous_distance + 32
            support_collapse = nodes[node]["support"] < previous_support
            context_collapse = nodes[node]["match"] + 8 <= previous_match
            if len(candidate) >= 3 and source_jump and (support_collapse or context_collapse):
                break
        candidate.append(token)
        confidence.append((nodes[node]["support"] / max(1, total), nodes[node]["support"],
                           nodes[node]["nearest"], nodes[node]["match"]))
    return candidate, [item[0] for item in confidence]


def evaluate(workloads, n_max, query_us, verify_us):
    report = {}
    for policy in POLICIES:
        cycles = yielded = proposed = accepted = 0
        squared_error = 0.0
        calibrated_positions = 0
        per_workload = {}
        for workload in workloads:
            prompt = list(workload["prompt_tokens"])
            output = list(workload["output_tokens"])
            local_cycles = local_yielded = local_proposed = local_accepted = 0
            for position in range(len(output)):
                history = prompt + output[:position]
                candidate, confidence = retrieve(history, policy, n_max)
                cycles += 1
                local_cycles += 1
                if not candidate:
                    continue
                yielded += 1
                local_yielded += 1
                truth = output[position:position + len(candidate)]
                matched = 0
                while matched < len(truth) and candidate[matched] == truth[matched]:
                    matched += 1
                proposed += len(candidate)
                accepted += matched
                local_proposed += len(candidate)
                local_accepted += matched
                for index, probability in enumerate(confidence):
                    outcome = 1.0 if index < matched else 0.0
                    squared_error += (probability - outcome) ** 2
                    calibrated_positions += 1
                    if index >= matched:
                        break
            per_workload[workload["name"]] = {
                "cycles": local_cycles,
                "candidate_coverage": local_yielded / max(1, local_cycles),
                "proposed_tokens": local_proposed,
                "accepted_tokens": local_accepted,
                "acceptance": local_accepted / max(1, local_proposed),
            }
        total_time = cycles * query_us + proposed * verify_us
        report[policy] = {
            "cycles": cycles,
            "candidate_coverage": yielded / max(1, cycles),
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "acceptance": accepted / max(1, proposed),
            "accepted_tokens_per_cycle": accepted / max(1, cycles),
            "calibration_brier": squared_error / max(1, calibrated_positions),
            "estimated_goodput_tokens_per_us": accepted / max(1.0, total_time),
            "workloads": per_workload,
        }
    eligible = [name for name in POLICIES if report[name]["acceptance"] >= 0.70]
    pool = eligible or list(POLICIES)
    selected = max(pool, key=lambda name: report[name]["estimated_goodput_tokens_per_us"])
    return {"selected_policy": selected, "policies": report}


def load_workloads(path, config):
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        selected = [item for item in records if item.get("config") == config]
        if not selected:
            raise SystemExit(f"no {config!r} records in {path}")
        missing = [item["workload"] for item in selected if "prompt_tokens" not in item]
        if missing:
            raise SystemExit("benchmark records lack prompt_tokens; rerun with the current run_bench.py")
        return [{
            "name": item["workload"],
            "prompt_tokens": item["prompt_tokens"],
            "output_tokens": item["tokens"],
        } for item in selected]
    return data.get("workloads", data) if isinstance(data, dict) else data


def main():
    parser = argparse.ArgumentParser(description="Evaluate ngram-retrieval at every output position")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="all")
    parser.add_argument("--n-max", type=int, default=48, choices=range(8, 49), metavar="8..48")
    parser.add_argument("--query-us", type=float, default=25.0)
    parser.add_argument("--verify-us-per-token", type=float, default=1000.0)
    parser.add_argument("--config", default="mtp", help="configuration to use when corpus is benchmark JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy-header", type=Path,
                        help="write the selected development policy as a C++ header")
    args = parser.parse_args()

    workloads = load_workloads(args.corpus, args.config)
    selected_names = DEVELOPMENT if args.split == "development" else HOLDOUT if args.split == "holdout" else None
    if selected_names is not None:
        workloads = [item for item in workloads if item["name"] in selected_names]
    if not workloads:
        raise SystemExit("no workloads selected")
    report = evaluate(workloads, args.n_max, args.query_us, args.verify_us_per_token)
    report["split"] = args.split
    report["workload_names"] = [item["name"] for item in workloads]
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    if args.policy_header:
        symbols = {
            "longest-recency": "COMMON_NGRAM_RETRIEVAL_LONGEST_RECENT",
            "majority-prefix": "COMMON_NGRAM_RETRIEVAL_MAJORITY_PREFIX",
            "weighted-prefix": "COMMON_NGRAM_RETRIEVAL_WEIGHTED_PREFIX",
        }
        args.policy_header.write_text(
            "#pragma once\n\n"
            "// Generated by benchmark/retrieval_evaluator.py from development workloads.\n"
            "constexpr common_ngram_retrieval_policy COMMON_NGRAM_RETRIEVAL_DEVELOPMENT_POLICY =\n"
            f"    {symbols[report['selected_policy']]};\n"
        )


if __name__ == "__main__":
    main()
