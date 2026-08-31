#!/usr/bin/env python3

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = PROJECT_ROOT / "launchers" / "Q3-bench.sh"
DEFAULT_CORPUS = PROJECT_ROOT / "benchmark" / "corpus.json"
DEFAULT_MATRIX = PROJECT_ROOT / "benchmark" / "matrix.json"
METRIC_PREFIXES = (
    "llamacpp:prompt_",
    "llamacpp:tokens_predicted_",
    "llamacpp:predicted_tokens_",
    "llamacpp:n_decode_",
    "llamacpp:spec_decode_",
    "llamacpp:spec_controller_",
    "llamacpp:spec_retrieval_",
)


def load_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def request(url, payload=None, timeout=10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def port_is_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(process, port, timeout, log_path):
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = tail_text(log_path)
            raise RuntimeError(f"server exited with {process.returncode}\n{tail}")
        try:
            health = json.loads(request(url, timeout=2))
            if health.get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready within {timeout}s\n{tail_text(log_path)}")


def tail_text(path, count=40):
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-count:])
    except OSError:
        return ""


def stop_server(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def parse_metrics(text):
    metrics = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            key, raw_value = line.rsplit(None, 1)
            name = key.split("{", 1)[0]
            if not name.startswith(METRIC_PREFIXES):
                continue
            metrics[key] = float(raw_value)
        except (ValueError, IndexError):
            continue
    return metrics


def metric_delta(before, after):
    result = {}
    for key, value in after.items():
        if key.endswith("_total") or "_total{" in key:
            result[key] = value - before.get(key, 0.0)
        else:
            result[key] = value
    return result


def process_rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def process_cpu_ms(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        ticks = int(fields[13]) + int(fields[14])
        return ticks * 1000.0 / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def vram_usage():
    result = {}
    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        try:
            result[str(path)] = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return result


def trace_offset(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def output_hash(tokens):
    encoded = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT / "llama.cpp-stage5-port",
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_named(items, explicit, suite, tag=None):
    if explicit:
        names = set(explicit.split(","))
        selected = [item for item in items if item["name"] in names]
        missing = names - {item["name"] for item in selected}
        if missing:
            raise ValueError(f"unknown names: {', '.join(sorted(missing))}")
        return selected
    if tag is not None:
        return [item for item in items if tag in item.get("tags", [])]
    return [item for item in items if suite in item.get("suites", [])]


def run_request(port, workload, seed, timeout, max_tokens_override, sampling, controller_namespace=None):
    n_predict = max_tokens_override or workload["max_tokens"]
    payload = {
        "prompt": workload["prompt"],
        "n_predict": n_predict,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "min_p": sampling["min_p"],
        "seed": seed,
        "return_tokens": True,
        "cache_prompt": False,
    }
    for field in ("stop", "grammar", "json_schema"):
        if field in workload:
            payload[field] = workload[field]
    if controller_namespace is not None:
        payload["speculative_controller_namespace"] = controller_namespace
    started = time.perf_counter()
    raw = request(f"http://127.0.0.1:{port}/completion", payload, timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    response = json.loads(raw)
    if "error" in response:
        raise RuntimeError(json.dumps(response["error"], sort_keys=True))
    return response, wall_ms, n_predict


def tokenize_prompt(port, prompt):
    response = json.loads(request(
        f"http://127.0.0.1:{port}/tokenize",
        {"content": prompt, "add_special": True},
        timeout=30,
    ))
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("tokenize endpoint did not return a token list")
    return tokens


def correctness_report(records, reference):
    grouped = {}
    for record in records:
        key = (record["workload"], record["repetition"])
        grouped.setdefault(key, {})[record["config"]] = record["output_hash"]

    mismatches = []
    missing_reference = []
    for (workload, repetition), outputs in sorted(grouped.items()):
        expected = outputs.get(reference)
        if expected is None:
            missing_reference.append({"workload": workload, "repetition": repetition})
            continue
        for config, actual in sorted(outputs.items()):
            if actual != expected:
                mismatches.append({
                    "workload": workload,
                    "repetition": repetition,
                    "reference": reference,
                    "config": config,
                    "expected_hash": expected,
                    "actual_hash": actual,
                })
    by_config_workload_seed = collections.defaultdict(set)
    for record in records:
        by_config_workload_seed[(
            record["config"],
            record["workload"],
            record["request_seed"],
        )].add(record["output_hash"])
    unstable = [
        {
            "config": config,
            "workload": workload,
            "request_seed": seed,
            "distinct_output_hashes": len(hashes),
        }
        for (config, workload, seed), hashes in sorted(by_config_workload_seed.items())
        if len(hashes) > 1
    ]

    return {
        "reference": reference,
        "comparisons": sum(max(0, len(outputs) - 1) for outputs in grouped.values()),
        "mismatches": mismatches,
        "missing_reference": missing_reference,
        "within_config_instability": unstable,
        "passed": not mismatches and not missing_reference and not unstable,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive MTP + n-gram speculation")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--suite", choices=("smoke", "core", "full"), default="smoke")
    parser.add_argument("--configs", help="comma-separated configuration names")
    parser.add_argument("--workloads", help="comma-separated workload names")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vary-seed-by-repetition", action="store_true",
                        help="use seed + repetition so stochastic runs cover multiple exact-comparison seeds")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--ctx-size", type=int, default=97280)
    parser.add_argument("--max-tokens", type=int, help="override every workload token limit")
    parser.add_argument("--server-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--shuffle-configs", action="store_true")
    parser.add_argument("--request-order", choices=("round-robin", "grouped"), default="round-robin",
                        help="interleave repetitions across workloads or repeat each workload immediately")
    parser.add_argument("--server-scope", choices=("config", "request"), default="config",
                        help="reuse one server per configuration or restart for every measured request")
    parser.add_argument("--allow-output-drift", action="store_true",
                        help="record token mismatches without returning failure")
    parser.add_argument("--no-controller-trace", action="store_true",
                        help="disable controller JSONL tracing to measure production-path overhead")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.temperature < 0:
        parser.error("--temperature must not be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must not be negative")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not port_is_available(args.port):
        parser.error(f"port {args.port} is already in use")

    configs = select_named(load_json(args.matrix)["configs"], args.configs, args.suite)
    all_workloads = load_json(args.corpus)["workloads"]
    if args.workloads:
        workloads = select_named(all_workloads, args.workloads, args.suite)
    elif args.suite == "full":
        workloads = all_workloads
    else:
        workloads = [item for item in all_workloads if args.suite in item.get("tags", [])]
    if not configs or not workloads:
        parser.error("configuration and workload selections must not be empty")
    if args.shuffle_configs and args.server_scope != "request":
        random.Random(args.seed).shuffle(configs)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "results.jsonl"
    manifest = {
        "created_utc": timestamp,
        "llama_cpp_revision": git_revision(),
        "launcher": str(args.launcher.resolve()),
        "launcher_sha256": sha256_file(args.launcher),
        "source_launcher_sha256": sha256_file(Path("/home/pc/launchers/Q3.sh")),
        "suite": args.suite,
        "seed": args.seed,
        "vary_seed_by_repetition": args.vary_seed_by_repetition,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        },
        "ctx_size": args.ctx_size,
        "repetitions": args.repetitions,
        "persistent_server_within_config": args.server_scope == "config",
        "server_scope": args.server_scope,
        "request_order": args.request_order,
        "shuffle_configs": args.shuffle_configs,
        "shuffle_scope": "balanced-workload-rotation" if args.shuffle_configs and args.server_scope == "request" else "campaign",
        "controller_trace": not args.no_controller_trace,
        "configs": configs,
        "workloads": workloads,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    records = []
    if args.server_scope == "request":
        sessions_dir = output_dir / "sessions"
        sessions_dir.mkdir()
        cases = []
        for repetition in range(args.repetitions):
            for workload in workloads:
                cell_configs = list(configs)
                if args.shuffle_configs:
                    random.Random(f"{args.seed}:{workload['name']}").shuffle(cell_configs)
                    offset = repetition % len(cell_configs)
                    cell_configs = cell_configs[offset:] + cell_configs[:offset]
                cases.extend((config, workload, repetition) for config in cell_configs)
        for case_index, (config, workload, repetition) in enumerate(cases):
            request_seed = args.seed + repetition if args.vary_seed_by_repetition else args.seed
            session_rel = Path("sessions") / config["name"] / f"{workload['name']}-r{repetition}"
            session_dir = output_dir / session_rel
            session_dir.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--corpus", str(args.corpus),
                "--matrix", str(args.matrix),
                "--launcher", str(args.launcher),
                "--suite", args.suite,
                "--configs", config["name"],
                "--workloads", workload["name"],
                "--repetitions", "1",
                "--seed", str(request_seed),
                "--temperature", str(args.temperature),
                "--top-p", str(args.top_p),
                "--top-k", str(args.top_k),
                "--min-p", str(args.min_p),
                "--port", str(args.port),
                "--ctx-size", str(args.ctx_size),
                "--server-timeout", str(args.server_timeout),
                "--request-timeout", str(args.request_timeout),
                "--server-scope", "config",
                "--output-dir", str(session_dir),
                "--allow-output-drift",
            ]
            if args.max_tokens:
                command.extend(("--max-tokens", str(args.max_tokens)))
            if args.no_controller_trace:
                command.append("--no-controller-trace")
            print(
                f"[{case_index + 1}/{len(cases)}] cold {config['name']} / {workload['name']} / rep {repetition + 1}",
                flush=True,
            )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
            with (session_dir / "results.jsonl").open(encoding="utf-8") as source:
                record = json.loads(next(line for line in source if line.strip()))
            record["repetition"] = repetition
            if record.get("trace"):
                record["trace"] = str(session_rel / record["trace"])
            records.append(record)
            with records_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, separators=(",", ":")) + "\n")

        reference = "mtp" if any(config["name"] == "mtp" for config in configs) else configs[0]["name"]
        correctness = correctness_report(records, reference)
        (output_dir / "correctness.json").write_text(
            json.dumps(correctness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"results: {output_dir}")
        print(f"correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
        if not correctness["passed"] and not args.allow_output_drift:
            sys.exit(1)
        return

    for config_index, config in enumerate(configs):
        config_dir = output_dir / config["name"]
        config_dir.mkdir()
        log_path = config_dir / "server.log"
        trace_path = config_dir / "controller.jsonl"
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config["env"].items()})
        env.update({
            "SERVER_PORT": str(args.port),
            "CTX_SIZE": str(args.ctx_size),
            "CONTROLLER_TRACE": "" if args.no_controller_trace else str(trace_path),
        })
        replay = config.get("replay")
        if replay:
            replay_path = config_dir / "decisions.replay"
            replay_path.write_text(
                (f"{int(replay['producer_id'])} {int(replay['prefix_length'])}\n") * 4096,
                encoding="ascii",
            )
            env["SPEC_CONTROLLER"] = "replay"
            env["CONTROLLER_REPLAY"] = str(replay_path)

        print(f"[{config_index + 1}/{len(configs)}] loading {config['name']}", flush=True)
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                [str(args.launcher)],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_ready(process, args.port, args.server_timeout, log_path)
                idle_cpu_start_ms = process_cpu_ms(process.pid)
                time.sleep(2.0)
                idle_cpu_end_ms = process_cpu_ms(process.pid)
                idle_server_cpu_ms = idle_cpu_end_ms - idle_cpu_start_ms if idle_cpu_start_ms is not None and idle_cpu_end_ms is not None else None
                if args.request_order == "round-robin":
                    request_cases = [
                        (workload_index, workload, repetition)
                        for repetition in range(args.repetitions)
                        for workload_index, workload in enumerate(workloads)
                    ]
                else:
                    request_cases = [
                        (workload_index, workload, repetition)
                        for workload_index, workload in enumerate(workloads)
                        for repetition in range(args.repetitions)
                    ]
                for case_index, (workload_index, workload, repetition) in enumerate(request_cases):
                        request_seed = args.seed + repetition if args.vary_seed_by_repetition else args.seed
                        label = f"{workload['name']} rep {repetition + 1}/{args.repetitions}"
                        print(f"  [{case_index + 1}/{len(request_cases)}] {label}", flush=True)
                        metrics_before = parse_metrics(request(
                            f"http://127.0.0.1:{args.port}/metrics", timeout=10
                        ))
                        prompt_tokens = tokenize_prompt(args.port, workload["prompt"])
                        trace_start = trace_offset(trace_path)
                        cpu_start_ms = process_cpu_ms(process.pid)
                        response, wall_ms, requested_tokens = run_request(
                            args.port,
                            workload,
                            request_seed,
                            args.request_timeout,
                            args.max_tokens,
                            manifest["sampling"],
                        )
                        trace_end = trace_offset(trace_path)
                        cpu_end_ms = process_cpu_ms(process.pid)
                        metrics_after = parse_metrics(request(
                            f"http://127.0.0.1:{args.port}/metrics", timeout=10
                        ))
                        tokens = response.get("tokens", [])
                        record = {
                            "config": config["name"],
                            "workload": workload["name"],
                            "workload_tags": workload.get("tags", []),
                            "expected_affinity": workload.get("expected_affinity", []),
                            "repetition": repetition,
                            "request_seed": request_seed,
                            "requested_tokens": requested_tokens,
                            "prompt_tokens": prompt_tokens,
                            "wall_ms": wall_ms,
                            "server_cpu_ms": cpu_end_ms - cpu_start_ms if cpu_start_ms is not None and cpu_end_ms is not None else None,
                            "idle_server_cpu_ms": idle_server_cpu_ms,
                            "tokens": tokens,
                            "output_hash": output_hash(tokens),
                            "content": response.get("content", ""),
                            "stop": response.get("stop"),
                            "stopped_eos": response.get("stopped_eos"),
                            "stopped_limit": response.get("stopped_limit"),
                            "stopping_word": response.get("stopping_word", ""),
                            "timings": response.get("timings", {}),
                            "metrics_delta": metric_delta(metrics_before, metrics_after),
                            "server_rss_bytes": process_rss_bytes(process.pid),
                            "vram_used_bytes": vram_usage(),
                            "trace": str(trace_path.relative_to(output_dir)) if trace_path.exists() else None,
                            "trace_start": trace_start,
                            "trace_end": trace_end,
                        }
                        records.append(record)
                        with records_path.open("a", encoding="utf-8") as sink:
                            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            finally:
                stop_server(process)

    reference = "mtp" if any(config["name"] == "mtp" for config in configs) else configs[0]["name"]
    correctness = correctness_report(records, reference)
    (output_dir / "correctness.json").write_text(
        json.dumps(correctness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"results: {output_dir}")
    print(f"correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
    if not correctness["passed"] and not args.allow_output_drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
