#!/usr/bin/env python3
"""Adaptive coordinate search for vLLM / vLLM-Ascend serving."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def combinations(server):
    keys = list(server)
    for vals in itertools.product(*(server[k] for k in keys)):
        yield dict(zip(keys, vals))


def shell_env(c):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(c.get("visible_devices", "0,1"))
    env.update({str(k): str(v) for k, v in c.get("env", {}).items()})
    return env


def make_server_cmd(c, s):
    cmd = [
        "vllm", "serve", c["model"],
        "--host", c.get("host", "127.0.0.1"),
        "--port", str(c.get("port", 8000)),
        "--served-model-name", c.get("served_model_name", "model"),
        "--tensor-parallel-size", str(s["tensor_parallel_size"]),
        "--max-model-len", str(s["max_model_len"]),
        "--max-num-seqs", str(s["max_num_seqs"]),
        "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
    ]
    if s.get("enable_chunked_prefill"):
        cmd.append("--enable-chunked-prefill")
    return cmd


def wait_health(url, proc, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False, f"server exited {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True, "ready"
        except Exception:
            pass
        time.sleep(3)
    return False, "health check timeout"


def stop_process(proc, timeout):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_metrics(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, list) and data:
        data = data[-1]
    if not isinstance(data, dict):
        return {}
    aliases = {
        "request_throughput": ["request_throughput", "req_throughput"],
        "output_throughput": ["output_throughput", "output_token_throughput"],
        "mean_ttft_ms": ["mean_ttft_ms", "ttft_mean_ms"],
        "mean_tpot_ms": ["mean_tpot_ms", "tpot_mean_ms"],
        "mean_itl_ms": ["mean_itl_ms", "itl_mean_ms"],
        "completed": ["successful_requests", "completed", "num_completed"],
        "failed": ["failed_requests", "failed", "num_failed"],
    }
    out = {}
    for dst, names in aliases.items():
        for name in names:
            if name in data:
                out[dst] = data[name]
                break
    return out


class Runner:
    def __init__(self, config, base):
        self.c = config
        self.base = base
        self.rows = []
        self.idx = 0
        self.max_total_benchmarks = int(
            config.get("auto_tune", {}).get("max_total_benchmarks", 400)
        )

    def trial(self, server, input_len, concurrency, label=""):
        if self.idx >= self.max_total_benchmarks:
            raise RuntimeError(
                f"max_total_benchmarks={self.max_total_benchmarks} reached; "
                "increase the budget or reduce candidate/concurrency counts"
            )
        self.idx += 1
        tid = f"trial_{self.idx:04d}"
        directory = self.base / tid
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "server": server,
            "input_len": input_len,
            "client_concurrency": concurrency,
            "label": label,
        }
        (directory / "config.json").write_text(json.dumps(metadata, indent=2))
        row = {
            "trial": tid,
            **server,
            "input_len": input_len,
            "server_max_num_seqs": server["max_num_seqs"],
            "client_concurrency": concurrency,
            "label": label,
            "status": "error",
        }
        proc = None
        with open(directory / "server.log", "w", encoding="utf-8") as server_log, \
             open(directory / "benchmark.log", "w", encoding="utf-8") as benchmark_log:
            try:
                cmd = make_server_cmd(self.c, server)
                (directory / "server_command.txt").write_text(" ".join(cmd))
                proc = subprocess.Popen(
                    cmd, stdout=server_log, stderr=subprocess.STDOUT,
                    env=shell_env(self.c), start_new_session=True,
                )
                url = (
                    f"http://{self.c.get('host', '127.0.0.1')}:"
                    f"{self.c.get('port', 8000)}/v1/models"
                )
                ok, message = wait_health(
                    url, proc, self.c.get("startup_timeout_sec", 900)
                )
                if not ok:
                    raise RuntimeError(message)

                benchmark = self.c["benchmark"]
                result = directory / "benchmark.json"
                bench_cmd = [
                    "vllm", "bench", "serve",
                    "--backend", benchmark.get("backend", "openai-chat"),
                    "--host", self.c.get("host", "127.0.0.1"),
                    "--port", str(self.c.get("port", 8000)),
                    "--endpoint", "/v1/chat/completions",
                    "--model", self.c["model"],
                    "--served-model-name", self.c.get("served_model_name", "model"),
                    "--dataset-name", benchmark.get("dataset_name", "random"),
                    "--random-input-len", str(input_len),
                    "--random-output-len", str(benchmark["output_len"]),
                    "--num-prompts", str(benchmark["num_prompts"]),
                    "--max-concurrency", str(concurrency),
                    "--save-result", "--result-filename", str(result),
                ]
                bench_cmd += [str(x) for x in benchmark.get("extra_args", [])]
                completed = subprocess.run(
                    bench_cmd, stdout=benchmark_log, stderr=subprocess.STDOUT,
                    env=shell_env(self.c),
                )
                row.update(parse_metrics(result))
                row["status"] = "ok" if completed.returncode == 0 else f"bench_exit_{completed.returncode}"
                if completed.returncode:
                    row["error"] = f"benchmark exited {completed.returncode}"
                if row.get("failed", 0) not in (0, None):
                    row["status"] = "request_failures"
            except Exception as error:
                row["error"] = str(error)
                print(f"[{tid}] ERROR: {error}", file=sys.stderr)
            finally:
                stop_process(proc, self.c.get("shutdown_timeout_sec", 30))
        self.rows.append(row)
        self.write_summary()
        print(
            f"[{self.idx}] {tid}: {row['status']} "
            f"input={input_len} client_concurrency={concurrency} "
            f"server_max_num_seqs={server['max_num_seqs']}"
        )
        return row

    def write_summary(self):
        if self.rows:
            with open(self.base / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=sorted({key for row in self.rows for key in row})
                )
                writer.writeheader()
                writer.writerows(self.rows)


def satisfies(row, limits):
    if row.get("status") != "ok":
        return False
    checks = [
        ("min_output_throughput", "output_throughput", lambda x, y: x >= y),
        ("max_mean_ttft_ms", "mean_ttft_ms", lambda x, y: x <= y),
        ("max_mean_tpot_ms", "mean_tpot_ms", lambda x, y: x <= y),
    ]
    for limit_name, metric_name, compare in checks:
        bound = limits.get(limit_name)
        if bound is not None:
            value = row.get(metric_name)
            if value is None:
                return False
            try:
                if not compare(float(value), float(bound)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def rank_key(row, objective):
    """Higher tuple is better for the selected objective."""
    throughput = float(row.get("output_throughput", 0) or 0)
    req_rate = float(row.get("request_throughput", 0) or 0)
    tpot = row.get("mean_tpot_ms")
    if objective == "latency":
        return (-(float(tpot) if tpot is not None else float("inf")), throughput)
    if objective == "request_throughput":
        return (req_rate, throughput)
    return (throughput, req_rate)


def adaptive(config, runner):
    """Exhaustively evaluate the configured discrete search space.

    This finds the best measured candidate within the finite configured grid,
    subject to the benchmark budget and workload/SLO constraints.
    """
    auto = config.get("auto_tune", {})
    limits = {
        "min_output_throughput": 0,
        "max_mean_ttft_ms": 10000,
        "max_mean_tpot_ms": 1000,
        **auto.get("limits", {}),
    }
    objective = auto.get("objective", "throughput")
    if objective not in {"throughput", "request_throughput", "latency"}:
        raise ValueError("objective must be throughput, request_throughput, or latency")

    output_len = int(config["benchmark"].get("output_len", 128))
    space = auto.get("search_space", {})
    defaults = {
        "context_lengths": [4096, 8192, 16384, 32768, 65536, 131072, 262144],
        "max_num_seqs": [1, 2, 4, 8, 16, 32],
        "max_num_batched_tokens": [1024, 2048, 4096, 8192, 16384],
        "gpu_memory_utilization": [0.80, 0.85, 0.90, 0.93],
    }
    for key, values in defaults.items():
        space.setdefault(key, values)
    dims = list(defaults)
    for key in dims:
        space[key] = sorted(set(space[key]))
        if not space[key]:
            raise ValueError(f"search_space.{key} cannot be empty")

    total_candidates = 1
    for key in dims:
        total_candidates *= len(space[key])

    tp = int(auto.get("tensor_parallel_size", config.get("server", {}).get("tensor_parallel_size", [1])[0]))
    chunked = bool(auto.get("enable_chunked_prefill", True))
    max_trials = int(auto.get("max_trials", 0))  # 0 means exhaustive; positive value is an explicit cap
    repeats = int(auto.get("final_validation_repeats", 3))
    validation_top_k = int(auto.get("final_validation_top_k", 5))
    if max_trials < 0 or repeats < 1 or validation_top_k < 1:
        raise ValueError("max_trials must be >=0; final_validation_repeats and final_validation_top_k must be >=1")

    budget = min(total_candidates, max_trials) if max_trials else total_candidates
    if budget > runner.max_total_benchmarks:
        raise ValueError(
            f"Exhaustive search needs {budget} search benchmarks, but "
            f"max_total_benchmarks={runner.max_total_benchmarks}. Increase "
            "auto_tune.max_total_benchmarks or explicitly set auto_tune.max_trials "
            "to accept a partial search."
        )

    evaluated = []
    metric_by_objective = {
        "throughput": "output_throughput",
        "request_throughput": "request_throughput",
        "latency": "mean_tpot_ms",
    }

    for index, values in enumerate(itertools.product(*(space[key] for key in dims)), start=1):
        candidate = dict(zip(dims, values))
        context = int(candidate["context_lengths"])
        seqs = int(candidate["max_num_seqs"])
        server = {
            "tensor_parallel_size": tp,
            "max_model_len": context + output_len,
            "max_num_seqs": seqs,
            "max_num_batched_tokens": int(candidate["max_num_batched_tokens"]),
            "gpu_memory_utilization": float(candidate["gpu_memory_utilization"]),
            "enable_chunked_prefill": chunked,
        }
        print(f"\n[GRID {index}/{budget}] candidate={candidate}")
        row = runner.trial(server, context, seqs, "exhaustive_grid_search")
        metric = metric_by_objective[objective]
        feasible = satisfies(row, limits) and row.get(metric) is not None
        score = rank_key(row, objective) if feasible else None
        evaluated.append({"server": server, "row": row, "score": score, "feasible": feasible})

    ranked = [item for item in evaluated if item["feasible"]]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    if not ranked:
        raise RuntimeError(
            "No candidate passed SLO/metric checks. Inspect runs/*/trial_*/ logs "
            "or adjust the configured search space and limits."
        )

    # Revalidate the top K measured candidates. Select the best mean validation
    # score among candidates that pass every repeat, rather than accepting the
    # first candidate merely because it passed.
    validation_results = []
    for item in ranked[:validation_top_k]:
        server = item["server"]
        context = int(item["row"]["input_len"])
        load = int(server["max_num_seqs"])
        trials = [
            runner.trial(server, context, load, f"final_validation_{i + 1}")
            for i in range(repeats)
        ]
        valid = all(satisfies(row, limits) and row.get(metric_by_objective[objective]) is not None for row in trials)
        if valid:
            scores = [rank_key(row, objective) for row in trials]
            mean_score = tuple(
                sum(score[i] for score in scores) / len(scores)
                for i in range(len(scores[0]))
            )
            validation_results.append((mean_score, item, trials))

    if not validation_results:
        raise RuntimeError(
            "No top candidate passed every final validation repeat. "
            "Inspect logs or adjust noisy SLO limits."
        )

    validation_results.sort(key=lambda entry: entry[0], reverse=True)
    _, selected, validation_rows = validation_results[0]
    best_server = selected["server"]
    chosen_row = max(validation_rows, key=lambda row: rank_key(row, objective))

    recommendation = {
        "objective": objective,
        "limits": limits,
        "recommended_server": best_server,
        "benchmark_concurrency_rule": "equals recommended_server.max_num_seqs",
        "server_max_num_seqs": best_server["max_num_seqs"],
        "selected_metrics": {
            key: chosen_row.get(key)
            for key in ("output_throughput", "request_throughput", "mean_ttft_ms", "mean_tpot_ms")
        },
        "search_configs_evaluated": len(evaluated),
        "search_space_total_configs": total_candidates,
        "search_complete": len(evaluated) == total_candidates,
        "search_budget": budget,
        "benchmarks_executed": len(runner.rows),
        "final_validation_repeats": repeats,
        "final_validation_candidates": min(validation_top_k, len(ranked)),
        "final_validation_passed": True,
        "search_method": "exhaustive discrete grid search",
        "optimality_scope": (
            "Best measured feasible candidate within the configured discrete search space and workload; "
            "not a guarantee of continuous-space or noise-free global optimum."
        ),
        "note": (
            "Client benchmark concurrency equals candidate server max_num_seqs. "
            "Every configured candidate was tested unless max_trials explicitly capped the search."
        ),
    }
    (runner.base / "recommendation.json").write_text(json.dumps(recommendation, indent=2))
    print("Recommendation saved:", runner.base / "recommendation.json")
    return best_server


def launch_best(config, server, base):
    log = open(base / "final_server.log", "w", encoding="utf-8")
    cmd = make_server_cmd(config, server)
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        env=shell_env(config), start_new_session=True,
    )
    url = (
        f"http://{config.get('host', '127.0.0.1')}:"
        f"{config.get('port', 8000)}/v1/models"
    )
    ok, message = wait_health(url, proc, config.get("startup_timeout_sec", 900))
    if not ok:
        stop_process(proc, config.get("shutdown_timeout_sec", 30))
        log.close()
        raise RuntimeError(f"best-config service failed to start: {message}; see final_server.log")
    (base / "final_server.json").write_text(
        json.dumps({"pid": proc.pid, "command": cmd, "server": server}, indent=2)
    )
    log.close()
    print(f"Best config service is running (pid={proc.pid}) on port {config.get('port', 8000)}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive/grid vLLM serving tuner")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=["grid", "adaptive"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-launch-best", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    mode = args.mode or config.get("mode", "adaptive")
    base = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        print(f"mode={mode}; config={args.config}")
        if mode == "adaptive":
            print(json.dumps(config.get("auto_tune", {}), indent=2))
        else:
            for server in combinations(config["server"]):
                print("SERVER:", " ".join(make_server_cmd(config, server)))
        return
    base.mkdir(parents=True, exist_ok=True)
    runner = Runner(config, base)
    if mode == "adaptive":
        best = adaptive(config, runner)
        if config.get("launch_best", True) and not args.no_launch_best:
            launch_best(config, best, base)
    elif mode == "grid":
        for server in combinations(config["server"]):
            for input_len, concurrency in itertools.product(
                config["benchmark"]["input_lengths"],
                config["benchmark"]["max_concurrency"],
            ):
                runner.trial(server, int(input_len), int(concurrency), "grid")
    else:
        raise ValueError(f"unsupported mode: {mode}")
    print(f"Done. Results: {base.resolve()}")


if __name__ == "__main__":
    main()
