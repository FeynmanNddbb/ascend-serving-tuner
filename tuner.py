#!/usr/bin/env python3
"""Small, auditable server-parameter grid search for vLLM-Ascend."""
from __future__ import annotations
import argparse, csv, itertools, json, os, signal, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def combinations(server):
    keys = list(server)
    for vals in itertools.product(*(server[k] for k in keys)):
        yield dict(zip(keys, vals))

def shell_env(config):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(config.get("visible_devices", "0,1"))
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})
    return env

def make_server_cmd(c, trial):
    s = trial
    cmd = ["vllm", "serve", c["model"],
           "--host", c.get("host", "127.0.0.1"),
           "--port", str(c.get("port", 8000)),
           "--served-model-name", c.get("served_model_name", "model"),
           "--tensor-parallel-size", str(s["tensor_parallel_size"]),
           "--max-model-len", str(s["max_model_len"]),
           "--max-num-seqs", str(s["max_num_seqs"]),
           "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
           "--gpu-memory-utilization", str(s["gpu_memory_utilization"])]
    if s.get("enable_chunked_prefill"):
        cmd.append("--enable-chunked-prefill")
    return cmd

def wait_health(url, proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
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
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()

def parse_metrics(path):
    """Extract only known metric keys; retain raw result for other schemas."""
    p = Path(path)
    if not p.exists(): return {}
    try: data = json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}
    if isinstance(data, list) and data: data = data[-1]
    if not isinstance(data, dict): return {}
    aliases = {
        "request_throughput": ["request_throughput", "req_throughput"],
        "output_throughput": ["output_throughput", "output_token_throughput"],
        "mean_ttft_ms": ["mean_ttft_ms", "ttft_mean_ms"],
        "mean_tpot_ms": ["mean_tpot_ms", "tpot_mean_ms"],
        "mean_itl_ms": ["mean_itl_ms", "itl_mean_ms"],
        "completed": ["successful_requests", "completed", "num_completed"],
        "failed": ["failed_requests", "failed", "num_failed"]
    }
    out = {}
    for dst, names in aliases.items():
        for name in names:
            if name in data:
                out[dst] = data[name]; break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    c = read_json(args.config)
    base = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    trials = list(combinations(c["server"]))
    total = len(trials) * len(c["benchmark"]["input_lengths"]) * len(c["benchmark"]["max_concurrency"])
    print(f"{len(trials)} server configs; {total} benchmark cases")
    idx = 0
    for server_cfg in trials:
        if args.dry_run:
            print("SERVER:", " ".join(make_server_cmd(c, server_cfg)))
            continue
        for input_len, conc in itertools.product(c["benchmark"]["input_lengths"], c["benchmark"]["max_concurrency"]):
            idx += 1
            tid = f"trial_{idx:04d}"
            d = base / tid; d.mkdir()
            (d/"config.json").write_text(json.dumps({"server":server_cfg,"input_len":input_len,"concurrency":conc},indent=2))
            server_log = open(d/"server.log","w",encoding="utf-8")
            bench_log = open(d/"benchmark.log","w",encoding="utf-8")
            proc = None
            row = {"trial":tid, **server_cfg, "input_len":input_len, "client_concurrency":conc, "status":"error"}
            try:
                cmd = make_server_cmd(c, server_cfg)
                (d/"server_command.txt").write_text(" ".join(cmd))
                proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, env=shell_env(c), start_new_session=True)
                ok, msg = wait_health(f"http://{c.get('host','127.0.0.1')}:{c.get('port',8000)}/v1/models", proc, c.get("startup_timeout_sec",900))
                if not ok: raise RuntimeError(msg)
                b = c["benchmark"]
                result = d/"benchmark.json"
                bench = ["vllm","bench","serve","--backend",b.get("backend","openai-chat"),
                         "--host",c.get("host","127.0.0.1"),"--port",str(c.get("port",8000)),
                         "--endpoint","/v1/chat/completions","--model",c["model"],
                         "--served-model-name",c.get("served_model_name","model"),
                         "--dataset-name",b.get("dataset_name","random"),
                         "--random-input-len",str(input_len),"--random-output-len",str(b["output_len"]),
                         "--num-prompts",str(b["num_prompts"]),"--max-concurrency",str(conc),
                         "--save-result","--result-filename",str(result)]
                bench += [str(x) for x in b.get("extra_args",[])]
                rc = subprocess.run(bench, stdout=bench_log, stderr=subprocess.STDOUT, env=shell_env(c)).returncode
                row["status"] = "ok" if rc == 0 else f"bench_exit_{rc}"
                row.update(parse_metrics(result))
            except Exception as e:
                row["error"] = str(e)
                print(f"[{tid}] ERROR: {e}", file=sys.stderr)
            finally:
                stop_process(proc, c.get("shutdown_timeout_sec",30))
                server_log.close(); bench_log.close()
                rows.append(row)
                with open(base/"summary.csv","w",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
                    w.writeheader(); w.writerows(rows)
            print(f"[{idx}/{total}] {tid}: {row['status']}")
    print(f"Done. Results: {base.resolve()}")

if __name__ == "__main__": main()
