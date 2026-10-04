#!/usr/bin/env python3
"""Grid and adaptive parameter search for vLLM / vLLM-Ascend serving."""
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

def make_server_cmd(c, s):
    cmd = ["vllm", "serve", c["model"],
           "--host", c.get("host", "127.0.0.1"), "--port", str(c.get("port", 8000)),
           "--served-model-name", c.get("served_model_name", "model"),
           "--tensor-parallel-size", str(s["tensor_parallel_size"]),
           "--max-model-len", str(s["max_model_len"]),
           "--max-num-seqs", str(s["max_num_seqs"]),
           "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
           "--gpu-memory-utilization", str(s["gpu_memory_utilization"])]
    if s.get("enable_chunked_prefill"): cmd.append("--enable-chunked-prefill")
    return cmd

def wait_health(url, proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None: return False, f"server exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200: return True, "ready"
        except Exception: pass
        time.sleep(3)
    return False, "health check timeout"

def stop_process(proc, timeout):
    if proc is None or proc.poll() is not None: return
    proc.send_signal(signal.SIGINT)
    try: proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()

def parse_metrics(path):
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
            if name in data: out[dst] = data[name]; break
    return out

class Runner:
    def __init__(self, c, base):
        self.c, self.base, self.rows, self.idx = c, base, [], 0

    def trial(self, server_cfg, input_len, conc, label=""):
        self.idx += 1
        tid = f"trial_{self.idx:04d}"
        d = self.base / tid; d.mkdir(parents=True, exist_ok=True)
        (d/"config.json").write_text(json.dumps(
            {"server":server_cfg,"input_len":input_len,"client_concurrency":conc,"label":label}, indent=2))
        row = {"trial":tid, **server_cfg, "input_len":input_len,
               "client_concurrency":conc, "label":label, "status":"error"}
        proc = None
        with open(d/"server.log","w",encoding="utf-8") as sl, open(d/"benchmark.log","w",encoding="utf-8") as bl:
            try:
                cmd = make_server_cmd(self.c, server_cfg)
                (d/"server_command.txt").write_text(" ".join(cmd))
                proc = subprocess.Popen(cmd, stdout=sl, stderr=subprocess.STDOUT,
                                        env=shell_env(self.c), start_new_session=True)
                ok, msg = wait_health(
                    f"http://{self.c.get('host','127.0.0.1')}:{self.c.get('port',8000)}/v1/models",
                    proc, self.c.get("startup_timeout_sec",900))
                if not ok: raise RuntimeError(msg)
                b = self.c["benchmark"]
                result = d/"benchmark.json"
                bench = ["vllm","bench","serve","--backend",b.get("backend","openai-chat"),
                         "--host",self.c.get("host","127.0.0.1"),"--port",str(self.c.get("port",8000)),
                         "--endpoint","/v1/chat/completions","--model",self.c["model"],
                         "--served-model-name",self.c.get("served_model_name","model"),
                         "--dataset-name",b.get("dataset_name","random"),
                         "--random-input-len",str(input_len),"--random-output-len",str(b["output_len"]),
                         "--num-prompts",str(b["num_prompts"]),"--max-concurrency",str(conc),
                         "--save-result","--result-filename",str(result)]
                bench += [str(x) for x in b.get("extra_args",[])]
                rc = subprocess.run(bench, stdout=bl, stderr=subprocess.STDOUT, env=shell_env(self.c)).returncode
                row.update(parse_metrics(result))
                row["status"] = "ok" if rc == 0 else f"bench_exit_{rc}"
                if rc != 0: row["error"] = f"benchmark exited {rc}"
                if row.get("failed", 0) not in (0, None): row["status"] = "request_failures"
            except Exception as e:
                row["error"] = str(e)
                print(f"[{tid}] ERROR: {e}", file=sys.stderr)
            finally:
                stop_process(proc, self.c.get("shutdown_timeout_sec",30))
        self.rows.append(row)
        self.write_summary()
        print(f"[{self.idx}] {tid}: {row['status']} input={input_len} concurrency={conc}")
        return row

    def write_summary(self):
        if not self.rows: return
        with open(self.base/"summary.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in self.rows for k in r}))
            w.writeheader(); w.writerows(self.rows)

def adaptive(c, runner):
    a = c.get("auto_tune", {})
    start = int(a.get("start_context", 4096))
    target = int(a.get("max_context", 262144))
    out_len = int(c["benchmark"].get("output_len",128))
    if start < 1 or target < start: raise ValueError("auto_tune: require 1 <= start_context <= max_context")
    # Probe a monotonic ladder; binary search the largest successful input length.
    ladder = []
    x = start
    while x < target:
        ladder.append(x); x *= 2
    ladder.append(target)
    lo, hi, best = 0, len(ladder)-1, None
    fixed = c["server"]
    base = {
        "tensor_parallel_size": int(a.get("tensor_parallel_size", fixed.get("tensor_parallel_size",[1])[0])),
        "max_model_len": 0,
        "max_num_seqs": int(a.get("probe_max_num_seqs", fixed.get("max_num_seqs",[1])[0])),
        "max_num_batched_tokens": int(a.get("probe_max_num_batched_tokens", fixed.get("max_num_batched_tokens",[2048])[0])),
        "gpu_memory_utilization": float(a.get("gpu_memory_utilization", fixed.get("gpu_memory_utilization",[0.9])[0])),
        "enable_chunked_prefill": bool(a.get("enable_chunked_prefill", fixed.get("enable_chunked_prefill",[True])[0]))
    }
    while lo <= hi:
        mid=(lo+hi)//2
        input_len=ladder[mid]
        cfg={**base,"max_model_len":input_len+out_len}
        row=runner.trial(cfg,input_len,int(a.get("probe_concurrency",1)),label="adaptive_context_probe")
        passed=row["status"]=="ok"
        if passed:
            best={"input_len":input_len,"server":cfg,"metrics":row}
            lo=mid+1
        else: hi=mid-1
    if best is None:
        print("No successful context probe. Inspect trial logs; try a lower start_context or safer baseline.")
        return
    # Tune serving concurrency and batch-token cap only at the largest proven context.
    tune_seqs=a.get("tune_max_num_seqs",[base["max_num_seqs"]])
    tune_tokens=a.get("tune_max_num_batched_tokens",[base["max_num_batched_tokens"]])
    tune_conc=a.get("benchmark_concurrency",[1,2,4,8])
    candidates=[]
    for seqs,tokens in itertools.product(tune_seqs,tune_tokens):
        cfg={**best["server"],"max_num_seqs":int(seqs),"max_num_batched_tokens":int(tokens)}
        for conc in tune_conc:
            row=runner.trial(cfg,best["input_len"],int(conc),label="adaptive_throughput_tune")
            if row["status"]=="ok":
                score=row.get("output_throughput",row.get("request_throughput"))
                if score is not None: candidates.append((float(score),cfg,int(conc),row))
    if candidates:
        score,cfg,conc,row=max(candidates,key=lambda z:z[0])
        recommendation={"objective":"max_context_then_output_throughput",
                        "max_proven_input_len":best["input_len"],
                        "recommended_server":cfg,"benchmark_concurrency":conc,
                        "selected_metric":score,"selected_result":row}
        (runner.base/"recommendation.json").write_text(json.dumps(recommendation,indent=2))
        print("Recommendation saved:",runner.base/"recommendation.json")
    else:
        (runner.base/"recommendation.json").write_text(json.dumps({
            "objective":"max_context","max_proven_input_len":best["input_len"],
            "recommended_server":best["server"],
            "note":"No parseable throughput metric; inspect benchmark logs before selecting a throughput optimum."
        },indent=2))

def main():
    ap=argparse.ArgumentParser(description="Grid/adaptive serving parameter tuner")
    ap.add_argument("--config",default="config.json")
    ap.add_argument("--mode",choices=["grid","adaptive"],default=None)
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    c=read_json(args.config)
    mode=args.mode or c.get("mode","grid")
    base=Path("runs")/datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        if mode=="adaptive":
            a=c.get("auto_tune",{})
            print("Adaptive context ladder:",a.get("start_context",4096),"->",a.get("max_context",262144),"doubling probes")
            print("Probe config:",json.dumps(a,indent=2))
        else:
            trials=list(combinations(c["server"]))
            print(f"{len(trials)} server configs")
            for s in trials: print("SERVER:"," ".join(make_server_cmd(c,s)))
        return
    base.mkdir(parents=True,exist_ok=True)
    runner=Runner(c,base)
    if mode=="adaptive":
        adaptive(c,runner)
    else:
        for s in combinations(c["server"]):
            for input_len,conc in itertools.product(c["benchmark"]["input_lengths"],c["benchmark"]["max_concurrency"]):
                runner.trial(s,int(input_len),int(conc),label="grid")
    print(f"Done. Results: {base.resolve()}")

if __name__=="__main__": main()
