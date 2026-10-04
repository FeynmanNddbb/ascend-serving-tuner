#!/usr/bin/env python3
"""Adaptive coordinate search for vLLM / vLLM-Ascend serving."""
from __future__ import annotations
import argparse, csv, itertools, json, os, signal, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def combinations(server):
    keys=list(server)
    for vals in itertools.product(*(server[k] for k in keys)): yield dict(zip(keys,vals))
def shell_env(c):
    env=os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"]=str(c.get("visible_devices","0,1"))
    env.update({str(k):str(v) for k,v in c.get("env",{}).items()})
    return env
def make_server_cmd(c,s):
    cmd=["vllm","serve",c["model"],"--host",c.get("host","127.0.0.1"),"--port",str(c.get("port",8000)),
         "--served-model-name",c.get("served_model_name","model"),"--tensor-parallel-size",str(s["tensor_parallel_size"]),
         "--max-model-len",str(s["max_model_len"]),"--max-num-seqs",str(s["max_num_seqs"]),
         "--max-num-batched-tokens",str(s["max_num_batched_tokens"]),
         "--gpu-memory-utilization",str(s["gpu_memory_utilization"])]
    if s.get("enable_chunked_prefill"): cmd.append("--enable-chunked-prefill")
    return cmd
def wait_health(url,proc,timeout):
    end=time.time()+timeout
    while time.time()<end:
        if proc.poll() is not None: return False,f"server exited {proc.returncode}"
        try:
            with urllib.request.urlopen(url,timeout=3) as r:
                if r.status==200:return True,"ready"
        except Exception:pass
        time.sleep(3)
    return False,"health check timeout"
def stop_process(proc,timeout):
    if proc is None or proc.poll() is not None:return
    proc.send_signal(signal.SIGINT)
    try:proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill()
def parse_metrics(path):
    p=Path(path)
    if not p.exists():return {}
    try:d=json.loads(p.read_text(encoding="utf-8"))
    except Exception:return {}
    if isinstance(d,list) and d:d=d[-1]
    if not isinstance(d,dict):return {}
    aliases={"request_throughput":["request_throughput","req_throughput"],
      "output_throughput":["output_throughput","output_token_throughput"],
      "mean_ttft_ms":["mean_ttft_ms","ttft_mean_ms"],"mean_tpot_ms":["mean_tpot_ms","tpot_mean_ms"],
      "mean_itl_ms":["mean_itl_ms","itl_mean_ms"],"completed":["successful_requests","completed","num_completed"],
      "failed":["failed_requests","failed","num_failed"]}
    out={}
    for dst,names in aliases.items():
        for name in names:
            if name in d:out[dst]=d[name];break
    return out

class Runner:
    def __init__(self,c,base):self.c,self.base,self.rows,self.idx=c,base,[],0
    def trial(self,s,input_len,conc,label=""):
        self.idx+=1; tid=f"trial_{self.idx:04d}"; d=self.base/tid;d.mkdir(parents=True,exist_ok=True)
        (d/"config.json").write_text(json.dumps({"server":s,"input_len":input_len,"client_concurrency":conc,"label":label},indent=2))
        row={"trial":tid,**s,"input_len":input_len,"client_concurrency":conc,"label":label,"status":"error"}
        proc=None
        with open(d/"server.log","w",encoding="utf-8") as sl,open(d/"benchmark.log","w",encoding="utf-8") as bl:
            try:
                cmd=make_server_cmd(self.c,s);(d/"server_command.txt").write_text(" ".join(cmd))
                proc=subprocess.Popen(cmd,stdout=sl,stderr=subprocess.STDOUT,env=shell_env(self.c),start_new_session=True)
                ok,msg=wait_health(f"http://{self.c.get('host','127.0.0.1')}:{self.c.get('port',8000)}/v1/models",proc,self.c.get("startup_timeout_sec",900))
                if not ok:raise RuntimeError(msg)
                b=self.c["benchmark"]; result=d/"benchmark.json"
                bench=["vllm","bench","serve","--backend",b.get("backend","openai-chat"),
                  "--host",self.c.get("host","127.0.0.1"),"--port",str(self.c.get("port",8000)),
                  "--endpoint","/v1/chat/completions","--model",self.c["model"],
                  "--served-model-name",self.c.get("served_model_name","model"),
                  "--dataset-name",b.get("dataset_name","random"),"--random-input-len",str(input_len),
                  "--random-output-len",str(b["output_len"]),"--num-prompts",str(b["num_prompts"]),
                  "--max-concurrency",str(conc),"--save-result","--result-filename",str(result)]
                bench += [str(x) for x in b.get("extra_args",[])]
                rc=subprocess.run(bench,stdout=bl,stderr=subprocess.STDOUT,env=shell_env(self.c)).returncode
                row.update(parse_metrics(result));row["status"]="ok" if rc==0 else f"bench_exit_{rc}"
                if rc:row["error"]=f"benchmark exited {rc}"
                if row.get("failed",0) not in (0,None):row["status"]="request_failures"
            except Exception as e:row["error"]=str(e);print(f"[{tid}] ERROR: {e}",file=sys.stderr)
            finally:stop_process(proc,self.c.get("shutdown_timeout_sec",30))
        self.rows.append(row);self.write_summary()
        print(f"[{self.idx}] {tid}: {row['status']} input={input_len} concurrency={conc}")
        return row
    def write_summary(self):
        if self.rows:
            with open(self.base/"summary.csv","w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=sorted({k for r in self.rows for k in r}));w.writeheader();w.writerows(self.rows)

def satisfies(row,limits):
    if row.get("status")!="ok":return False
    checks=[("min_output_throughput","output_throughput",lambda x,y:x>=y),
            ("max_mean_ttft_ms","mean_ttft_ms",lambda x,y:x<=y),
            ("max_mean_tpot_ms","mean_tpot_ms",lambda x,y:x<=y)]
    for limit,key,fn in checks:
        bound=limits.get(limit)
        if bound is not None:
            value=row.get(key)
            if value is None or not fn(float(value),float(bound)):return False
    return True
def score(row,objective):
    key={"throughput":"output_throughput","request_throughput":"request_throughput","latency":"mean_tpot_ms"}.get(objective,"output_throughput")
    v=row.get(key)
    if v is None:return None
    return -float(v) if objective=="latency" else float(v)

def adaptive(c,runner):
    a=c.get("auto_tune",{}); lim=a.get("limits",{})
    # Defaults intentionally permissive: throughput has no minimum; latency caps are broad.
    limits={"min_output_throughput":0,"max_mean_ttft_ms":10000,"max_mean_tpot_ms":1000,**lim}
    objective=a.get("objective","throughput")
    out_len=int(c["benchmark"].get("output_len",128))
    # Search starts from the lowest configured value for each dimension.
    space=a.get("search_space",{})
    defaults={"context_lengths":[4096,8192,16384,32768,65536,131072,262144],
      "max_num_seqs":[1,2,4,8,16,32],"max_num_batched_tokens":[1024,2048,4096,8192,16384],
      "gpu_memory_utilization":[0.80,0.85,0.90,0.93]}
    for k,v in defaults.items():space.setdefault(k,v)
    dims=["context_lengths","max_num_seqs","max_num_batched_tokens","gpu_memory_utilization"]
    for k in dims:
        space[k]=sorted(set(space[k]))
        if not space[k]:raise ValueError(f"search_space.{k} cannot be empty")
    tp=int(a.get("tensor_parallel_size",c.get("server",{}).get("tensor_parallel_size",[1])[0]))
    chunk=bool(a.get("enable_chunked_prefill",True))
    concs=sorted(set(int(x) for x in a.get("benchmark_concurrency",[1,2,4,8])))
    # Coordinate ascent: test one-step neighbors from the minimum feasible point, move only on score improvement.
    idx={k:0 for k in dims}; seen={}; max_trials=int(a.get("max_trials",40)); rounds=0
    def evaluate(pos):
        key=tuple(pos[k] for k in dims)
        if key in seen:return seen[key]
        if len(seen)>=max_trials:return None
        context=space["context_lengths"][pos["context_lengths"]]
        seqs=space["max_num_seqs"][pos["max_num_seqs"]]
        tokens=space["max_num_batched_tokens"][pos["max_num_batched_tokens"]]
        mem=space["gpu_memory_utilization"][pos["gpu_memory_utilization"]]
        s={"tensor_parallel_size":tp,"max_model_len":int(context)+out_len,"max_num_seqs":int(seqs),
           "max_num_batched_tokens":int(tokens),"gpu_memory_utilization":float(mem),"enable_chunked_prefill":chunk}
        # Probe at concurrency 1 for stable constraint measurements; then configured levels.
        rows=[runner.trial(s,int(context),conc,"adaptive_coordinate_search") for conc in concs]
        valid=[r for r in rows if satisfies(r,limits)]
        scored=[(score(r,objective),r) for r in valid]
        scored=[x for x in scored if x[0] is not None]
        result=max(scored,key=lambda x:x[0]) if scored else (None,rows[0] if rows else None)
        seen[key]=result
        return result
    current=evaluate(idx)
    if current is None or current[0] is None:
        raise RuntimeError("Lowest candidate did not satisfy constraints or produced no parseable objective metric. Check logs/limits.")
    improved=True
    while improved and len(seen)<max_trials:
        improved=False; rounds+=1
        for dim in dims:
            for delta in (-1,1):
                nxt=idx.copy();nxt[dim]+=delta
                if not 0<=nxt[dim]<len(space[dim]):continue
                candidate=evaluate(nxt)
                if candidate and candidate[0] is not None and candidate[0]>current[0]:
                    idx,current=nxt,candidate;improved=True
        if rounds>=int(a.get("max_rounds",12)):break
    best_context=space["context_lengths"][idx["context_lengths"]]
    best_seq=space["max_num_seqs"][idx["max_num_seqs"]]
    best_tokens=space["max_num_batched_tokens"][idx["max_num_batched_tokens"]]
    best_mem=space["gpu_memory_utilization"][idx["gpu_memory_utilization"]]
    best={"tensor_parallel_size":tp,"max_model_len":int(best_context)+out_len,"max_num_seqs":int(best_seq),
          "max_num_batched_tokens":int(best_tokens),"gpu_memory_utilization":float(best_mem),"enable_chunked_prefill":chunk}
    chosen_row=current[1]
    rec={"objective":objective,"limits":limits,"recommended_server":best,
         "benchmark_concurrency":chosen_row.get("client_concurrency"),
         "selected_metrics":{k:chosen_row.get(k) for k in ("output_throughput","request_throughput","mean_ttft_ms","mean_tpot_ms")},
         "trials_evaluated":len(runner.rows),"search_method":"discrete coordinate hill-climb from minimum candidate values",
         "note":"Recommendation is local to configured discrete candidates and this benchmark workload."}
    (runner.base/"recommendation.json").write_text(json.dumps(rec,indent=2))
    print("Recommendation saved:",runner.base/"recommendation.json")
    return best

def launch_best(c,s,base):
    log=open(base/"final_server.log","w",encoding="utf-8")
    cmd=make_server_cmd(c,s)
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=shell_env(c),start_new_session=True)
    ok,msg=wait_health(f"http://{c.get('host','127.0.0.1')}:{c.get('port',8000)}/v1/models",proc,c.get("startup_timeout_sec",900))
    if not ok:
        stop_process(proc,c.get("shutdown_timeout_sec",30));log.close()
        raise RuntimeError(f"best-config service failed to start: {msg}; see final_server.log")
    (base/"final_server.json").write_text(json.dumps({"pid":proc.pid,"command":cmd,"server":s},indent=2))
    # Deliberately leave the final server running; detach the log descriptor from Python.
    log.close()
    print(f"Best config service is running (pid={proc.pid}) on port {c.get('port',8000)}")

def main():
    ap=argparse.ArgumentParser(description="Adaptive/grid vLLM serving tuner")
    ap.add_argument("--config",default="config.json");ap.add_argument("--mode",choices=["grid","adaptive"])
    ap.add_argument("--dry-run",action="store_true");ap.add_argument("--no-launch-best",action="store_true")
    args=ap.parse_args();c=read_json(args.config);mode=args.mode or c.get("mode","adaptive")
    base=Path("runs")/datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        print(f"mode={mode}; config={args.config}")
        if mode=="adaptive":print(json.dumps(c.get("auto_tune",{}),indent=2))
        else:
            for s in combinations(c["server"]):print("SERVER:"," ".join(make_server_cmd(c,s)))
        return
    base.mkdir(parents=True,exist_ok=True);runner=Runner(c,base)
    if mode=="adaptive":
        best=adaptive(c,runner)
        if c.get("launch_best",True) and not args.no_launch_best:launch_best(c,best,base)
    else:
        for s in combinations(c["server"]):
            for n,conc in itertools.product(c["benchmark"]["input_lengths"],c["benchmark"]["max_concurrency"]):
                runner.trial(s,int(n),int(conc),"grid")
    print(f"Done. Results: {base.resolve()}")
if __name__=="__main__":main()
