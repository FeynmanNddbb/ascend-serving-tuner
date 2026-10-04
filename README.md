<div align="center">

# ⚡ Ascend Serving Tuner

**Adaptive parameter search for vLLM / vLLM-Ascend inference serving**

从最低候选参数开始，通过离散坐标爬山搜索上下文、服务端并发、批处理 token 和内存利用率；支持吞吐、TTFT、TPOT 约束，并可按推荐参数自动启动服务。

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/vLLM-Serving-6C5CE7" alt="vLLM">
  <img src="https://img.shields.io/badge/Ascend-NPU-FF6B35" alt="Ascend">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Experimental-orange" alt="Status">
</p>

[快速开始](#-快速开始) · [部署前必改项](DEPLOYMENT.md#1-部署前必须修改的参数) · [全部参数说明](DEPLOYMENT.md#2-全部配置项解释) · [自适应搜索](#-自适应搜索) · [配置示例](#-配置示例) · [结果输出](#-结果与自动启动)

</div>

---

## 🎯 项目简介

Serving 参数相互耦合：上下文长度影响 KV Cache 占用，并发与批处理 token 上限影响吞吐和延迟。固定网格搜索容易产生大量无效实验。

本项目提供：
- **从低到高探索：** 各参数从候选列表最低值开始。
- **约束筛选：** 可设置最低输出吞吐、最大 TTFT、最大 TPOT。
- **梯度式离散调整：** 使用坐标爬山，每轮只尝试当前点相邻候选，指标改善且满足约束时才移动。
- **自动启动最终服务：** 搜索完成后按推荐参数重新启动服务并等待健康检查。

> 这是离散启发式搜索，不是对连续参数求导的数学梯度，也不保证全局最优。推荐结果只对给定候选空间、模型、硬件和压测负载有效。

## 🚀 快速开始

```bash
git clone https://github.com/FeynmanNddbb/ascend-serving-tuner.git
cd ascend-serving-tuner
cp config.example.json config.json
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 tuner.py --config config.json --mode adaptive --dry-run
python3 tuner.py --config config.json --mode adaptive

# 只搜索，不启动最终服务
python3 tuner.py --config config.json --mode adaptive --no-launch-best
```

## 🛠️ 部署参数文档\n\n首次部署建议先阅读 [部署与参数完整说明](DEPLOYMENT.md)：其中列出必须修改的模型路径、设备编号、TP、端口与 CANN 初始化步骤，并逐项解释所有 JSON 参数。\n\n## 🧠 自适应搜索逻辑

1. 将上下文长度、`max_num_seqs`、`max_num_batched_tokens`、`gpu_memory_utilization` 按候选值升序排列。
2. 从所有维度的最低值组成起点。
3. 测试当前点相邻的参数候选；服务启动或 benchmark 失败、约束不满足的候选不会成为推荐项。
4. 对满足约束的候选按目标指标评分，找到更优邻居后移动，直到没有改进、达到轮数或试验预算。
5. 保存推荐配置，并默认按该配置启动最终服务。

当前实现采用**离散坐标爬山**，避免一次性穷举笛卡尔积；可能陷入局部最优。重要部署建议在推荐点附近做网格复测。

## ⚙️ 配置示例

```json
{
  "mode": "adaptive",
  "model": "/workspace/work/data/models/Qwen3.8-27B-w8a8",
  "served_model_name": "qwen3.8",
  "host": "127.0.0.1",
  "port": 8000,
  "visible_devices": "0,1",
  "launch_best": true,
  "auto_tune": {
    "objective": "throughput",
    "tensor_parallel_size": 2,
    "enable_chunked_prefill": true,
    "search_space": {
      "context_lengths": [4096, 8192, 16384, 32768, 65536, 131072, 262144],
      "max_num_seqs": [1, 2, 4, 8, 16, 32],
      "max_num_batched_tokens": [1024, 2048, 4096, 8192, 16384],
      "gpu_memory_utilization": [0.80, 0.85, 0.90, 0.93]
    },
    "benchmark_concurrency": [1, 2, 4, 8],
    "max_trials": 40,
    "max_rounds": 12,
    "limits": {
      "min_output_throughput": 0,
      "max_mean_ttft_ms": 10000,
      "max_mean_tpot_ms": 1000
    }
  },
  "benchmark": {
    "output_len": 128,
    "num_prompts": 16,
    "backend": "openai-chat",
    "dataset_name": "random",
    "extra_args": []
  }
}
```

### 约束与默认值

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `min_output_throughput` | 0 tokens/s | 不设吞吐下限 |
| `max_mean_ttft_ms` | 10000 ms | 宽松首 token 延迟上限 |
| `max_mean_tpot_ms` | 1000 ms | 宽松单 token 延迟上限 |

默认值是起步保护值，不代表业务 SLA。可按业务收紧，例如 TTFT 2000ms、TPOT 80ms。若约束启用但指标未能从当前版本结果 JSON 解析，该候选不会被判定为满足约束。

目标支持 `throughput`（默认最大化输出 token 吞吐）、`request_throughput` 和 `latency`（最小化 TPOT）。

## 📊 结果与自动启动

`runs/<timestamp>/` 包含 `summary.csv`、`recommendation.json`、每次试验日志，以及最终服务的 `final_server.json` 和 `final_server.log`。默认 `launch_best: true`；不想启动时设置 false 或传入 `--no-launch-best`。最终服务会留在后台运行。

## ⚠️ 注意事项

- 运行前在当前 shell 初始化 CANN；`cann_env` 路径字段不代表脚本自动 source。
- vLLM 与 benchmark 参数随版本变化，请先核对本机 help。
- 请使用空闲端口和专用实验环境；搜索会反复启停测试服务。
- 搜索是离散局部优化，不保证全局最优；建议对推荐点做多轮复测。
- 当前未实现硬件内存自动探测、OOM 专项分类回退及多目标 Pareto 最优。

## 🗺️ Roadmap

- [x] 网格搜索
- [x] 离散坐标自适应搜索与 SLO 限制
- [x] 推荐参数自动启动
- [ ] OOM 分类、自动回退与设备内存探测
- [ ] TTFT/TPOT 多目标 Pareto 优化
- [ ] CUDA backend adapter、Ascend Profiler、HTML 报告

## 📄 License

MIT

---

## 🛠️ 部署与完整参数说明

首次部署请先看本节。**必须按实际环境修改的项目已明确标注**；其他参数可先沿用示例，再根据业务负载调整。

## 1. 部署前必须修改的参数

| 配置键 | 必须修改？ | 怎么填 |
|---|---|---|
| `model` | **必须** | 本机模型目录，确认目录存在且当前 vLLM-Ascend 能加载该模型 |
| `visible_devices` | **必须核对** | 参与推理的设备编号，如单卡 `"0"`、双卡 `"0,1"`；需与 TP 对应 |
| `auto_tune.tensor_parallel_size` | **必须核对** | 张量并行卡数；通常应与可见设备数/模型部署方式匹配 |
| `host` | 建议修改/核对 | 本机测试可用 `127.0.0.1`；需要局域网访问时按网络与安全策略设置，不建议无认证暴露公网 |
| `port` | **必须确认空闲** | 默认 8000；搜索会反复启动服务，不能占用已有服务端口 |
| CANN 环境 | **必须执行** | 在运行脚本的 shell 中 source 与已安装驱动/CANN 匹配的 set_env.sh |
| `served_model_name` | 必须保持一致 | API 服务名；脚本会把它传给 serve 与 benchmark，调用端也用此名称 |

示例：

```bash
# 根据实际安装路径修改
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 确认模型目录
ls /workspace/work/data/models/Qwen3.8-27B-w8a8

# 确认端口没有被其他服务占用
ss -ltnp | grep ':8000'
```

`cann_env`（若旧配置中仍存在）只是文档提示字段，当前脚本不会自动 source；请在启动脚本前手动 source。

## 2. 全部配置项解释

### 顶层配置

| 键 | 含义 | 示例/说明 |
|---|---|---|
| `mode` | 默认模式 | `adaptive` 自适应；`grid` 固定网格 |
| `model` | 模型路径 | 本地权重目录或当前环境支持的模型标识 |
| `served_model_name` | OpenAI API 暴露的模型名 | 例如 `qwen3.8` |
| `host` | 服务监听地址 | `127.0.0.1` 仅本机访问；远程访问需谨慎设置 |
| `port` | API 端口 | 必须空闲 |
| `visible_devices` | 传给 `ASCEND_RT_VISIBLE_DEVICES` 的设备列表 | 例如 `"0,1"` |
| `startup_timeout_sec` | 每次启动等待健康检查的最长秒数 | 大模型首次加载可设置更长 |
| `shutdown_timeout_sec` | 结束测试服务前等待 SIGINT 的秒数 | 超时后脚本尝试 terminate/kill |
| `launch_best` | 搜索结束后是否启动推荐服务 | `true` 会留下后台服务；`false` 只生成结果 |
| `env` | 注入子进程的环境变量 | 仅填写当前环境需要的变量 |

### auto_tune

| 键 | 含义 | 如何设置 |
|---|---|---|
| `objective` | 优化目标 | `throughput` 最大化输出 token/s；`request_throughput` 最大化 req/s；`latency` 最小化平均 TPOT |
| `tensor_parallel_size` | TP 张量并行度 | 固定值，不参与搜索；必须适配设备数和模型 |
| `enable_chunked_prefill` | 是否传入 Chunked Prefill 开关 | 当前代码按布尔值固定，不参与搜索；以本机版本支持情况为准 |
| `search_space.context_lengths` | 输入长度候选（token） | 从小到大排列；脚本将 output_len 加到 `max_model_len` |
| `search_space.max_num_seqs` | 服务端每轮可处理的序列上限候选 | 不是客户端压测并发 |
| `search_space.max_num_batched_tokens` | 单个调度迭代可处理 token 上限候选 | 太低可能限制 Prefill 吞吐，太高可能增加内存/调度压力 |
| `search_space.gpu_memory_utilization` | vLLM 可用设备内存比例候选 | 参数名沿用 vLLM；Ascend 插件是否支持需按版本核对 |
| `benchmark_concurrency` | 客户端压测并发候选 | 模拟同时发起的请求数，不等同服务端 max_num_seqs |
| `max_trials` | 最多评估多少个不同参数点 | 越大搜索更充分但耗时越久；每点会跑所有 benchmark_concurrency |
| `max_rounds` | 坐标搜索最多迭代轮数 | 用于限制搜索时间 |
| `limits.min_output_throughput` | 输出吞吐最低门槛，tokens/s | 默认 0 表示不设实际下限，但该指标仍需能从结果 JSON 解析 |
| `limits.max_mean_ttft_ms` | 平均首 token 延迟上限，毫秒 | 默认 10000；按业务 SLO 收紧 |
| `limits.max_mean_tpot_ms` | 平均每输出 token 时间上限，毫秒 | 默认 1000；按业务 SLO 收紧 |

默认 SLO 是宽松的启动基线，不代表生产服务承诺。若设置某项限制但当前 vLLM bench 输出无法解析对应指标，候选会被拒绝。建议先跑一次基准确认本版本 JSON 字段。

### benchmark

| 键 | 含义 | 说明 |
|---|---|---|
| `output_len` | 每个请求期望生成的 token 数 | 同时用于计算 `max_model_len = input_len + output_len` |
| `num_prompts` | 每次 benchmark 请求样本数 | 太少时指标波动大；增加会延长每次试验 |
| `backend` | 压测客户端后端 | 示例 `openai-chat` |
| `dataset_name` | 数据集类型 | 示例 `random`；随机长度用于参数比较，不等于真实业务分布 |
| `extra_args` | 额外传给 `vllm bench serve` 的 CLI 参数 | 必须是当前安装版本支持的参数列表 |

### env

| 变量 | 用途 | 说明 |
|---|---|---|
| `HCCL_BUFFSIZE` | HCCL 通信相关配置 | 保留当前环境已验证值；不确定时不要随意改 |
| `PYTORCH_NPU_ALLOC_CONF` | PyTorch NPU allocator 配置 | 示例 `expandable_segments:True`；需确认当前 torch_npu 版本支持 |

## 3. 首次部署推荐流程

1. 复制 `config.example.json` 为 `config.json`。
2. 修改模型路径、设备编号、TP、端口；确认 CANN 环境可用。
3. 初次只保留少量候选，例如上下文 `[4096,8192,16384]`、服务端序列 `[1,2,4]`、batch tokens `[1024,2048,4096]`，并把 `max_trials` 设为 8。
4. 先运行 `python3 tuner.py --config config.json --mode adaptive --dry-run`。
5. 确认端口空闲后执行小规模测试。检查 `runs/<timestamp>/trial_*/benchmark.json` 与日志，确认 TTFT/TPOT/吞吐字段有被正确解析。
6. 再逐步扩大搜索空间，最后按业务 SLO 设定限制。

## 4. 启动与停止最终服务

搜索成功后，若 `launch_best=true`，程序会在搜索结束后启动推荐配置并等待 `/v1/models` 健康检查，最终进程留在后台。PID 和命令写入 `final_server.json`，日志写入 `final_server.log`。

不要直接假设 PID 永远有效。停止前先检查 PID 与命令确实属于本次 tuner 启动的实例，再发送 SIGINT/TERM；不要对未知 PID 执行 kill。

## 5. 参数含义的上游参考

- [vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM bench serve CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)

vLLM-Ascend 的参数支持可能与上游 vLLM 不完全一致，以当前安装环境的 `vllm serve --help` 和 `vllm bench serve --help` 为准。
