<div align="center">

# Ascend Serving Tuner

**面向 vLLM / vLLM-Ascend 的自适应推理服务参数调优工具**

从较低配置开始逐步探索上下文长度、服务端并发、Batch Tokens 与显存利用率；可设置吞吐、TTFT、TPOT 约束，并在搜索结束后按推荐参数启动服务。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![vLLM](https://img.shields.io/badge/vLLM-Serving-6C5CE7)
![Ascend](https://img.shields.io/badge/Ascend-NPU-FF6B35)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

</div>

---

## 1. 项目简介

大模型 Serving 参数彼此影响：上下文越长，KV Cache 占用越高；并发和批处理 token 上限会改变吞吐、排队与延迟。手动试参容易遗漏组合，完整网格搜索又可能耗时很长。

本项目提供：
- 从候选参数的低值起步，进行离散坐标搜索。
- 使用最低吞吐、最大 TTFT、最大 TPOT 作为可配置筛选条件。
- 以输出吞吐、请求吞吐或 TPOT 作为优化目标。
- 保存每次试验日志、CSV 汇总和推荐配置。
- 搜索完成后可自动启动推荐配置的服务。

> 搜索算法为离散坐标爬山，不是连续数学梯度法，也不保证全局最优。结果只对当前模型、设备、软件版本、候选空间和压测负载有效。

## 2. 快速部署

### 2.1 先修改这几个部署参数

**首次部署只需要优先核对以下项目；其余搜索和压测参数可先保留默认值。**

| 参数 | 必须做什么 | 示例 |
|---|---|---|
| `model` | 改为本机模型目录 | `/workspace/work/data/models/Qwen3.8-27B-w8a8` |
| `visible_devices` | 改为本次分配给服务的设备编号 | 双卡：`"0,1"`；单卡：`"0"` |
| `auto_tune.tensor_parallel_size` | 与设备数及模型并行方式匹配 | 双卡 TP：`2` |
| `port` | 确保端口空闲；搜索会反复启动/停止测试服务 | `8000` |
| `served_model_name` | 设定 API 使用的模型名；保持服务和 benchmark 一致 | `"qwen3.8"` |
| CANN 环境 | 按机器实际安装路径初始化环境 | 见下方命令 |

`host` 默认是 `127.0.0.1`，适合本机压测。只有确实需要其他机器访问时才修改，并遵循网络访问控制；不要无认证地暴露到公网。

### 2.2 安装与运行

```bash
git clone https://github.com/FeynmanNddbb/ascend-serving-tuner.git
cd ascend-serving-tuner
cp config.example.json config.json

# 按本机 CANN 安装位置修改路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 先检查配置与将要执行的模式
python3 tuner.py --config config.json --mode adaptive --dry-run

# 执行搜索；launch_best=true 时结束后自动启动推荐服务
python3 tuner.py --config config.json --mode adaptive
```

只搜索、不启动最终服务：

```bash
python3 tuner.py --config config.json --mode adaptive --no-launch-best
```

**注意：** 请在空闲端口和专用实验环境运行。程序只应管理它自己启动的测试实例；不要把端口指向已有业务服务。

## 3. 配置示例

下面示例适用于双设备 Ascend 环境。部署时先按上一节修改模型路径、设备、TP 和端口。

```json
{
  "mode": "adaptive",
  "model": "/workspace/work/data/models/Qwen3.8-27B-w8a8",
  "served_model_name": "qwen3.8",
  "host": "127.0.0.1",
  "port": 8000,
  "visible_devices": "0,1",
  "startup_timeout_sec": 900,
  "shutdown_timeout_sec": 30,
  "launch_best": true,
  "auto_tune": {
    "objective": "throughput",
    "tensor_parallel_size": 2,
    "enable_chunked_prefill": true,
    "search_space": {
      "context_lengths": [4096, 8192, 16384, 32768, 65536, 131072],
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
  },
  "env": {
    "HCCL_BUFFSIZE": "512",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"
  }
}
```

## 4. 核心调优参数

以下参数名与 `config.json` 中的实际 JSON 键保持一致，按配置层级书写。

| 配置键 | 作用 | 调整建议 |
|---|---|---|
| `auto_tune.search_space.context_lengths` | 输入上下文长度候选（token）；脚本将 `benchmark.output_len` 加到 `max_model_len` | 从小到大设置；先测 4K/8K/16K，再逐步扩大 |
| `auto_tune.search_space.max_num_seqs` | 服务端同时调度的序列数上限候选 | 从 1、2、4 逐步增加；它不是客户端压测并发数 |
| `auto_tune.search_space.max_num_batched_tokens` | 单次调度迭代可处理的 token 上限候选 | 小值可能限制 Prefill 吞吐，大值可能增加内存压力 |
| `auto_tune.search_space.gpu_memory_utilization` | vLLM 设备内存利用率候选 | 从保守值开始；确认当前 vLLM-Ascend 版本支持该参数 |
| `auto_tune.benchmark_concurrency` | 压测客户端并发请求数候选 | 用来模拟业务负载；不等于服务端 `max_num_seqs` |
| `auto_tune.objective` | 选择优化目标 | `throughput` 最大化输出 token/s；`request_throughput` 最大化 req/s；`latency` 最小化平均 TPOT |

### 性能限制（SLO）

| 配置键 | 默认值 | 含义 |
|---|---:|---|
| `auto_tune.limits.min_output_throughput` | `0` tokens/s | 不设有效吞吐下限，但吞吐指标仍需可解析 |
| `auto_tune.limits.max_mean_ttft_ms` | `10000` ms | 平均首 token 延迟上限，宽松起步值 |
| `auto_tune.limits.max_mean_tpot_ms` | `1000` ms | 平均每输出 token 时间上限，宽松起步值 |

默认值是避免过早筛掉候选的起点，不是生产 SLA。可按实际业务收紧，例如 TTFT 2000 ms、TPOT 80 ms。若启用的指标无法从当前版本 benchmark JSON 解析，该候选不会通过约束筛选。

## 5. 其他参数说明

以下参数通常无需首次部署时修改，只有在需要控制实验时间、服务行为或运行环境时再调整。

| 参数 | 作用 | 说明 |
|---|---|---|
| `mode` | 默认运行模式 | `adaptive` 自适应搜索；`grid` 固定网格搜索 |
| `startup_timeout_sec` | 等待服务健康检查的最长时间 | 大模型加载较慢时可增加 |
| `shutdown_timeout_sec` | 停止测试服务时等待 SIGINT 的时间 | 超时后脚本尝试 terminate/kill |
| `launch_best` | 搜索结束后是否启动推荐服务 | `true` 会让最终服务留在后台运行 |
| `auto_tune.tensor_parallel_size` | 张量并行度 | 固定值，不参与搜索；必须与设备和模型适配 |
| `auto_tune.enable_chunked_prefill` | 是否启用 Chunked Prefill | 固定开关，不参与搜索；需确认当前版本支持 |
| `auto_tune.max_trials` | 最多评估的不同参数点数 | 越大搜索更充分、耗时也越长；每个点会测试所有客户端并发候选 |
| `auto_tune.max_rounds` | 坐标搜索迭代轮数上限 | 防止搜索时间无限增长 |
| `benchmark.output_len` | 每个请求生成的 token 数 | 同时影响服务端所需最大模型长度 |
| `benchmark.num_prompts` | 每次 benchmark 请求数 | 数量太少指标波动较大；增加会延长实验 |
| `benchmark.backend` | 压测客户端后端 | 示例为 `openai-chat` |
| `benchmark.dataset_name` | 压测数据类型 | `random` 便于可重复对比，但不等于真实业务请求分布 |
| `benchmark.extra_args` | 额外传给 `vllm bench serve` 的参数 | 需符合当前安装版本 CLI |
| `env` | 子进程环境变量 | 示例中的 HCCL 与 NPU allocator 变量不确定时不要随意改 |

## 6. 搜索结果与自动启动

每次运行生成 `runs/<timestamp>/`：

- `summary.csv`：每次试验的参数、状态与可解析指标。
- `recommendation.json`：推荐服务参数、优化目标、约束和选中指标。
- `trial_xxxx/server.log`、`benchmark.log`、`benchmark.json`：单次试验记录。
- `final_server.json`、`final_server.log`：最终服务的 PID、启动命令和日志。

默认 `launch_best: true`。停止服务前，请先核对 PID 和命令确实属于本次 tuner 启动的实例，不要对未知 PID 执行 kill。

## 7. 搜索方法与注意事项

程序从候选参数的低值起步，通过离散坐标爬山测试相邻候选；只有约束通过且目标指标改善时才移动。该方法比完整笛卡尔积搜索节省试验数，但可能陷入局部最优。建议对最终推荐点进行多轮复测。

- 运行前在当前 shell 执行正确的 CANN `set_env.sh`。配置中的 `cann_env` 路径（若有）不会被脚本自动 source。
- vLLM 与 vLLM-Ascend 参数支持随版本变化；先检查本机 `vllm serve --help` 和 `vllm bench serve --help`。
- 当前尚未实现设备内存自动探测、OOM 专项分类回退和多目标 Pareto 优化。
- 随机数据的 benchmark 适合参数相对比较；生产部署应使用接近真实业务的输入/输出长度与请求分布。

## 8. 参考与许可

- [vLLM Serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM Bench Serve CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)

License: MIT
