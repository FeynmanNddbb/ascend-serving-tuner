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
