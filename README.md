<div align="center">

# Ascend Serving Tuner

**基于 vLLM Bench 的 vLLM / vLLM-Ascend 推理服务参数调优工具**

通过自动启动服务、调用 `vllm bench serve` 发起压测、解析性能指标并搜索服务端配置，帮助评估长上下文与并发场景下的 Serving 性能。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![vLLM](https://img.shields.io/badge/Benchmark-vllm%20bench-6C5CE7)
![Ascend](https://img.shields.io/badge/Ascend-NPU-FF6B35)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

</div>

---

## 项目简介

大模型推理服务的上下文长度、调度并发、Batch Tokens 和显存利用率相互影响。手动试参容易遗漏配置，完整笛卡尔积搜索则可能带来大量启动与压测开销。

本项目围绕 **vLLM 自带的 `vllm bench serve`** 构建实验闭环：

1. 从配置文件读取模型、设备、服务端候选参数与压测条件。
2. 启动一个候选 vLLM / vLLM-Ascend 服务，并等待 `/v1/models` 健康检查。
3. 调用 `vllm bench serve`，通过 OpenAI-compatible Chat Completions API 发起请求。
4. 将 benchmark 结果保存为 JSON，解析吞吐、TTFT、TPOT 等指标。
5. 按 SLO 过滤候选，并依据目标指标进行离散坐标搜索。
6. 对候选结果重复验证；通过后写出推荐配置，可选择启动推荐服务。

当前 adaptive 模式中，客户端压测并发自动等于当前候选的服务端 `max_num_seqs`。**用户只需在参数列表中手动填写希望测试的候选值，不需要单独配置客户端并发列表。**

> 本项目是实验性参数调优工具，不保证全局最优。结果仅适用于本次模型、硬件、软件版本、候选参数和压测负载。当前尚未在所有 vLLM-Ascend 版本与设备组合上完成兼容性验证。

## 工作流程

```text
config.json
    |
    v
选择服务端候选参数
    |
    v
启动 vllm serve
    |
    v
等待 /v1/models 就绪
    |
    v
vllm bench serve
  --random-input-len
  --random-output-len
  --num-prompts
  --max-concurrency = max_num_seqs
    |
    v
benchmark.json + benchmark.log
    |
    v
解析指标 / 检查 SLO / 记录 summary.csv
    |
    v
离散坐标搜索 + 最终重复验证
    |
    v
recommendation.json
    |
    +---- launch_best=true ---> 启动推荐配置
```

## 主要能力

- **vLLM Bench 压测**：通过 `vllm bench serve` 对已启动的 OpenAI-compatible 服务发起请求，不是自定义 HTTP 压测器。
- **服务端参数搜索**：上下文长度、`max_num_seqs`、`max_num_batched_tokens`、`gpu_memory_utilization`。
- **并发联动**：每个候选的客户端 `--max-concurrency` 取该候选的 `max_num_seqs`。
- **目标与 SLO**：输出吞吐、请求吞吐或 TPOT 延迟作为目标；支持最低输出吞吐、最大平均 TTFT、最大平均 TPOT 约束。
- **可追溯实验**：保存服务日志、benchmark 日志、原始结果 JSON、CSV 汇总和推荐配置。
- **最终复测**：对排名靠前的候选重复 benchmark；每次均通过约束才推荐。
- **可选启动**：搜索成功后可自动启动推荐服务。

## 环境要求

- Linux 环境，已安装并可在当前 shell 中调用 `vllm`。
- Python 3.10+。
- vLLM 或与当前设备匹配的 vLLM-Ascend、PyTorch、CANN 等运行环境。
- 模型权重已在本机可访问的目录中。
- 压测端口空闲，设备没有其他进程占用或干扰实验。

对于 Ascend，运行前按实际安装路径初始化 CANN，例如：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

先确认当前版本命令参数：

```bash
vllm serve --help
vllm bench serve --help
```

vLLM 与 vLLM-Ascend 的 CLI 参数会随版本变化；若本机不支持脚本使用的参数，应先适配版本，不要直接把其他版本的 benchmark 命令当作兼容保证。

## 快速开始

```bash
git clone https://github.com/FeynmanNddbb/ascend-serving-tuner.git
cd ascend-serving-tuner
cp config.example.json config.json
```

编辑 `config.json`，至少确认模型路径、设备、TP、端口和候选参数。先执行 dry-run 查看模式与配置：

```bash
python3 tuner.py --config config.json --mode adaptive --dry-run
```

开始调优：

```bash
python3 tuner.py --config config.json --mode adaptive
```

只搜索、不启动最终推荐服务：

```bash
python3 tuner.py --config config.json --mode adaptive --no-launch-best
```

请在专用实验环境与空闲端口运行。脚本会反复启动和停止自己创建的服务实例；不要将端口指向正在承载业务流量的服务。

## 配置说明

### 首次需要修改的参数

| 参数 | 用途 | 示例 |
|---|---|---|
| `model` | 本机模型目录 | `/workspace/work/data/models/Qwen3.8-27B-w8a8` |
| `served_model_name` | API 暴露的模型名，服务端与 benchmark 保持一致 | `qwen3.8` |
| `visible_devices` | 本次实验可见设备 | 双卡 `"0,1"` |
| `auto_tune.tensor_parallel_size` | 张量并行度 | 双卡 TP=2 |
| `host` / `port` | 服务监听与压测地址 | `127.0.0.1:8000` |
| `auto_tune.search_space` | 手动填写待测试候选列表 | 见下方 |

### 示例配置

以下是双设备 Ascend 示例。**请按目标机器能力调整候选列表；不要将示例中的 128K/256K 或并发档位视为硬件保证。**

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
      "context_lengths": [8192, 16384, 32768, 65536],
      "max_num_seqs": [1, 2, 4, 8, 16],
      "max_num_batched_tokens": [2048, 4096, 8192, 16384],
      "gpu_memory_utilization": [0.85, 0.90, 0.93]
    },
    "max_trials": 40,
    "max_total_benchmarks": 400,
    "max_rounds": 12,
    "final_validation_repeats": 3,
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

### 服务端参数候选

| 参数 | 含义 | 配置建议 |
|---|---|---|
| `context_lengths` | 随机输入长度候选，单位 token | 先从 4K/8K/16K 开始，确认稳定后再增加长上下文档位。服务端 `max_model_len = context_length + output_len`。 |
| `max_num_seqs` | 服务端调度序列数上限 | 手动填写希望测的并发档位，如 `[1, 2, 4, 8]`。当前候选的客户端压测并发会自动设为同一数值。 |
| `max_num_batched_tokens` | 单次调度迭代的 token 上限 | 从较低值开始，逐步观察吞吐、延迟与显存变化。 |
| `gpu_memory_utilization` | vLLM 设备内存利用率 | 从保守值开始，逐步提高；确认当前后端版本支持该参数。 |

### vLLM Bench 压测参数

本项目当前通过以下命令参数组织每次压测（具体命令由脚本按配置生成）：

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/chat/completions \
  --model /path/to/model \
  --served-model-name model \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 128 \
  --num-prompts 16 \
  --max-concurrency 8 \
  --save-result \
  --result-filename runs/trial/benchmark.json
```

参数映射：

| 配置项 | 对应 vLLM Bench 参数 | 说明 |
|---|---|---|
| `benchmark.backend` | `--backend` | 默认 `openai-chat` |
| `benchmark.dataset_name` | `--dataset-name` | 示例为 `random` |
| 当前候选上下文长度 | `--random-input-len` | 每个请求的随机输入长度 |
| `benchmark.output_len` | `--random-output-len` | 每个请求的生成长度 |
| `benchmark.num_prompts` | `--num-prompts` | 本轮请求总数 |
| 当前候选 `max_num_seqs` | `--max-concurrency` | 客户端并发自动跟随服务端候选 |
| 结果文件 | `--save-result --result-filename` | 保存 benchmark JSON |
| `benchmark.extra_args` | 追加 CLI 参数 | 必须是当前安装版本支持的参数 |

**并发测试逻辑示例：** 当候选 `max_num_seqs=8` 时，服务端使用 `--max-num-seqs 8`，vLLM Bench 同时使用 `--max-concurrency 8`；候选为 16 时，两者都变为 16。用户不需要维护第二份客户端并发列表。

`num_prompts` 应足以覆盖目标并发负载。若请求数少于并发档位，客户端无法持续提供足够请求，压测结果可能低估服务端吞吐。建议根据实验时长和目标场景手动设置请求数。

### 目标与 SLO

| 参数 | 作用 |
|---|---|
| `objective: "throughput"` | 优先最大化输出 token 吞吐 |
| `objective: "request_throughput"` | 优先最大化请求处理速率 |
| `objective: "latency"` | 优先降低平均 TPOT；吞吐用于次级比较 |
| `min_output_throughput` | 输出吞吐最低门槛，tokens/s |
| `max_mean_ttft_ms` | 平均首 Token 延迟上限，ms |
| `max_mean_tpot_ms` | 平均每输出 Token 时间上限，ms |

约束字段未能从当前版本 benchmark JSON 解析时，该候选无法通过对应 SLO。建议先用少量请求运行一次，检查结果 JSON 字段，再设置严格门槛。

## 搜索策略与边界

- adaptive 模式采用从各维度最低候选起步的**离散坐标爬山**，逐维尝试相邻候选；不是完整笛卡尔积穷举，也不保证全局最优。
- `max_trials` 限制不同服务端参数点数量。
- `max_total_benchmarks` 限制搜索和最终验证的 benchmark 总次数。
- `max_rounds` 限制坐标搜索轮数。
- `final_validation_repeats` 控制最终候选重复压测次数；候选每次都通过 SLO 才会被推荐。
- 当前客户端并发不是独立搜索维度，而是由当前候选 `max_num_seqs` 派生。
- 当前未实现自动设备内存探测、OOM 专项分类回退或 Pareto 多目标优化。
- 随机数据适合相对参数比较，不等价于真实业务的输入分布、输出分布或到达过程。

## 结果目录

每次运行在 `runs/<timestamp>/` 下生成结果：

| 文件 | 内容 |
|---|---|
| `summary.csv` | 所有已执行试验的参数、状态与可解析指标 |
| `recommendation.json` | 推荐服务端配置、目标、SLO、选中指标和搜索信息 |
| `trial_xxxx/server.log` | 候选服务启动日志 |
| `trial_xxxx/benchmark.log` | vLLM Bench 标准输出与错误信息 |
| `trial_xxxx/benchmark.json` | vLLM Bench 原始结果 |
| `trial_xxxx/config.json` | 本次试验参数及输入长度、客户端并发 |
| `trial_xxxx/server_command.txt` | 本次服务启动命令 |
| `final_server.json` / `final_server.log` | 推荐服务的 PID、启动命令与日志 |

默认 `launch_best: true` 会在搜索成功后启动推荐配置并保持运行。停止前请核对 PID 与命令确实属于本次 tuner 实例。

## 常见问题

**为什么客户端并发不单独配置？**  
当前设计将客户端 `--max-concurrency` 与候选服务端 `max_num_seqs` 对齐，方便比较每个服务端并发档位的表现。它测的是该并发档位下的表现，不等同于独立探索客户端过载/排队曲线。

**为什么某些候选失败？**  
可能是模型加载失败、设备内存不足、CLI 参数版本不兼容、服务未通过健康检查、benchmark 非零退出或请求失败。查看对应 `server.log` 与 `benchmark.log`。

**为什么搜索结果不一定是全局最优？**  
adaptive 使用离散坐标爬山并受 `max_trials`、`max_rounds` 限制。想覆盖更多区域，需手动调整候选列表或增加搜索预算。

## 许可与参考

- [vLLM Serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM Bench Serve CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)

License: MIT
