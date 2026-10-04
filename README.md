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
5. 按 SLO 过滤候选，并依据目标指标进行离散网格穷举搜索（可通过预算显式限制为部分搜索）。
6. 对候选结果重复验证；通过后写出推荐配置，可选择启动推荐服务。

当前 adaptive 模式中，客户端压测并发自动等于当前候选的服务端 `max_num_seqs`。**用户需要手动配置四组服务端候选参数；客户端并发不需要单独配置。**

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
离散网格搜索 + Top-K 最终重复验证
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

编辑 `config.json`，至少确认模型路径、设备、TP、端口，以及下文列出的四组候选参数。先执行 dry-run 查看模式与配置：

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

### 首次必须核对/填写的参数

以下项目都需要根据实际机器和目标实验确认。尤其是四组 `search_space` 列表：它们决定 tuner 会探索哪些参数档位。不要直接照搬示例值作为硬件能力结论。

| 参数 | 用途 | 示例 |
|---|---|---|
| `model` | 本机模型目录 | `/workspace/work/data/models/Qwen3.8-27B-w8a8` |
| `served_model_name` | API 暴露的模型名，服务端与 benchmark 保持一致 | `qwen3.8` |
| `visible_devices` | 本次实验可见设备 | 双卡 `"0,1"` |
| `auto_tune.tensor_parallel_size` | 张量并行度，需与设备数及模型适配 | 双卡 TP=2 |
| `host` / `port` | 服务监听与压测地址，端口须空闲 | `127.0.0.1:8000` |
| `auto_tune.search_space.context_lengths` | 必须手动确定输入长度候选 | `[8192, 16384, 32768, 65536]` |
| `auto_tune.search_space.max_num_seqs` | 必须手动确定服务端并发候选 | `[1, 2, 4, 8, 16]` |
| `auto_tune.search_space.max_num_batched_tokens` | 必须手动确定 Batch Tokens 候选 | `[2048, 4096, 8192, 16384]` |
| `auto_tune.search_space.gpu_memory_utilization` | 必须手动确定显存利用率候选 | `[0.85, 0.90, 0.93]` |

### 四组候选参数配置指南

#### 1. `context_lengths`：输入上下文长度

```json
"context_lengths": [8192, 16384, 32768, 65536]
```

- **含义：** vLLM Bench 每个请求的随机输入 token 长度候选。
- **如何设置：** 按目标业务的上下文需求设置档位，例如短文本问答可从 4K/8K/16K 起步；长文档、RAG 场景可逐步加入 32K、64K、128K。
- **注意：** 服务端 `max_model_len` 按输入长度加 `benchmark.output_len` 设置，因此实际模型最大长度需要覆盖输入与输出之和。
- **建议：** 首次不要直接把最大上下文设到模型标称上限。先验证低档位能稳定启动、完成请求，再增加更长档位；观察 TTFT、显存与失败情况。

#### 2. `max_num_seqs`：服务端并发序列上限

```json
"max_num_seqs": [1, 2, 4, 8, 16]
```

- **含义：** vLLM 服务端调度器允许同时处理的序列数候选。
- **如何设置：** 根据目标负载填写希望测试的并发档位，例如单请求基线用 `[1]`，逐步扩展并发可用 `[1, 2, 4, 8]`；设备资源充足且需要更高并发时再加入 16、32 等档位。
- **并发联动：** 每个候选的客户端 `vllm bench serve --max-concurrency` 自动取同一个 `max_num_seqs` 值。例如服务端候选为 8，客户端压测并发也为 8。
- **注意：** `max_num_seqs` 是服务端上限，不代表一定能在任意上下文长度下承载该并发。长上下文会显著增加 KV Cache 需求；提高并发时应同时观察 OOM、吞吐与延迟。
- **请求数：** `benchmark.num_prompts` 应大于并发档位并留有足够请求，避免请求总量太少导致压测不充分。

#### 3. `max_num_batched_tokens`：单次调度 token 预算

```json
"max_num_batched_tokens": [2048, 4096, 8192, 16384]
```

- **含义：** 服务端调度迭代中可处理的 token 数量上限，影响 Prefill/Decode 的调度与吞吐权衡。
- **如何设置：** 可从 `[1024, 2048, 4096]` 这类较保守档位开始，再按设备能力加入 8192、16384 等候选。
- **调大可能的收益：** 有机会提高 Prefill 阶段处理效率与整体吞吐。
- **代价与风险：** 可能增加瞬时显存占用、Prefill 对 Decode 的干扰或延迟；具体效果依模型、上下文、硬件和后端实现而变。
- **建议：** 长上下文优先小步增加，结合 TTFT、TPOT、吞吐和启动/请求失败情况判断，不要只看吞吐单项。

#### 4. `gpu_memory_utilization`：设备内存利用率目标

```json
"gpu_memory_utilization": [0.85, 0.90, 0.93]
```

- **含义：** vLLM 用于规划模型执行与 KV Cache 等资源的设备内存利用率目标。
- **如何设置：** 从 0.80–0.85 的保守值开始，稳定后再试 0.90、0.93 等更高档位。
- **调高可能的收益：** 可能为 KV Cache 留出更多空间，支持更长上下文或更高并发。
- **代价与风险：** 运行时可用余量变少，其他进程、内存碎片或额外工作区可能导致启动失败或 OOM。
- **注意：** 参数名称与实际支持情况取决于 vLLM / vLLM-Ascend 版本和后端；先确认当前版本支持，不要默认 0.93 一定安全，也不建议直接设为 1.0。

### 候选列表如何组合

四组列表共同定义搜索空间。列表越长，可能探索的参数区域越广，但试验耗时也会增加。当前 adaptive 默认穷举候选列表的笛卡尔积；`max_trials` 可显式限制为部分搜索，`max_total_benchmarks` 控制搜索与验证总预算。

建议按以下顺序逐步扩展：

1. **先建立可运行基线：** 短上下文、低并发、较低 Batch Tokens、保守内存利用率。
2. **确定上下文范围：** 逐步增加 `context_lengths`，找到目标场景下可稳定完成请求的范围。
3. **探索并发：** 增加 `max_num_seqs` 候选；客户端并发会自动联动。
4. **调节批处理与内存：** 再增加 `max_num_batched_tokens` 和 `gpu_memory_utilization` 档位，比较吞吐与延迟。
5. **设置 SLO：** 获得基线数据后，再填写吞吐、TTFT、TPOT 约束，避免门槛过严导致所有候选被过滤。

### 示例配置

以下是双设备 Ascend 示例。按上一节指南修改四组候选值：

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
    "max_trials": 0,
    "max_total_benchmarks": 1200,
    "final_validation_top_k": 5,
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

- adaptive 模式默认采用**离散网格穷举**，遍历配置候选列表的笛卡尔积；在候选集合内比较实测可行配置，不因局部邻居变差而提前停止。
- `max_trials: 0` 表示不限制搜索点数、尝试完整候选集合；正整数表示显式截断搜索，结果会标记 `search_complete: false`。
- `max_total_benchmarks` 限制搜索与最终验证的 benchmark 总次数；预算不足时会在开跑前报错。
- `final_validation_top_k` 控制复测排名靠前的候选数；`final_validation_repeats` 控制每个候选的重复次数。
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

**搜索结果是否是全局最优？**  
完整搜索时，它是当前离散候选集合、当前 workload、SLO 和硬件软件环境下的最佳实测候选；不代表连续参数空间或其他负载下的数学全局最优。若 `max_trials` 显式限制搜索，`search_complete` 会为 `false`。

## 许可与参考

- [vLLM Serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM Bench Serve CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)

License: MIT


## 使用调优结果启动推理服务

调优结束后，打开本次运行目录中的 `recommendation.json`，将 `recommended_server` 的结果填入下面命令。尖括号中的内容均为**占位符**，必须替换为项目实际输出值；不要把尖括号原样复制到 shell。

以下示例启用已确认的通用 vLLM Serving 开关：**Chunked Prefill** 与 **Prefix Caching**，并使用调优得到的 TP、上下文长度、并发、Batch Tokens 和内存利用率。注意：Prefix Caching 尚未纳入 tuner 搜索，推荐指标是在当前 tuner 压测配置下得到的；若实际业务有重复前缀，应使用重复前缀 workload 重新 benchmark。Partial Prefill 与 Long Partial Prefill 暂不放入示例，待确认 vLLM-Ascend 0.23.0 对应 CLI 参数及 A3/Qwen3.8 组合行为后再加。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /path/to/<MODEL_DIR> \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --served-model-name <SERVED_MODEL_NAME> \\
  --tensor-parallel-size <RECOMMENDED_TENSOR_PARALLEL_SIZE> \\
  --max-model-len <RECOMMENDED_MAX_MODEL_LEN> \\
  --max-num-seqs <RECOMMENDED_MAX_NUM_SEQS> \\
  --max-num-batched-tokens <RECOMMENDED_MAX_NUM_BATCHED_TOKENS> \\
  --gpu-memory-utilization <RECOMMENDED_GPU_MEMORY_UTILIZATION> \\
  --enable-prefix-caching \\
  --enable-chunked-prefill
```

参数映射示例（以下数值仅演示映射方式，不是推荐性能值）：

| 启动参数 | 从哪里取得 |
|---|---|
| `<MODEL_DIR>` | `config.json` 的 `model` |
| `<SERVED_MODEL_NAME>` | `config.json` 的 `served_model_name` |
| `<RECOMMENDED_TENSOR_PARALLEL_SIZE>` | `recommendation.json` → `recommended_server.tensor_parallel_size` |
| `<RECOMMENDED_MAX_MODEL_LEN>` | `recommendation.json` → `recommended_server.max_model_len` |
| `<RECOMMENDED_MAX_NUM_SEQS>` | `recommendation.json` → `recommended_server.max_num_seqs` |
| `<RECOMMENDED_MAX_NUM_BATCHED_TOKENS>` | `recommendation.json` → `recommended_server.max_num_batched_tokens` |
| `<RECOMMENDED_GPU_MEMORY_UTILIZATION>` | `recommendation.json` → `recommended_server.gpu_memory_utilization` |

> 当前 tuner 搜索与压测命令中仅启用了 `--enable-chunked-prefill`；本节额外加入 `--enable-prefix-caching` 作为部署示例中的已知 vLLM 开关，但它并非 tuner 已验证的最优项。Prefix Caching 只有在请求共享相同前缀时才可能带来 Prefill 收益。Partial Prefill、Long Partial Prefill、KV Cache dtype、Attention Backend、CUDA Graph 和 Speculative Decoding 暂不加入示例，待确认当前 vLLM-Ascend 版本与模型组合支持后再纳入。
