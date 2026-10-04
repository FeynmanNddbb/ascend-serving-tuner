# Ascend Serving Tuner

面向 vLLM-Ascend 的服务端参数联合搜索工具：对每组服务端配置重启服务、等待 API 就绪、执行 `vllm bench serve`、保存原始日志与结果，并生成可复核的汇总表。

> 当前版本是可运行的实验脚手架，不宣称自动发现所有 vLLM-Ascend 参数或保证所有版本 CLI 完全兼容。先用 `--dry-run` 检查命令，再小规模验证。

## 能做什么

- 联合搜索 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`
- 可选启用 Chunked Prefill
- 每个试验独立启动/停止服务，避免把客户端并发误当成服务端 `max-num-seqs`
- 记录启动参数、benchmark stdout/stderr、原始 JSON（若当前 vLLM 版本支持）和 CSV 汇总
- 对启动失败、健康检查超时、benchmark 失败记录状态并继续

## 环境

在已安装并能运行 vLLM-Ascend 的环境中使用。模型权重需已在本地。

## 快速开始

```bash
git clone https://github.com/FeynmanNddbb/ascend-serving-tuner.git
cd ascend-serving-tuner
cp config.example.json config.json
# 修改 model、served_model_name、visible_devices、候选参数
python3 tuner.py --config config.json --dry-run
python3 tuner.py --config config.json
```

首次建议缩小候选范围：1-2个 `max_num_seqs`、1-2个 `max_num_batched_tokens`、短输入、少量请求。完整搜索可能耗时很久。

## 重要说明

1. 该工具会停止自己启动的服务进程；不要把 `server_command` 改成会影响其他服务的命令。
2. 默认只管理由本工具启动的进程，不会杀掉已有模型服务。
3. 搜索每组服务端参数需要重启模型；若你要复用当前已启动服务，只能搜索客户端负载参数，不能真实改变服务端启动参数。
4. `vllm bench serve` 的 CLI 与结果 schema 随版本变化。遇到参数不支持时，按本机 `vllm bench serve --help` 修改配置中的 `bench_extra_args` 或源码。
5. CSV只提取已识别字段；未知schema仍保留原始日志/JSON，不会虚构指标。
6. 当前重点是单实例、单模型、固定硬件的参数搜索，不包含多机拓扑、模型量化生成或在线流量控制。

## 输出

运行后生成 `runs/<时间戳>/`：
- `summary.csv`
- 每组 `trial_xxx/server.log`
- 每组 `trial_xxx/benchmark.log`
- 每组 `trial_xxx/benchmark.json`（如果命令生成）
- `trial_xxx/config.json`
