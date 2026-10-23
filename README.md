# TrainLoopDotNet8

把 [JerrettDavis/TrainLoop](https://github.com/JerrettDavis/TrainLoop)（MIT）移植到 **.NET 8**，并加上 **Qwen2.5-0.5B** 的纯 CPU 推理和纯 CPU LoRA 训练编排。

上游是人工标注、评审员校准和 meta-model 规划，**不会**微调大语言模型。本仓库保留标注领域，另外接上 Qwen 的训练/推理。权重不进 git。不需要 NVIDIA GPU。

| 项 | 选择 |
| --- | --- |
| 上游快照 | `2d39a803ae627592f3a64fde2bb1dffc282b16c8`（2026-10-19） |
| 目标框架 | `net8.0`，SDK 8.0.x（`global.json` 从 8.0.100 向前滚动） |
| 推理 | 进程内 [LLamaSharp 0.19](https://www.nuget.org/packages/LLamaSharp/0.19.0) + `LLamaSharp.Backend.Cpu`（llama.cpp `958367bf`）。`GpuLayerCount` 固定为 0，不引用 CUDA 包。不用 0.20+，那些版本会带上 Microsoft.Extensions 9/10 |
| 训练 | C# 只编排外部进程。默认是本仓库的 `tools/cpu-trainer/train_qwen_lora.py`（transformers + peft，fp32，`device=cpu`）。可选 LLaMA-Factory YAML |

推理不用 ONNX Runtime：Qwen2.5 的官方 GGUF 可以直接被 llama.cpp 加载，不必再导出 ONNX、分词器和生成循环。LLaMA-Factory 能在 CPU 上做 Qwen2.5 LoRA（清空 `CUDA_VISIBLE_DEVICES`，并关掉 `bf16`/`fp16`），但它没有稳定的 `device=cpu` 开关，所以默认训练器是仓库里的脚本，脚本里显式 `model.to(cpu)`。

许可和署名见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。上游版权是 `Copyright (c) 2026 Jerrett Davis`，MIT 正文保持原样。

## 环境

- .NET SDK **8.0.x**（本机验证用的是 8.0.426）
- 训练另需 Python 3.10+ 和 **CPU 版** PyTorch
- 推理建议 8 GB 内存；Q4_K_M 权重约 469 MiB，加载后进程大约 1.5–2.5 GB
- LoRA 训练建议 16 GB 内存。0.5B 以 fp32 加载大约 2 GB 权重，加上激活后常见占用数 GB。8 GB 机器可能在加载阶段被系统杀掉
- CPU 训练很慢。示例数据、`max_steps=20`、序列长度 256、batch 1，在普通笔记本 CPU 上可能要几分钟到几十分钟。不要用这个路径跑大样本全参微调

## 还原和构建

```bash
dotnet restore
dotnet build
dotnet test
```

`dotnet build` 不下载模型，也不需要 PostgreSQL 或 GPU。

解决方案是 `TrainLoop.sln`。上游的 `TrainLoop.slnx` 需要 .NET 9+ SDK，这里改成传统 sln。

## 标注 API 和网页

标注数据在 PostgreSQL。连接串在 `src/TrainLoop.Api/appsettings.json` 的 `ConnectionStrings:DefaultConnection`（默认 `postgres/postgres`，库名 `trainloop`）。没有数据库时进程仍能启动，数据集接口会在第一次访问时失败。Qwen 接口不连数据库。

```bash
dotnet run --project src/TrainLoop.Api --launch-profile http
# http://localhost:5231
# GET /api/datasets
# GET /api/reviewers
# GET /api/datasets/{id}/corrections.jsonl   把已有标注导出成训练 JSONL
# 开发环境另有 /swagger （.NET 8 没有 net9+ 的 MapOpenApi，这里用 Swashbuckle）
```

```bash
dotnet run --project src/TrainLoop.Web --launch-profile http
# http://localhost:5094
# /            首页
# /llm         CPU 路径说明
```

`TrainLoop.Worker` 仍是上游的后台循环，不负责下载或训练 Qwen。

## 下载 Qwen2.5-0.5B

推理用 GGUF，训练用 Hugging Face 上的原始仓库。两套文件都不要提交。

```bash
bash tools/download-qwen-gguf.sh
# 得到 models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

等价命令：

```bash
mkdir -p models
curl -L --fail --retry 3 -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

这是官方仓库 `Qwen/Qwen2.5-0.5B-Instruct-GGUF` 的 `qwen2.5-0.5b-instruct-q4_k_m.gguf`。配置项 `Llm:GgufPath` 的默认值与此路径一致。API 会从内容根目录向上查找该相对路径。

训练脚本第一次运行时会自己下载 `Qwen/Qwen2.5-0.5B-Instruct`（Hugging Face id，safetensors 大约 1 GB）到 `~/.cache/huggingface`。这是公开模型，不需要把 token 写进仓库。

## CPU 推理

```bash
dotnet run --project src/TrainLoop.Api --launch-profile http
```

```bash
curl -s http://localhost:5231/api/llm/runtime

curl -s http://localhost:5231/api/llm/infer \
  -H 'content-type: application/json' \
  -d '{"prompt":"用一句话说明什么是标注理由。","maxTokens":64,"temperature":0.2}'
```

实现是 `LlamaSharpQwenInferenceService`：

- 只引用 `LLamaSharp.Backend.Cpu`
- `GpuLayerCount = 0`，`FlashAttention = false`
- 上下文默认 1024，线程数默认 `Environment.ProcessorCount`
- 提示词按 Qwen2.5 Instruct 标记手工拼接（`<|im_start|>` / `<|im_end|>`），不依赖 GPU 模板
- 同一进程同时只跑一条生成
- 找不到 GGUF 时返回 400，并提示下载命令

Linux 上如果原生库加载失败，安装 `libgomp1`。不要改去引用 `LLamaSharp.Backend.Cuda*`。

## CPU 训练

默认训练器 **不是** 纯 C#。`ExternalProcessTrainingOrchestrator` 用参数数组启动进程（不经 shell），并设置：

- `CUDA_VISIBLE_DEVICES=`
- `HIP_VISIBLE_DEVICES=`

脚本在 `import torch` 之前再次清空这两个变量，然后 `model.to(torch.device("cpu"))`，`fp16=false`，`bf16=false`，LoRA rank 8，batch size 1。

### 1. 安装 CPU 版 PyTorch 和依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r tools/cpu-trainer/requirements.txt
```

如果后面的安装把 torch 换成了 CUDA wheel，再执行一次 CPU wheel 的安装命令。

### 2. 直接训练示例纠错集

示例文件是 `tools/cpu-trainer/sample/corrections.jsonl`（4 条 messages）。

```bash
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
  bash tools/cpu-trainer/run-cpu-train.sh
```

或：

```bash
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
python3 tools/cpu-trainer/train_qwen_lora.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset tools/cpu-trainer/sample/corrections.jsonl \
  --output saves/qwen2.5-0.5b-cpu-lora \
  --max-steps 20 \
  --batch-size 1 \
  --grad-accum 4 \
  --max-length 256 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --learning-rate 1e-4
```

`--max-steps 0` 时改用 `--epochs`。产物是 PEFT adapter 目录（`adapter_config.json`、`adapter_model.safetensors`），已写入 `.gitignore`。

不下载模型的自检：

```bash
python3 tools/cpu-trainer/train_qwen_lora.py --dry-run \
  --dataset tools/cpu-trainer/sample/corrections.jsonl \
  --output /tmp/trainloop-dry-run
```

应打印 `"device": "cpu"`。

JSONL 每行可以是：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"标签：...\n理由：..."}]}
```

或：

```json
{"instruction":"...","input":"...","output":"..."}
```

已有标注可从 API 导出：

```bash
curl -s "http://localhost:5231/api/datasets/<dataset-id>/corrections.jsonl" -o corrections.jsonl
```

导出的助手句是 `标签：{label}\n理由：{rationale}`。这是监督样本，不是“错误标签 → 改正”的对比数据。示例 JSONL 才是带审阅意见的纠错格式。

### 3. 经 API 编排

`dryRun` 默认 **true**，只返回计划，不启动进程。真正训练要显式关掉，并且本机已装好上面的 Python 环境。

```bash
curl -s http://localhost:5231/api/llm/train/plan \
  -H 'content-type: application/json' \
  -d '{"maxSteps":20,"batchSize":1}'

curl -s http://localhost:5231/api/llm/train \
  -H 'content-type: application/json' \
  -d '{"dryRun":false,"maxSteps":20,"batchSize":1,"datasetPath":"tools/cpu-trainer/sample/corrections.jsonl"}'
```

编排器拒绝 batch size 大于 4、序列长度大于 2048，以及以 `-` 开头的路径，避免把路径当成 flag。训练接口会在本机启动进程，不要暴露到不受信任的网络。

### 4. 可选：LLaMA-Factory

配置在 `tools/cpu-trainer/llamafactory/qwen25_0.5b_cpu_lora.yaml`。数据是 alpaca JSONL，登记在 `tools/cpu-trainer/llamafactory/data/dataset_info.json` 的 `trainloop_corrections`。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llamafactory
# llamafactory 可能会换上 CUDA 版 torch，装完后把 CPU wheel 再装一次
pip install torch --index-url https://download.pytorch.org/whl/cpu

CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
  llamafactory-cli train tools/cpu-trainer/llamafactory/qwen25_0.5b_cpu_lora.yaml
```

API 里把 `Llm:Trainer` 设为 `llamafactory` 后，编排器会调用 `llamafactory-cli train`，并强制 `bf16=false`、`fp16=false`、`flash_attn=disabled`。此时 `datasetPath` 必须是含 `dataset_info.json` 的目录；messages JSONL 请继续用默认的 `transformers-peft`。

### 5. 在 CPU 上跑刚训出的 adapter

LLamaSharp 这条推理路径 **不加载** LoRA。用 Python 验证适配器：

```bash
CUDA_VISIBLE_DEVICES="" python3 tools/cpu-trainer/infer_peft_cpu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter saves/qwen2.5-0.5b-cpu-lora \
  --prompt "文本：音质不错但是太贵。当前标签：positive。审阅意见：价格是否定的。" \
  --max-new-tokens 64
```

把 adapter 合并回 GGUF 再交给 LLamaSharp，需要额外的 llama.cpp 转换，本仓库没有做这一步。

## 项目布局

| 路径 | 作用 |
| --- | --- |
| `src/TrainLoop.Core` | 标注实体，以及 Qwen 的请求/计划/JSONL 导出（无原生依赖） |
| `src/TrainLoop.Infrastructure` | EF Core + PostgreSQL，测验引擎 |
| `src/TrainLoop.Api` | 标注 REST，以及 `/api/llm/*` |
| `src/TrainLoop.Llm` | `LlamaSharpQwenInferenceService`、`ExternalProcessTrainingOrchestrator` |
| `src/TrainLoop.Web` | Blazor 首页和 `/llm` 说明页 |
| `src/TrainLoop.Worker` | 上游后台循环 |
| `tools/cpu-trainer` | CPU LoRA 脚本、示例数据和 LLaMA-Factory YAML |
| `docs/RFC-0001-trainloop-architecture.md` | 上游架构说明，未改写成 LLM 微调设计 |

## 已知限制

- 训练不是 C# 实现。不要把编排器描述成进程内反向传播。
- 推理服务的是基础 GGUF，不是刚训完的 adapter。
- CPU 上 0.5B 的 20 step 只适合打通流程，不足以得到可用的校对模型。
- 上游 RFC 里的 ML.NET meta-model、共识策略、MCP、outbox 都还没有实现。
- 标注 API 需要单独准备 PostgreSQL。示例连接串是本地开发默认值，不是密钥管理方案。
- 量化推理（Q4_K_M）和 fp32 LoRA 训练用的不是同一份权重文件。
- .NET 10 模板里的 `MapStaticAssets`、`@Assets`、`ImportMap` 在 net8.0 上不可用，网页已改回 .NET 8 的静态文件和 Blazor 路由写法。

## English

This is an MIT-licensed port of [JerrettDavis/TrainLoop](https://github.com/JerrettDavis/TrainLoop) (`2d39a803`, 2026-10-19) retargeted to **net8.0**. Upstream labels data and calibrates reviewers; it does not fine-tune LLMs. This repo adds:

- **CPU inference** of `Qwen/Qwen2.5-0.5B-Instruct` through LLamaSharp 0.19 and `LLamaSharp.Backend.Cpu` (llama.cpp with Qwen2 support, which is the architecture Qwen2.5 GGUF uses). `GpuLayerCount` is forced to 0. Versions 0.20 and later pull Microsoft.Extensions 9.x/10.x, so they are not referenced. Download the official Q4_K_M GGUF with `bash tools/download-qwen-gguf.sh`, then `POST /api/llm/infer`.
- **CPU LoRA training** orchestrated from C#. The default trainer is `tools/cpu-trainer/train_qwen_lora.py` (transformers + peft, fp32, `model.to("cpu")`, batch size 1). LLaMA-Factory is an optional YAML (`tools/cpu-trainer/llamafactory/qwen25_0.5b_cpu_lora.yaml`) launched with `CUDA_VISIBLE_DEVICES=""` and `bf16/fp16` off. Training is not in-process C#.

```bash
dotnet build
dotnet test
bash tools/download-qwen-gguf.sh
dotnet run --project src/TrainLoop.Api --launch-profile http
python3 tools/cpu-trainer/train_qwen_lora.py --dry-run \
  --dataset tools/cpu-trainer/sample/corrections.jsonl \
  --output /tmp/trainloop-dry-run
```

The inference service loads the base GGUF only. Validate a trained adapter with `tools/cpu-trainer/infer_peft_cpu.py`. Expect roughly 1.5–2.5 GB RAM for Q4 inference and 8–16 GB for a short fp32 LoRA run. A 20-step CPU job is a smoke test, not a production fine-tune.
