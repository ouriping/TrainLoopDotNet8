#!/usr/bin/env bash
# CPU LoRA for Qwen2.5-0.5B-Instruct. Does not use an NVIDIA GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM="false"

PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET="${DATASET:-${ROOT}/tools/cpu-trainer/sample/corrections.jsonl}"
OUTPUT="${OUTPUT:-${ROOT}/saves/qwen2.5-0.5b-cpu-lora}"
MAX_STEPS="${MAX_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LORA_RANK="${LORA_RANK:-8}"

exec "${PYTHON}" "${ROOT}/tools/cpu-trainer/train_qwen_lora.py" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --output "${OUTPUT}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --max-length "${MAX_LENGTH}" \
  --lora-rank "${LORA_RANK}" \
  "$@"
