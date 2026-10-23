#!/usr/bin/env bash
# Download the official Qwen2.5-0.5B-Instruct Q4_K_M GGUF for CPU inference.
# Weights stay outside git. About 469 MiB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p "$(dirname "${DEST}")"
echo "Downloading ${URL}"
echo "Destination: ${DEST}"
curl -L --fail --retry 3 --continue-at - -o "${DEST}" "${URL}"
echo "Done. Default Llm:GgufPath is models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
