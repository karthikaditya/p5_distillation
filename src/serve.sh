#!/usr/bin/env bash
# Phase 5: serve the base model + LoRA adapters with vLLM (continuous batching + paged attention).
# Usage: bash src/serve.sh outputs/r8 [outputs/r32]
set -e
BASE=${STUDENT_BASE:-Qwen/Qwen2.5-1.5B-Instruct}
MODULES="student=$1"
[ -n "$2" ] && MODULES="$MODULES student_r32=$2"
vllm serve "$BASE" --enable-lora --lora-modules $MODULES --max-lora-rank 32 \
  --max-model-len 1024 --gpu-memory-utilization 0.90 --port 8000
