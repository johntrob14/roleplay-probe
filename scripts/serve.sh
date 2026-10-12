#!/bin/bash
# vLLM OpenAI server for Qwen3.6-27B (bf16, text only) on one A100-80GB. Serves both generation
# and the judge. The qwen3 reasoning parser splits <think> content into reasoning_content.
set -euo pipefail
cd /workspace/roleplay-probe
export PATH="/workspace/roleplay-probe/.venv/bin:$PATH"
[ -x .venv/bin/ninja ] || uv pip install --python .venv/bin/python ninja
exec .venv/bin/vllm serve Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --host 127.0.0.1 --port 8000
