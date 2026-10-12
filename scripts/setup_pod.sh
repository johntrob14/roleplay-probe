#!/bin/bash
# One-time pod setup: project venv with vLLM, and Qwen3.6-27B weights into HF_HOME (scratch).
set -euxo pipefail
cd /workspace/roleplay-probe
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python vllm datasets pandas pyarrow huggingface_hub openai tqdm
.venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
.venv/bin/hf download Qwen/Qwen3.6-27B --exclude '*.mp4'
echo SETUP_DONE
