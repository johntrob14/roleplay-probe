# roleplay-probe

Token-labeled Qwen3.6-27B rollouts for training probes that detect role-play (the model acting as
something other than an AI) and talk about role-play in the chain of thought.

Pipeline (runs on the RunPod pod mats4; results in /workspace/results/roleplay-probe):

1. `scripts/serve.sh`: vLLM server for Qwen/Qwen3.6-27B (bf16), used for generation and judging.
2. `src/build_inputs.py`: imports eekay/weirdchat-replays Qwen3.6-27B rollouts, collects the unique
   weirdchat prompts, and samples benign English first-turn WildChat-1M prompts.
3. `src/generate.py`: new on-policy rollouts (thinking mode) for weirdchat and WildChat prompts.
4. `src/judge.py`: span judge (3 runs, unanimity) and output judge (3 runs, majority), prompts in
   `prompts/`.
5. `src/assemble.py`: full templated text with labeled char and token spans -> labeled/dataset.jsonl.
