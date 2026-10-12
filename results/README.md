# Data

Gzipped JSONL from the run on 2026-10-12. See `notes/final_report.md` for counts, judge setup, deviations, and known defects.

- `labeled/dataset.jsonl.gz`: 3,179 labeled rollouts with full chat-templated text, char and token spans, quote spans, behavior and output labels.
- `labeled/counts.json`: counts produced by `src/report_counts.py`.
- `judge/judgments.jsonl.gz`: raw judgments (3 span-judge runs and 3 output-judge runs per rollout, with raw judge outputs).
- `rollouts/weirdchat_onpolicy.jsonl.gz`, `rollouts/wildchat.jsonl.gz`: Qwen3.6-27B (bf16, thinking) generations.
- `inputs/`: 347 weirdchat prompts and 6,000 sampled WildChat-1M prompts.

The full replay import (37,604 rows) is not included. It comes from `eekay/weirdchat-replays` on Hugging Face; the 1,441 judged replay rows are in the labeled dataset and judgments.
WildChat prompts are from allenai/WildChat-1M (ODC-BY).
