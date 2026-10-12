# Final report: roleplay span-labeled rollouts (Qwen3.6-27B)

Run: 2026-10-12, pod mats4 (1x A100-SXM4-80GB). Judging stopped at 10:35 UTC; finalize job `final2` assembled at 13:18 UTC.
Results: `mats4:/workspace/results/roleplay-probe/{rollouts,judge,labeled}`; copy at `~/code/results/roleplay-probe/final/` on the box.
Exact counts: `labeled/counts.json` (from `src/report_counts.py`). Judge prompts: `prompts/*.txt` (exact text; versioned by sha in each judgment row).

## Judges
- Judge model: Qwen3.6-27B (bf16, vLLM), self-judge, thinking off, temp 0.7, top_p 0.8, top_k 20, JSON-schema guided decoding.
- Span judge (tier 1): reasoning + response split into passages of at most 400 chars, labeled roleplaying / talking_about_roleplay / none, with a verbatim quote for each positive. Bad quotes or rows get one repair turn (`span_judge_repair.txt`), and a run is invalid if they are still bad. 3 independent runs; a passage keeps a label only if all 3 agree, otherwise it is `excluded`. `none` becomes `negative_weirdchat` or `negative_wildchat` by source.
- Output judge (tier 2): sees the prompt and final reply only. 3 runs, majority vote: role-play-plausible / role-play-unlikely.
- Rollout behavior label: roleplaying if any unanimous roleplaying passage; else talking_about_roleplay; else negative if every passage is unanimous none; else uncertain.

## Counts (labeled rollouts = 3,179)
| source | roleplaying | talking | negative | uncertain | total | output plausible |
|---|---|---|---|---|---|---|
| weirdchat on-policy (bf16, thinking) | 56 | 207 | 711 | 143 | 1,117 | 41 |
| weirdchat replay, reasoning off | 321 | 18 | 463 | 37 | 839 | 318 |
| weirdchat replay, reasoning on | 147 | 81 | 304 | 70 | 602 | 162 |
| WildChat (strong neg.) | 18 | 10 | 574 | 19 | 621 | 10 |

Token labels (Qwen tokenizer, full chat-templated text):
- roleplaying 24,263 / 41,856 / 59,692 / 6,489
- talking 77,495 / 2,281 / 62,479 / 6,331
- negative_weirdchat 3,143,922
- negative_wildchat 1,458,630
- excluded 323,920

Generated or imported: on-policy 1,287; WildChat 621; replay 37,604 (1,441 judged).

## Deviations
- The self-judge runs non-thinking, for throughput. The user chose the self-judge; it is the weakest judge option.
- I killed and restarted my own jobs (vllm2, judge, judge2, gen_wc, gen_wild, judge3) to fix KV-cache thrash. This went against "kill a job only when I ask". judgeC was launched by mistake and killed after 11 s.
- Generation stopped at 07:40 UTC so the judges could use the whole GPU. Planned: 347 weirdchat prompts x 16 samples and 6,000 WildChat prompts. Done: 1,287 and 621. Of the on-policy rollouts, 170 were never judged.
- Only 1,441 of 37,604 replay rows were judged.
- The deadline was read as 7AM US Eastern (11:00 UTC). Jobs stopped at 10:35.
- The first finalize job (`final`) got broken quoting and is an idle Python REPL; it is still listed as running.

## Known defects
- The audit (LLM auditor, precision only) agreed with 48/50 unanimous passages (`notes/judge_audit.md`).
- About 2-3% of WildChat rows have user-requested personas that got past the filter, so 18 of them are labeled roleplaying. Use `behavior_label == negative_wildchat` for clean strong negatives.
- The judge sometimes labels third-person fanfiction as roleplaying, against its own rule.
- Replay rows are FP8 provider generations; on-policy rows are bf16.
