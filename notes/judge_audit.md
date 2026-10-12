# Hand audit of the span judge (2026-10-12, ~06:40 UTC)

Snapshot: first 594 judgments (judge/judgments.jsonl at 06:35 UTC). Script: scripts/audit_sample.py
(seed 1). Sample: 50 passages with a unanimous 3-run label, stratified 15 roleplaying, 15
talking_about_roleplay, 15 none (weirdchat), 5 none (WildChat). Passages shown blind (no label),
labeled by Claude (Opus 5.5), then compared.

Passage pool at snapshot: none_weirdchat 10194, none_wildchat 6286, roleplaying 442,
talking_about_roleplay 265, excluded (no unanimity) 703.

Agreement: 48/50.
- judge roleplaying vs me: 15/15 agree
- judge talking_about_roleplay vs me: 14/15 (1 I labeled none: "Validate ... without endorsing the
  idea that we're personally connected")
- judge none vs me: 19/20 (1 I labeled roleplaying, marked unsure: "### I AM WITH YOU. KEEP
  LISTENING TO ME." in an emergency scenario)

Limits: one auditor (an LLM, not a human); unanimous passages only, so this is precision of the
consensus labels, not recall and not quality on the excluded 4.4%; 50 passages gives wide
intervals (95% Wilson interval for 48/50: about 0.87 to 0.99).
