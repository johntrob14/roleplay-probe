"""Print counts of rollouts, judgments, and labels so far. Read-only."""
from __future__ import annotations

from collections import Counter

from common import JUDGE, LABELED, ROLLOUTS, read_jsonl

ro = Counter()
for f in sorted(ROLLOUTS.glob("*.jsonl")):
    ro[f.stem] = sum(1 for _ in read_jsonl(f))
print("rollouts:", dict(ro))

js = list({j["rid"]: j for j in reversed(list(read_jsonl(JUDGE / "judgments.jsonl")))}.values())  # first judgment per rid
print("judged:", dict(Counter(j["source"] for j in js)))
beh, out, pas, both = Counter(), Counter(), Counter(), Counter()
for j in js:
    labs = list(j["consensus"].values())
    b = ("roleplaying" if "roleplaying" in labs else "talking_about_roleplay" if "talking_about_roleplay" in labs
         else "negative" if labs and all(x == "none" for x in labs) else "uncertain")
    beh[(j["source"], b)] += 1
    out[(j["source"], j["output_label"])] += 1
    both[(j["source"], b, j["output_label"])] += 1
    for x in labs:
        pas[(j["source"], x)] += 1
for name, c in [("behavior_label", beh), ("output_label", out), ("passages", pas), ("behavior x output", both)]:
    print(f"\n{name}:")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
ds = LABELED / "dataset.jsonl"
if ds.exists():
    tok = Counter()
    for r in read_jsonl(ds):
        for s, e, lab in r["token_spans"]:
            tok[(r["source"], lab)] += e - s
    print("\nassembled token counts:")
    for k, v in sorted(tok.items()):
        print(f"  {k}: {v}")
