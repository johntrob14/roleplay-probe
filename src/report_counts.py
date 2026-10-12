"""Final counts from labeled/dataset.jsonl, written to labeled/counts.json and printed. Read-only on inputs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from common import LABELED, ROLLOUTS, read_jsonl

rows = list(read_jsonl(LABELED / "dataset.jsonl"))
gen = {f.stem: sum(1 for _ in read_jsonl(f)) for f in sorted(ROLLOUTS.glob("*.jsonl"))}


def split(r):
    if r["source"] == "weirdchat_replay":
        return f"weirdchat_replay (reasoning {'on' if r['reasoning_enabled'] else 'off'})"
    return r["source"]


c = defaultdict(Counter)
for r in rows:
    s = split(r)
    c["rollouts_by_behavior_label"][f"{s} | {r['behavior_label']}"] += 1
    c["rollouts_by_output_label"][f"{s} | {r['output_label']}"] += 1
    c["behavior_x_output"][f"{s} | {r['behavior_label']} | {r['output_label']}"] += 1
    for k, v in r["passage_label_counts"].items():
        c["passages"][f"{s} | {k}"] += v
    for k, v in r["token_label_counts"].items():
        c["tokens"][f"{s} | {k}"] += v
    for q in r["quote_spans"]:
        if q["run"] == 0:
            c["quote_spans_run0"][f"{s} | {q['label']}"] += 1
    c["totals"]["labeled_rollouts"] += 1
    c["totals"][f"labeled_rollouts | {s}"] += 1
out = {"generated_or_imported_rollouts": gen, **{k: dict(sorted(v.items())) for k, v in c.items()}}
(LABELED / "counts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
