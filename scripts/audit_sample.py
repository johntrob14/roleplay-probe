"""Blinded hand-audit sample: prints passages without judge labels; key written separately."""
import json, random, sys, glob
sys.path.insert(0, "/home/john/code/roleplay-probe/src")
import pandas as pd
from common import passages_for, read_jsonl
from pathlib import Path
S = Path(sys.argv[1]); WC = sys.argv[2]
ro = {}
for f in S.glob("*.jsonl"):
    for r in read_jsonl(f): ro[r["rid"]] = r
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(WC + "/records_*.parquet"))])
df = df[df.model == "qwen3.6-27b"]
js = list({j["rid"]: j for j in reversed(list(read_jsonl(S / "judge/judgments.jsonl")))}.values())
need = {j["rid"] for j in js if j["source"] == "weirdchat_replay"}
for r in df.itertuples():
    rid = f"wcr:{r.run}:{r.idx}"
    if rid in need:
        ro[rid] = dict(rid=rid, source="weirdchat_replay", prompt=r.prompt, reasoning=r.reasoning if isinstance(r.reasoning, str) else "", response=r.completion if isinstance(r.completion, str) else "")
for r in ro.values():
    r["reasoning"] = (r.get("reasoning") or "").strip("\n"); r["response"] = (r.get("response") or "").lstrip("\n")
pool = {}
for j in js:
    if j["rid"] not in ro: continue
    ps = {p["pid"]: p for p in passages_for(ro[j["rid"]])}
    for pid, c in j["consensus"].items():
        key = c if c != "none" else ("none_wild" if j["source"] == "wildchat" else "none_wc")
        pool.setdefault(key, []).append((j["rid"], pid, ps[pid]))
rng = random.Random(1)
quota = {"roleplaying": 15, "talking_about_roleplay": 15, "none_wc": 15, "none_wild": 5}
items = [(k, x) for k, n in quota.items() for x in rng.sample(pool.get(k, []), min(n, len(pool.get(k, []))))]
rng.shuffle(items)
print({k: len(v) for k, v in pool.items()})
key = []
for i, (k, (rid, pid, p)) in enumerate(items):
    print(f"\n#{i} [{p['channel']}] prompt: {ro[rid]['prompt'][:140]!r}\n   >> {p['text'][:420]!r}")
    key.append((i, k, rid, pid))
json.dump(key, open("/tmp/claude-1000/-home-john-claude-box/65bc1927-57c8-4b2c-9f01-6de44da5775b/scratchpad/audit_key.json", "w"))
