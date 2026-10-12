"""Build the input sets.

1. rollouts/weirdchat_replay.jsonl : Ethan's existing Qwen3.6-27B rollouts (eekay/weirdchat-replays,
   records config, model == qwen3.6-27b), imported as is.
2. inputs/weirdchat_prompts.jsonl  : every unique weirdchat prompt (all models), for new on-policy
   generations with Qwen3.6-27B in thinking mode.
3. inputs/wildchat_prompts.jsonl   : benign first-turn English prompts from allenai/WildChat-1M.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re

import pandas as pd
from huggingface_hub import hf_hub_download

from common import INPUTS, ROLLOUTS, append_jsonl, done_keys

WC_REPO = "eekay/weirdchat-replays"
WC_FILES = ["records/train-00000-of-00002.parquet", "records/train-00001-of-00002.parquet"]

# Prompts that ask for role-play, personas, or fiction are removed from WildChat so it stays a
# clean "normal assistant" negative. Case-insensitive.
WILD_EXCLUDE = re.compile(
    r"role[- ]?play|\brp\b|pretend|act as|acting as|you are (a|an|my|now)|you're (a|an|my|now)|"
    r"persona|character|in character|stay in|story|fanfic|fiction|narrat|scenario|imagine you|"
    r"jailbreak|\bDAN\b|developer mode|simulat|girlfriend|boyfriend|waifu|"
    r"respond as|reply as|speak as|talk as|answer as|from now on|"
    r"are you (human|real|a bot|an ai|sentient|conscious|alive)",
    re.I,
)


def sha(s: str, n: int = 12) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def build_weirdchat():
    df = pd.concat([pd.read_parquet(hf_hub_download(WC_REPO, f, repo_type="dataset")) for f in WC_FILES])
    INPUTS.mkdir(parents=True, exist_ok=True)

    out = INPUTS / "weirdchat_prompts.jsonl"
    have = done_keys(out, "prompt_id")
    for prompt, g in df.groupby("prompt", sort=True):
        pid = "wc-" + sha(prompt)
        if pid in have:
            continue
        append_jsonl(out, dict(prompt_id=pid, prompt=prompt,
                               behavior_id=sorted(g.behavior_id.unique().tolist())[0],
                               behavior_ids=sorted(g.behavior_id.unique().tolist()),
                               models=sorted(g.model.unique().tolist()), n_records=len(g)))

    q = df[df.model == "qwen3.6-27b"]
    out = ROLLOUTS / "weirdchat_replay.jsonl"
    have = done_keys(out)
    for r in q.itertuples(index=False):
        rid = f"wcr:{r.run}:{r.idx}"
        if rid in have:
            continue
        append_jsonl(out, dict(
            rid=rid, source="weirdchat_replay", prompt_id="wc-" + sha(r.prompt), prompt=r.prompt,
            reasoning=r.reasoning or "", response=r.completion or "",
            reasoning_enabled=bool(r.reasoning_enabled), behavior_id=r.behavior_id,
            gen=dict(model="qwen/qwen3.6-27b (Qwen3.6-27B-FP8 via OpenRouter)", provider=r.provider,
                     run=r.run, idx=int(r.idx), pattern_id=r.pattern_id),
            ethan_judge=dict(match=bool(r.judge_match), explanation=r.judge_explanation),
        ))
    print(f"weirdchat: {df.prompt.nunique()} unique prompts, {len(q)} qwen3.6-27b rollouts")


def build_wildchat(n: int, seed: int):
    from datasets import load_dataset

    # Shard order and a 20k-row buffer are shuffled so the sample is not just the earliest chats.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True).shuffle(seed=seed, buffer_size=20000)
    rng = random.Random(seed)
    seen, kept = set(), []
    # Take the first 40n eligible rows of the shuffled stream, then sample n of them.
    for row in ds:
        if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
            continue
        conv = row["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue
        text = conv[0]["content"].strip()
        if not (20 <= len(text) <= 2000) or WILD_EXCLUDE.search(text):
            continue
        mod = conv[0].get("openai_moderation") or {}
        if mod.get("flagged"):
            continue
        key = text.lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(prompt_id="wild-" + sha(text), prompt=text,
                         wildchat_hash=row.get("conversation_hash"), model=row.get("model")))
        if len(kept) >= 40 * n:
            break
    rng.shuffle(kept)
    out = INPUTS / "wildchat_prompts.jsonl"
    have = done_keys(out, "prompt_id")
    for r in kept[:n]:
        if r["prompt_id"] not in have:
            append_jsonl(out, r)
    print(f"wildchat: {len(kept)} eligible scanned, wrote {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wildchat-n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_weirdchat()
    build_wildchat(a.wildchat_n, a.seed)
