"""Generate on-policy Qwen3.6-27B rollouts (thinking mode) against the local vLLM server.

  python generate.py --source weirdchat_onpolicy --samples 8
  python generate.py --source wildchat --samples 1

Appends to rollouts/<source>.jsonl, one row per (prompt_id, sample). Rerunning skips done rows.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import time

from openai import AsyncOpenAI

from common import (BASE_URL, GEN_MAX_TOKENS, GEN_SAMPLING, INPUTS, ROLLOUTS, SERVED_NAME,
                    append_jsonl, done_keys, read_jsonl)

PROMPT_FILE = {"weirdchat_onpolicy": "weirdchat_prompts.jsonl", "wildchat": "wildchat_prompts.jsonl"}


async def one(client, sem, src, p, k, out, stats, until):
    rid = f"{src}:{p['prompt_id']}:{k}"
    async with sem:
        if until and dt.datetime.now(dt.timezone.utc) >= until:
            return
        t0 = time.time()
        try:
            r = await client.chat.completions.create(
                model=SERVED_NAME, messages=[{"role": "user", "content": p["prompt"]}],
                max_tokens=GEN_MAX_TOKENS, seed=k, **GEN_SAMPLING)
        except Exception as e:  # logged and retried on the next run
            stats["err"] += 1
            print("ERR", rid, repr(e)[:200], flush=True)
            return
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or ""
    append_jsonl(out, dict(
        rid=rid, source=src, prompt_id=p["prompt_id"], prompt=p["prompt"], sample=k,
        reasoning=reasoning, response=m.content or "", reasoning_enabled=True,
        behavior_id=p.get("behavior_id"), finish_reason=r.choices[0].finish_reason,
        usage=dict(prompt=r.usage.prompt_tokens, completion=r.usage.completion_tokens),
        gen=dict(model="Qwen/Qwen3.6-27B", dtype="bf16", engine="vllm", seed=k,
                 max_tokens=GEN_MAX_TOKENS, sampling={k2: v for k2, v in GEN_SAMPLING.items()}),
        secs=round(time.time() - t0, 1)))
    stats["ok"] += 1
    if stats["ok"] % 50 == 0:
        print(f"{src}: {stats['ok']} done, {stats['err']} errors", flush=True)


async def main(a):
    prompts = list(read_jsonl(INPUTS / PROMPT_FILE[a.source]))
    if a.limit:
        prompts = prompts[: a.limit]
    out = ROLLOUTS / f"{a.source}.jsonl"
    have = done_keys(out)
    # Sample index outer loop: every prompt gets sample 0 before any gets sample 1, so a run stopped
    # early still covers all prompts.
    todo = [(p, k) for k in range(a.samples) for p in prompts
            if f"{a.source}:{p['prompt_id']}:{k}" not in have]
    print(f"{a.source}: {len(prompts)} prompts x {a.samples} samples, {len(todo)} to do", flush=True)
    client = AsyncOpenAI(base_url=BASE_URL, api_key="x", timeout=3600)
    sem = asyncio.Semaphore(a.concurrency)
    stats = {"ok": 0, "err": 0}
    until = dt.datetime.fromisoformat(a.until) if a.until else None
    await asyncio.gather(*(one(client, sem, a.source, p, k, out, stats, until) for p, k in todo))
    print(f"{a.source}: finished, {stats['ok']} ok, {stats['err']} errors", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(PROMPT_FILE), required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--until", default="", help="UTC ISO time; no new requests start after it")
    asyncio.run(main(ap.parse_args()))
