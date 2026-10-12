"""Label rollouts with two judges, both Qwen3.6-27B (non-thinking) served by the local vLLM server.

Tier A, span judge (sees prompt, reasoning, response): labels every passage roleplaying /
  talking_about_roleplay / none, with a verbatim quote for each positive. Follows the passage-index
  design of Bergen et al. 2026 (arXiv:2609.19101): one row per passage, quotes checked against the
  passage, one repair turn for missing rows or unmatched quotes, then the run's bad rows are marked
  invalid. N_CONSENSUS independent runs; a passage gets a label only when all runs agree.
Tier B, output judge (sees prompt and final response only): role-play-plausible vs
  role-play-unlikely, N_CONSENSUS runs, majority vote.

Runs as a loop: rescans rollouts/*.jsonl, judges pending rollouts in a stratified order (each source
gets an equal share; inside a source, round-robin over behavior x reasoning x Ethan-judge strata),
appends to judge/judgments.jsonl. Stop with --until (UTC ISO time) or kill it; rerun resumes.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI

from common import (BASE_URL, JUDGE, JUDGE_SAMPLING, N_CONSENSUS, ROLLOUTS, SERVED_NAME,
                    SPAN_LABELS, append_jsonl, done_keys, passages_for, read_jsonl)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
P = {f.stem: f.read_text() for f in PROMPTS.glob("*.txt")}
PROMPT_VERSION = hashlib.sha256("".join(P[k] for k in sorted(P)).encode()).hexdigest()[:10]
OUT = JUDGE / "judgments.jsonl"

SPAN_SCHEMA = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "pid": {"type": "string"},
            "label": {"type": "string", "enum": list(SPAN_LABELS)},
            "evidence": {"type": "string", "enum": ["certain", "ambiguous"]},
            "quote": {"type": "string"},
            "why": {"type": "string"},
        },
        "required": ["pid", "label"],
    }}},
    "required": ["rows"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["role-play-plausible", "role-play-unlikely"]},
        "odd_behavior": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["label", "odd_behavior", "why"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def render_passages(ps):
    return "\n\n".join(f"[{p['pid']} | {p['channel']}]\n{p['text']}" for p in ps)


def check_rows(rows, ps):
    """Return (by_pid, problems). by_pid maps pid -> validated row."""
    want = {p["pid"]: p for p in ps}
    by_pid, problems = {}, []
    for r in rows:
        pid = r.get("pid")
        if pid not in want:
            problems.append(f"- {pid}: no such passage")
            continue
        if pid in by_pid:
            problems.append(f"- {pid}: duplicated row")
            continue
        if r["label"] != "none":
            q = r.get("quote") or ""
            if not q.strip() or norm(q) not in norm(want[pid]["text"]):
                problems.append(f"- {pid}: quote is not an exact substring of {pid}")
                continue
        by_pid[pid] = r
    missing = [pid for pid in want if pid not in by_pid and not any(pr.startswith(f"- {pid}:") for pr in problems)]
    if missing:
        problems.append(f"- missing rows for: {', '.join(missing)}")
    return by_pid, problems


async def chat(client, messages, schema, max_tokens, seed):
    r = await client.chat.completions.create(
        model=SERVED_NAME, messages=messages, max_tokens=max_tokens, seed=seed,
        response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        **JUDGE_SAMPLING)
    return r.choices[0].message.content or "", r.usage.completion_tokens


def parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


async def span_run(client, ro, ps, seed):
    n = len(ps)
    msgs = [{"role": "system", "content": P["span_judge_system"]},
            {"role": "user", "content": P["span_judge_user"].format(
                prompt=ro["prompt"], n=n, passages=render_passages(ps))}]
    max_tokens = min(16000, 400 + 60 * n)
    raw, toks = await chat(client, msgs, SPAN_SCHEMA, max_tokens, seed)
    obj = parse(raw)
    by_pid, problems = check_rows((obj or {}).get("rows", []), ps)
    log = [dict(raw=raw, tokens=toks, problems=problems)]
    if problems:
        msgs += [{"role": "assistant", "content": raw},
                 {"role": "user", "content": P["span_judge_repair"].format(problems="\n".join(problems[:40]), n=n)}]
        raw2, toks2 = await chat(client, msgs, SPAN_SCHEMA, max_tokens, seed + 1000)
        obj2 = parse(raw2)
        by2, problems2 = check_rows((obj2 or {}).get("rows", []), ps)
        # Keep first-pass rows that were valid; take repaired rows for the rest.
        for pid, r in by2.items():
            by_pid.setdefault(pid, r)
        log.append(dict(raw=raw2, tokens=toks2, problems=problems2))
    labels = {p["pid"]: (by_pid[p["pid"]]["label"] if p["pid"] in by_pid else "invalid") for p in ps}
    pos = {pid: {k: r.get(k) for k in ("label", "evidence", "quote", "why")}
           for pid, r in by_pid.items() if r["label"] != "none"}
    return dict(labels=labels, positives=pos, calls=log)


async def output_run(client, ro, seed):
    msgs = [{"role": "system", "content": P["output_judge_system"]},
            {"role": "user", "content": P["output_judge_user"].format(prompt=ro["prompt"], response=ro["response"])}]
    raw, toks = await chat(client, msgs, OUTPUT_SCHEMA, 600, seed)
    obj = parse(raw) or {}
    return dict(label=obj.get("label", "invalid"), odd_behavior=obj.get("odd_behavior"),
                why=obj.get("why"), raw=raw, tokens=toks)


async def judge_one(client, sem, ro, stats):
    ps = passages_for(ro)
    async with sem:
        t0 = time.time()
        try:
            span_runs = await asyncio.gather(*(span_run(client, ro, ps, s) for s in range(N_CONSENSUS)))
            if ro["response"].strip():
                out_runs = await asyncio.gather(*(output_run(client, ro, 100 + s) for s in range(N_CONSENSUS)))
            else:
                out_runs = []
        except Exception as e:
            stats["err"] += 1
            print("ERR", ro["rid"], repr(e)[:300], flush=True)
            return
    consensus = {}
    for p in ps:
        ls = {r["labels"][p["pid"]] for r in span_runs}
        consensus[p["pid"]] = ls.pop() if len(ls) == 1 and "invalid" not in ls else "excluded"
    votes = Counter(r["label"] for r in out_runs)
    if not out_runs:
        out_label = "no_response"
    else:
        top, c = votes.most_common(1)[0]
        out_label = top if c * 2 > len(out_runs) and top != "invalid" else "no_majority"
    append_jsonl(OUT, dict(
        rid=ro["rid"], source=ro["source"], judge_model="Qwen/Qwen3.6-27B (non-thinking, self-judge)",
        prompt_version=PROMPT_VERSION, n_runs=N_CONSENSUS,
        passages=[{k: p[k] for k in ("pid", "channel", "start", "end")} for p in ps],
        consensus=consensus, span_runs=span_runs,
        output_label=out_label, output_votes=dict(votes), output_runs=out_runs,
        secs=round(time.time() - t0, 1)))
    stats["ok"] += 1
    if stats["ok"] % 100 == 0:
        el = time.time() - stats["t0"]
        print(f"judged {stats['ok']} ({stats['ok'] / el * 3600:.0f}/h), {stats['err']} errors", flush=True)


def stratified_order(rollouts):
    """Interleave sources one by one; inside a source, round-robin over strata
    (behavior x reasoning_enabled x Ethan's judge match), in a fixed hash order inside each stratum."""
    strata = defaultdict(list)
    for r in rollouts:
        ej = (r.get("ethan_judge") or {}).get("match")
        strata[(r["source"], str(r.get("behavior_id")), str(r.get("reasoning_enabled")), str(ej))].append(r)
    per_source = defaultdict(list)
    for k in sorted(strata):
        rs = sorted(strata[k], key=lambda r: hashlib.sha256(r["rid"].encode()).hexdigest())
        per_source[k[0]] += [((i, k), r) for i, r in enumerate(rs)]
    queues = {s: [r for _, r in sorted(v, key=lambda x: x[0])] for s, v in sorted(per_source.items())}
    order, i = [], 0
    while any(i < len(q) for q in queues.values()):
        order += [q[i] for q in queues.values() if i < len(q)]
        i += 1
    return order


async def main(a):
    until = dt.datetime.fromisoformat(a.until) if a.until else None
    client = AsyncOpenAI(base_url=BASE_URL, api_key="x", timeout=3600)
    sem = asyncio.Semaphore(a.concurrency)
    stats = {"ok": 0, "err": 0, "t0": time.time()}
    while True:
        if until and dt.datetime.now(dt.timezone.utc) >= until:
            print("reached --until, stopping", flush=True)
            return
        done = done_keys(OUT)
        pending = [r for f in sorted(ROLLOUTS.glob("*.jsonl")) for r in read_jsonl(f)
                   if r["rid"] not in done and (not a.sources or r["source"] in a.sources)]
        if not pending:
            print("nothing pending; sleeping 120s", flush=True)
            await asyncio.sleep(120)
            continue
        batch = stratified_order(pending)[: a.batch]
        print(f"{len(pending)} pending; judging batch of {len(batch)} "
              f"({dict(Counter(r['source'] for r in batch))})", flush=True)
        await asyncio.gather(*(judge_one(client, sem, r, stats) for r in batch))
        if a.once:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--batch", type=int, default=600)
    ap.add_argument("--until", default="", help="UTC ISO time, e.g. 2026-10-12T11:00:00+00:00")
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--once", action="store_true")
    asyncio.run(main(ap.parse_args()))
