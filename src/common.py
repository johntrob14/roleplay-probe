"""Shared config and helpers for the roleplay-probe data pipeline.

Every stage appends one JSONL row per unit of work, keyed by `rid` (rollout id), and skips keys
already present, so any stage can be killed and rerun.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

MODEL = "Qwen/Qwen3.6-27B"
SERVED_NAME = "qwen3.6-27b"
BASE_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")

RESULTS = Path(os.environ.get("RP_RESULTS", "/workspace/results/roleplay-probe"))
INPUTS = RESULTS / "inputs"        # prompt lists and imported replay rollouts
ROLLOUTS = RESULTS / "rollouts"    # one file per source
JUDGE = RESULTS / "judge"          # raw judge calls and per-rollout judgments
LABELED = RESULTS / "labeled"      # assembled token-level datasets

# Qwen3.6 recommended sampling (model card): thinking mode, general tasks.
GEN_SAMPLING = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5,
                    extra_body=dict(top_k=20, min_p=0.0))
GEN_MAX_TOKENS = 8192
# Judge runs in non-thinking mode (recommended non-thinking params). Temperature > 0 so the three
# consensus runs are independent samples.
JUDGE_SAMPLING = dict(temperature=0.7, top_p=0.8, presence_penalty=0.0,
                      extra_body=dict(top_k=20, min_p=0.0,
                                      chat_template_kwargs={"enable_thinking": False}))
N_CONSENSUS = 3

SPAN_LABELS = ("roleplaying", "talking_about_roleplay", "none")
NEG_LABEL_BY_SOURCE = {
    "weirdchat_replay": "negative_weirdchat",
    "weirdchat_onpolicy": "negative_weirdchat",
    "wildchat": "negative_wildchat",
}


def read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a partially written last line after a kill
                    continue
                # Replay rows imported from parquet can carry NaN for a missing reasoning/completion.
                for k in ("reasoning", "response"):
                    if k in row and not isinstance(row[k], str):
                        row[k] = ""
                # vLLM's qwen3 reasoning parser leaves the newlines around </think> on the content;
                # the chat template supplies them, so strip them here for every consumer.
                if "reasoning" in row:
                    row["reasoning"] = row["reasoning"].strip("\n")
                if "response" in row:
                    row["response"] = row["response"].lstrip("\n")
                yield row


def done_keys(path: Path, key: str = "rid") -> set:
    return {r[key] for r in read_jsonl(path)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[*A-Z0-9])")
MAX_PASSAGE_CHARS = 400


def segment(text: str, channel: str) -> list[dict]:
    """Split a channel into passages: paragraphs (blank-line or line breaks), then sentence groups
    of at most MAX_PASSAGE_CHARS. Returns [{channel, start, end, text}] with char offsets into text."""
    out = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):  # paragraph = run of non-blank lines
        p_start, para = m.start(), m.group(0)
        # break long paragraphs: first at single newlines (lists), then at sentence ends
        pieces, pos = [], 0
        for line in re.finditer(r"[^\n]+", para):
            pieces.append((line.start(), line.group(0)))
        cur_s = cur_e = None
        for ls, line in pieces:
            sents, last = [], 0
            for sm in _SENT_SPLIT.finditer(line):
                sents.append((ls + last, ls + sm.start()))
                last = sm.end()
            sents.append((ls + last, ls + len(line)))
            for s, e in sents:
                if cur_s is None:
                    cur_s, cur_e = s, e
                elif e - cur_s <= MAX_PASSAGE_CHARS:
                    cur_e = e
                else:
                    out.append((p_start + cur_s, p_start + cur_e))
                    cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((p_start + cur_s, p_start + cur_e))
    return [dict(channel=channel, start=s, end=e, text=text[s:e]) for s, e in out if text[s:e].strip()]


def passages_for(rollout: dict) -> list[dict]:
    ps = segment(rollout.get("reasoning") or "", "reasoning") + segment(rollout["response"], "response")
    for i, p in enumerate(ps):
        p["pid"] = f"P{i + 1}"
    return ps


def norm_quote_span(passage: str, quote: str):
    """Char span of quote inside passage, matching any run of whitespace loosely; None if absent."""
    words = quote.split()
    if not words:
        return None
    m = re.search(r"\s+".join(re.escape(w) for w in words), passage)
    return (m.start(), m.end()) if m else None
