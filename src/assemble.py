"""Assemble judged rollouts into a token-labeled dataset: labeled/dataset.jsonl.

Each row holds the full chat-templated text the model would see (Qwen3.6 template, user turn,
assistant turn with <think> block), and labeled spans in both character and token coordinates.

Span labels (one per token):
  prompt                  user turn and template tokens before the assistant's output
  roleplaying             passage the span judge unanimously labeled roleplaying
  talking_about_roleplay  passage unanimously labeled talking_about_roleplay
  negative_weirdchat      passage unanimously labeled none, rollout from weirdchat (in-domain negative)
  negative_wildchat       passage unanimously labeled none, rollout from WildChat (strong negative)
  excluded                passage without unanimous agreement, or a run returned an invalid row
  template                think tags, whitespace between passages, end-of-turn tokens

Rollout-level labels:
  behavior_label  roleplaying if any roleplaying passage; else talking_about_roleplay if any such
                  passage; else negative_<source> if every passage is unanimous none; else uncertain
  output_label    tier-B output judge majority: role-play-plausible / role-play-unlikely
                  (no_majority, no_response possible)
"""
from __future__ import annotations

import bisect
from collections import Counter

from transformers import AutoTokenizer

from common import (JUDGE, LABELED, MODEL, NEG_LABEL_BY_SOURCE, ROLLOUTS, append_jsonl, done_keys,
                    norm_quote_span, read_jsonl)

TOK = AutoTokenizer.from_pretrained(MODEL)


def templated(ro):
    """Return (text, prefix_len, reasoning_off, response_off) for the full conversation."""
    think = bool(ro.get("reasoning_enabled", True))
    prefix = TOK.apply_chat_template([{"role": "user", "content": ro["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=think)
    reasoning = ro.get("reasoning") or ""
    if prefix.rstrip().endswith("</think>"):  # thinking disabled: template already closed the block
        text = prefix
        r_off = None
    else:
        if not prefix.endswith("<think>\n"):
            prefix = prefix + "<think>\n"
        r_off = len(prefix)
        text = prefix + reasoning + "\n</think>\n\n"
    resp_off = len(text)
    text = text + ro["response"] + "<|im_end|>"
    return text, len(prefix), r_off, resp_off


def main():
    LABELED.mkdir(parents=True, exist_ok=True)
    out = LABELED / "dataset.jsonl"
    have = done_keys(out)
    rollouts = {r["rid"]: r for f in ROLLOUTS.glob("*.jsonl") for r in read_jsonl(f)}
    n = 0
    for j in read_jsonl(JUDGE / "judgments.jsonl"):
        if j["rid"] in have or j["rid"] not in rollouts:
            continue
        ro = rollouts[j["rid"]]
        text, prompt_len, r_off, resp_off = templated(ro)
        neg = NEG_LABEL_BY_SOURCE[ro["source"]]
        char_spans = [[0, prompt_len, "prompt"]]
        quotes = []
        for p in j["passages"]:
            base = r_off if p["channel"] == "reasoning" else resp_off
            if base is None:
                continue
            c = j["consensus"][p["pid"]]
            lab = {"none": neg}.get(c, c)
            s, e = base + p["start"], base + p["end"]
            char_spans.append([s, e, lab])
            if lab in ("roleplaying", "talking_about_roleplay"):
                for k, run in enumerate(j["span_runs"]):
                    pos = run["positives"].get(p["pid"])
                    if pos and pos.get("quote"):
                        qs = norm_quote_span(text[s:e], pos["quote"])
                        if qs:
                            quotes.append(dict(run=k, pid=p["pid"], label=lab, evidence=pos.get("evidence"),
                                               start=s + qs[0], end=s + qs[1], why=pos.get("why")))
        enc = TOK(text, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        ends = [b for a, b in offs]
        tok_labels = ["template"] * len(offs)
        for s, e, lab in char_spans:
            i = bisect.bisect_right(ends, s)  # first token ending after s
            while i < len(offs) and offs[i][0] < e:
                if tok_labels[i] == "template" or lab != "prompt":
                    tok_labels[i] = lab
                i += 1
        tok_spans, i = [], 0
        while i < len(tok_labels):
            k = i
            while k < len(tok_labels) and tok_labels[k] == tok_labels[i]:
                k += 1
            tok_spans.append([i, k, tok_labels[i]])
            i = k
        for q in quotes:
            q["tok_start"] = bisect.bisect_right(ends, q["start"])
            q["tok_end"] = bisect.bisect_left(ends, q["end"]) + 1
        labs = [j["consensus"][p["pid"]] for p in j["passages"]]
        if "roleplaying" in labs:
            beh = "roleplaying"
        elif "talking_about_roleplay" in labs:
            beh = "talking_about_roleplay"
        elif labs and all(x == "none" for x in labs):
            beh = neg
        else:
            beh = "uncertain"
        append_jsonl(out, dict(
            rid=ro["rid"], source=ro["source"], prompt_id=ro["prompt_id"], behavior_id=ro.get("behavior_id"),
            reasoning_enabled=ro.get("reasoning_enabled"), behavior_label=beh, output_label=j["output_label"],
            passage_label_counts=dict(Counter(labs)), token_label_counts=dict(Counter(tok_labels)),
            n_tokens=len(offs), text=text, char_spans=char_spans, token_spans=tok_spans, quote_spans=quotes,
            tokenizer=MODEL, judge_prompt_version=j["prompt_version"]))
        n += 1
    print(f"assembled {n} new rows into {out}")


if __name__ == "__main__":
    main()
