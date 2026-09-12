# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Pilot run: base accuracy, blindfolded prior, and the screen-vs-generation disagreement.

This is the first experiment the plan actually needs, and it needs no probes. It answers
four questions that everything downstream depends on:

  1. Does the difficulty ladder put the base error rate near 40-50%? If not, the generator
     is recalibrated before any GPU hour is spent on probes.
  2. What is the per-item **measured** prior, p_blind(true answer)? The generator's
     canonical/anti arm is only a knob for producing spread in this quantity; the analysis
     conditions on the measurement, not on the arm label. That matters because for counting
     and spatial the arm is recoverable from the answer, so an arm-level comparison would
     confound prior with answer class.
  3. What share of items are answered correctly with no visual evidence at all (category L)?
  4. Do candidate-scoring and free generation agree? The plan screens by scoring and decides
     by generation; the disagreement rate is a reported finding, not a threshold to pass.

Blindfold = a mid-grey image of the same size, not a removed image: the visual token count
and the positional structure stay identical, so the only thing withheld is the content.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadm as L

def parse_in_space(text, space):
    """Match the first token in `text` that is a member of THIS item's answer space.

    The previous per-family regexes hardcoded an alphabet (chart used [A-G]) and silently
    scored every answer outside it as wrong. At chart level 3 the space is A..I, so 59% of
    the model's replies -- every "H" and "I" -- were counted as errors. Deriving the pattern
    from the item's own answer_space makes that class of bug impossible.
    """
    t = text.strip()
    up = t.upper()
    # longest-first so "10" is preferred over "1", and multi-word options match whole
    for cand in sorted(space, key=len, reverse=True):
        import re as _re
        if _re.search(r"(?<![A-Za-z0-9])" + _re.escape(cand.upper()) + r"(?![A-Za-z0-9])", up):
            return cand
    return None


PARSE = {
    "counting": lambda t: (re.search(r"\d+", t) or [None])[0] if re.search(r"\d+", t) else None,
    "tracking": lambda t: (re.search(r"\d+", t) or [None])[0] if re.search(r"\d+", t) else None,
    "spatial":  lambda t: next((w for w in re.findall(r"[a-z]+", t.lower())
                                if w in ("left", "right", "above", "below")), None),
    "chart":    lambda t: next((c for c in re.findall(r"\b([A-G])\b", t.upper())), None),
}
for k in ("counting", "tracking"):
    PARSE[k] = (lambda t: (m.group(0) if (m := re.search(r"\d+", t)) else None))


def parse(fam, text, space=None):
    return parse_in_space(text, space) if space else PARSE[fam](text)


@torch.inference_mode()
def score_candidates(model, proc, image, question, candidates, device):
    """Sum of token log-probs of each candidate as the assistant's reply. One forward pass
    per candidate; answer spaces here are <= 13 so this stays cheap."""
    out = []
    for c in candidates:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + c
        enc_p = proc(text=[prompt], images=[image], return_tensors="pt").to(device)
        enc_f = proc(text=[full],  images=[image], return_tensors="pt").to(device)
        n_prompt = enc_p.input_ids.shape[1]
        logits = model(**enc_f).logits[0]                       # [T, V]
        ids = enc_f.input_ids[0]
        lp = torch.log_softmax(logits[:-1].float(), -1)
        tok_lp = lp[torch.arange(n_prompt - 1, ids.shape[0] - 1), ids[n_prompt:]]
        out.append((float(tok_lp.sum()), int(tok_lp.numel())))
    return out


@torch.inference_mode()
def generate(model, proc, image, question, device, max_new=12):
    """max_new=12 is enough for a model that answers tersely and not for one that does not.
    Qwen2.5-VL-7B prefixes its answer ('The value for the bar labelled "18-24' ...), so at 12
    tokens the number never appears, the parse returns None, and the item scores WRONG. That
    understates model accuracy and therefore inflates every probe-minus-model gap -- 277/300 of
    the 7B's real-chart items were lost that way. The real families need a larger budget; the
    synthetic ones lose under 3% and are unaffected."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=[image], return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=proc.tokenizer.eos_token_id)
    return proc.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main(a):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = L.load(a.model, dev, a.load_4bit).eval()
    print(f"loaded {a.model}  VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    recs = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))]
    if a.base_only:
        recs = [r for r in recs if r["cf_of"] is None]
    if a.cf_only:
        # the mirror of --base-only, for a control set whose base half is already scored under
        # the canonical run. Re-scoring it produced identical answers on all 461 chart items in
        # every one of the five models, so it is redundant rather than a second estimator --
        # run/verify_protocol.py asserts that identity where both files exist.
        recs = [r for r in recs if r["cf_of"] is not None]
    if a.limit:
        per = {}
        keep = []
        for r in recs:
            per[r["family"]] = per.get(r["family"], 0) + 1
            if per[r["family"]] <= a.limit:
                keep.append(r)
        recs = keep
    print(f"{len(recs)} items from {a.data}", flush=True)

    grey = Image.new("RGB", (448, 448), (127, 127, 127))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "w")
    t0 = time.time()
    for i, r in enumerate(recs):
        img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
        space = r["answer_space"]

        gen_v = generate(model, proc, img,  r["question"], dev, a.max_new)
        if a.gen_only:
            rec = dict(id=r["id"], family=r["family"], arm=r["prior_arm"],
                       level=r["difficulty"].get("level"), hard=bool(r["difficulty"]["hard"]),
                       answer=r["answer"], chance=r["chance"], gen_vis=gen_v,
                       parse_vis=parse(r["family"], gen_v, r["answer_space"]))
            rec["gen_correct"] = rec["parse_vis"] == r["answer"]
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(recs)}  {el/(i+1):.2f}s/item", flush=True)
            continue
        gen_b = generate(model, proc, grey, r["question"], dev, a.max_new)
        sc_v = score_candidates(model, proc, img,  r["question"], space, dev)
        sc_b = score_candidates(model, proc, grey, r["question"], space, dev)

        lp_v = [s for s, n in sc_v]; lp_b = [s for s, n in sc_b]
        norm = lambda xs: [x - torch.logsumexp(torch.tensor(xs), 0).item() for x in xs]
        p_v, p_b = norm(lp_v), norm(lp_b)
        ti = space.index(r["answer"])

        rec = dict(id=r["id"], family=r["family"], arm=r["prior_arm"],
                   hard=bool(r["difficulty"]["hard"]), level=r["difficulty"].get("level"),
                   cf_of=r["cf_of"],
                   answer=r["answer"], chance=r["chance"],
                   gen_vis=gen_v, gen_blind=gen_b,
                   parse_vis=parse(r["family"], gen_v, space), parse_blind=parse(r["family"], gen_b, space),
                   score_vis=space[max(range(len(space)), key=lambda k: p_v[k])],
                   score_blind=space[max(range(len(space)), key=lambda k: p_b[k])],
                   logp_true_vis=p_v[ti], logp_true_blind=p_b[ti],
                   margin_true=p_v[ti] - p_b[ti])
        rec["gen_correct"] = rec["parse_vis"] == r["answer"]
        rec["gen_correct_blind"] = rec["parse_blind"] == r["answer"]
        rec["score_correct"] = rec["score_vis"] == r["answer"]
        rec["score_correct_blind"] = rec["score_blind"] == r["answer"]
        f.write(json.dumps(rec) + "\n"); f.flush()
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(recs)}  {el/(i+1):.2f}s/item  eta {(len(recs)-i-1)*el/(i+1)/60:.1f} min",
                  flush=True)
    f.close()
    print(f"wrote {a.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-new", type=int, default=12,
                   help="generation budget; verbose models need more than a terse answer's worth")
    p.add_argument("--load-4bit", action="store_true",
                   help="load in nf4; needed for 7B on 16 GB, and available on "
                        "smaller models so the scale axis holds precision constant")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/smoke")
    p.add_argument("--out", default="runs/pilot_3b.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--cf-only", action="store_true",
                   help="score only counterfactual records; the base half comes from the "
                        "canonical <tag>_gen.jsonl")
    p.add_argument("--gen-only", action="store_true", help="generation only; for difficulty calibration")
    main(p.parse_args())
