"""Best-of-N self-consistency: the honest 'just spend more compute' competitor.

A router that costs k times the base model has to beat the simplest thing you can do with
k times the base model -- sample k answers and take the majority. If sampling recovers the
same accuracy, the locus machinery is decoration.

It also answers a question the branch grid cannot: how much of the base model's error is
sampling noise rather than a fixed deficit. If majority-vote over 8 samples moves accuracy
by a point, the errors are deterministic, and only an intervention that changes the
computation can move them.
"""
from __future__ import annotations
import argparse, json, os, time
from collections import Counter
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import branches as B


@torch.inference_mode()
def sample_k(model, proc, image, question, dev, k, temp, mx=10):
    enc = proc(text=[B.build_prompt(proc, question)], images=[image], return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=mx, do_sample=True, temperature=temp,
                         top_p=0.95, num_return_sequences=k,
                         pad_token_id=proc.tokenizer.eos_token_id)
    n = enc.input_ids.shape[1]
    return [proc.tokenizer.decode(o[n:], skip_special_tokens=True).strip() for o in out]


def main(a):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=dev, attn_implementation="eager").eval()
    P = np.load(a.probes, allow_pickle=True).item()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl")))}
    torch.manual_seed(a.seed)
    f = open(a.out, "w"); t0 = time.time(); n = 0
    for fam, pf in P.items():
        if a.families and fam not in a.families: continue
        ids = json.load(open(a.ids))[fam] if a.ids else pf["test_ids"]
        ids = ids[:a.limit] if a.limit else ids
        for iid in ids:
            r = recs[iid]; img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
            q = r["question"]
            if a.cot:
                q += (" Think step by step, briefly, then end with "
                      "'ANSWER: <your answer>'.")
            raw = sample_k(model, proc, img, q, dev, a.k, a.temp, mx=140 if a.cot else 10)
            if a.cot:
                raw = [t.split("ANSWER:")[-1] if "ANSWER:" in t else t[-40:] for t in raw]
            got = [B.parse_in_space(t, r["answer_space"]) for t in raw]
            f.write(json.dumps(dict(id=iid, family=fam, answer=r["answer"], samples=got)) + "\n")
            f.flush(); n += 1
            if n % 50 == 0: print(f"  {n}  {(time.time()-t0)/n:.2f}s/item", flush=True)
    f.close()

    rows = [json.loads(l) for l in open(a.out)]
    fams = sorted({r["family"] for r in rows})
    print(f"\nBest-of-N majority vote, temperature {a.temp}, cot={a.cot}, {len(rows)} items")
    div = np.mean([len(set(x for x in r["samples"] if x is not None)) for r in rows])
    print(f"mean distinct answers per item over {a.k} samples: {div:.2f}")
    print(f"{'family':10s}" + "".join(f"{'k='+str(k):>8s}" for k in (1, 2, 3, 5, 8)) + f"{'anyof8':>9s}")
    for fam in fams + ["ALL"]:
        g = [r for r in rows if fam == "ALL" or r["family"] == fam]
        line = f"{fam:10s}"
        for k in (1, 2, 3, 5, 8):
            acc = np.mean([vote(r["samples"][:k]) == r["answer"] for r in g])
            line += f"{100*acc:7.0f}%"
        anyk = np.mean([any(s == r["answer"] for s in r["samples"]) for r in g])
        line += f"{100*anyk:8.0f}%"
        print(line)
    print("\n'any of 8' is the sampling oracle: the ceiling for ANY selection rule over samples.")


def vote(ss):
    ss = [s for s in ss if s is not None]
    if not ss: return None
    return Counter(ss).most_common(1)[0][0]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--out", default="runs/bon_3b.jsonl")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default="")
    p.add_argument("--families", nargs="*", default=None,
                   help="restrict to these families; the rest of the file is left untouched")
    p.add_argument("--cot", action="store_true",
                   help="sample CoT chains instead of direct answers -- the literature's "
                        "self-consistency, and the only version with real answer diversity")
    main(p.parse_args())
