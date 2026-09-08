"""Capture hidden states for the *counterfactual* images, at every layer.

run/cfprobe.py computes follow rates by running the edited images through the model and reading
one layer -- `out.hidden_states[0][pf["layer"]]`, the layer the serialised probe was frozen at,
chosen on the selection split. Nothing is kept, so a follow rate at any other layer needs the
GPU again. That is why the published follow rates use a different estimator from every other
number in the paper, including the probe accuracy printed beside them.

This fixes it once. One plain forward pass per counterfactual image, all layers kept, so
run/cffollow.py can compute the follow rate at the final layer -- or any layer -- for free
thereafter. No blindfold pass: a counterfactual's blindfolded state is its original's, since the
image is replaced by the same grey field and the question is unchanged by construction.

Item set: the counterfactuals of the held-out items, which is what cfprobe.py scored, so the two
are directly comparable.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadm as L
from capture import last_token_states


def main(a):
    dev = "cuda"
    cf_root = a.cf_root or a.data
    P = np.load(a.probes, allow_pickle=True).item()
    keep = {i for f in P for i in P[f]["test_ids"]}
    cfs = [json.loads(l) for l in open(os.path.join(cf_root, "manifest.jsonl"))]
    cfs = [r for r in cfs if r["cf_of"] in keep]
    if not cfs: raise SystemExit(f"no counterfactuals in {cf_root} for {a.probes}")
    cfs.sort(key=lambda r: r["id"])
    print(f"{len(cfs)} counterfactuals of held-out items from {cf_root}")

    proc = AutoProcessor.from_pretrained(a.model)
    model = L.load(a.model, dev, a.load_4bit).eval()      # eager, as capture.py pins it
    V, meta, t0 = None, [], time.time()
    for i, r in enumerate(cfs):
        img = Image.open(os.path.join(cf_root, r["image"])).convert("RGB")
        h = last_token_states(model, proc, img, r["question"], dev)
        if V is None:
            V = np.zeros((len(cfs), *h.shape), np.float16)
            print(f"  states {h.shape} (layers x dim); buffer {V.nbytes/2**20:.0f} MB", flush=True)
        V[i] = h
        meta.append(dict(id=r["id"], cf_of=r["cf_of"], family=r["family"],
                         answer=r["answer"], answer_space=r["answer_space"]))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cfs)}  {el/(i+1):.2f}s/item  "
                  f"eta {(len(cfs)-i-1)*el/(i+1)/60:.1f} min", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, vis=V)
    json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"))
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/2**20:.0f} MB) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--data", required=True, help="dataset dir holding the ORIGINAL manifest")
    p.add_argument("--cf-root", default=None,
                   help="dir holding the counterfactual manifest and images (default: --data)")
    p.add_argument("--probes", required=True, help="runs/probes_<tag>.npy, for the held-out ids")
    p.add_argument("--out", required=True)
    main(p.parse_args())
