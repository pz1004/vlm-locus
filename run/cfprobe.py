# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Does the probe READ the attribute, or a correlate? The counterfactual control.

A linear probe answering held-out items at 98.6% (chart) while the model answers 34% is the
strongest result in this study, and it is exactly the kind of result the probing-methodology
literature says not to trust on accuracy alone. High probe accuracy is consistent with the
probe reading a correlate of the attribute rather than the attribute.

The generators were built as spec -> render precisely so this control would be exact. Each
counterfactual is a one-field edit to the spec that changes the probed attribute and nothing
else, re-rendered from the same seed:

  counting  drop one target                 count - 1
  spatial   swap the two referent boxes      relation flips
  chart     shift the queried bar by 10      value +/- 10
  tracking  drop the last swap               end position changes

If the probe reads the attribute, its prediction FOLLOWS the edit. If it reads a correlate,
it does not. The same measurement on the model's own answer is the readout claim made causal:
the state tracks the edit, the emitted answer does not.

Guard: the original spec is re-rendered and checked bit-identical to the stored image before
any counterfactual is trusted, and the counterfactual question is checked unchanged -- an edit
that also changes the question would confound the control.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gen"))
import generate as G
import branches as B
import loadm as L


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def build_cf(data, out, ids):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(data, "manifest.jsonl")))}
    rows, bad_render, bad_q, skipped = [], 0, 0, 0
    tmp = os.path.join(out, "_check.png")
    for iid in ids:
        r = recs[iid]; spec = r["spec"]; fam = r["family"]
        G.RENDER[fam](spec).save(tmp, optimize=True)           # guard 1: bit-identical re-render
        if sha(tmp) != sha(os.path.join(data, r["image"])): bad_render += 1; continue
        cs = G.CF[fam](spec)
        if cs is None: skipped += 1; continue
        lab = G.LABEL[fam](cs)
        if lab["question"] != r["question"]: bad_q += 1; continue   # guard 2: question unchanged
        p = os.path.join("images", iid + "_cf.png")
        G.RENDER[fam](cs).save(os.path.join(out, p), optimize=True)
        rows.append(dict(id=iid + "_cf", cf_of=iid, family=fam, image=p,
                         question=lab["question"], answer=lab["answer"],
                         answer_space=lab["answer_space"], attribute=lab["attribute"],
                         referents=lab["referents"], orig_answer=r["answer"]))
    os.remove(tmp)
    with open(os.path.join(out, "manifest.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"counterfactuals: {len(rows)} built, {skipped} not applicable, "
          f"{bad_render} failed the re-render check, {bad_q} changed the question")
    if bad_render: raise SystemExit("re-render mismatch: the spec does not reproduce the image")
    return rows


@torch.inference_mode()
def probe_and_answer(model, proc, img, question, dev, pf, max_new=32):
    enc = proc(text=[B.build_prompt(proc, question)], images=[img], return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_hidden_states=True,
                         return_dict_in_generate=True, pad_token_id=proc.tokenizer.eos_token_id)
    hs = out.hidden_states[0][pf["layer"]][0, -1].detach().float().cpu().numpy()
    txt = proc.tokenizer.decode(out.sequences[0, enc.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()
    z = ((hs - np.array(pf["mean"])) / np.array(pf["scale"]) - np.array(pf["pca_mean"])) \
        @ np.array(pf["components"]).T
    pv = softmax(np.array(pf["coef"]) @ z + np.array(pf["intercept"]))
    return pf["classes"][int(np.argmax(pv))], txt, float(pv.max())


def load_prebuilt(data, ids):
    """Real-image datasets ship their counterfactuals in the manifest: there is no spec to
    re-render from, so exactness was established at build time by gen/verify_real.py's pixel
    identity guards (flip, paste mask, glyph box) rather than by a re-render hash here."""
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(data, "manifest.jsonl")))}
    keep = set(ids)
    rows = [dict(id=r["id"], cf_of=r["cf_of"], family=r["family"], image=r["image"],
                 question=r["question"], answer=r["answer"], answer_space=r["answer_space"],
                 attribute=r["attribute"], referents=r.get("referents", []),
                 orig_answer=recs[r["cf_of"]]["answer"])
            for r in recs.values() if r["cf_of"] in keep]
    bad = [r["id"] for r in rows if r["question"] != recs[r["cf_of"]]["question"]]
    if bad: raise SystemExit(f"counterfactual changed the question: {bad[:3]}")
    same = [r["id"] for r in rows if r["answer"] == r["orig_answer"]]
    if same: raise SystemExit(f"counterfactual did not change the answer: {same[:3]}")
    print(f"counterfactuals: {len(rows)} prebuilt, covering "
          f"{len({r['cf_of'] for r in rows})}/{len(keep)} test items")
    return rows


def main(a):
    P = np.load(a.probes, allow_pickle=True).item()
    ids = [i for f in P if not a.families or f in a.families
           for i in P[f]["test_ids"]]
    cf = load_prebuilt(a.data, ids) if a.prebuilt else build_cf(a.data, a.out, ids)
    cf_root = a.data if a.prebuilt else a.out
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = L.load(a.model, dev, a.load_4bit).eval()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl")))}
    res = []
    for k, c in enumerate(cf):
        pf = P[c["family"]]
        o = recs[c["cf_of"]]
        io = Image.open(os.path.join(a.data, o["image"])).convert("RGB")
        ic = Image.open(os.path.join(cf_root, c["image"])).convert("RGB")
        p0, t0, _ = probe_and_answer(model, proc, io, o["question"], dev, pf)
        p1, t1, _ = probe_and_answer(model, proc, ic, c["question"], dev, pf)
        res.append(dict(id=c["cf_of"], family=c["family"],
                        a0=o["answer"], a1=c["answer"],
                        probe0=p0, probe1=p1,
                        model0=B.parse_in_space(t0, o["answer_space"]),
                        model1=B.parse_in_space(t1, c["answer_space"])))
        if (k + 1) % 60 == 0: print(f"  {k+1}/{len(cf)}", flush=True)
    json.dump(res, open(a.res, "w"), indent=1)
    report(res)


def report(res):
    fams = sorted({r["family"] for r in res})
    print(f"\ncounterfactual control, {len(res)} paired items\n")
    print(f"{'family':10s}{'n':>4s}   {'--- probe ---':>22s}   {'--- model ---':>22s}")
    print(f"{'':10s}{'':4s}{'acc0':>8s}{'acc1':>7s}{'follow':>8s}"
          f"{'acc0':>9s}{'acc1':>7s}{'follow':>8s}")
    for f in fams + ["ALL"]:
        g = [r for r in res if f == "ALL" or r["family"] == f]
        # follow = among items the reader got right BEFORE the edit, how often it now gives
        # the new answer. Conditioning on being right before separates tracking the edit from
        # being wrong in a new way.
        def stat(pre, post):
            a0 = np.mean([r[pre] == r["a0"] for r in g])
            a1 = np.mean([r[post] == r["a1"] for r in g])
            ok = [r for r in g if r[pre] == r["a0"]]
            fl = np.mean([r[post] == r["a1"] for r in ok]) if ok else float("nan")
            return a0, a1, fl
        pa0, pa1, pf_ = stat("probe0", "probe1")
        ma0, ma1, mf = stat("model0", "model1")
        print(f"{f:10s}{len(g):4d}{100*pa0:7.0f}%{100*pa1:6.0f}%{100*pf_:7.0f}%"
              f"{100*ma0:8.0f}%{100*ma1:6.0f}%{100*mf:7.0f}%")
    print("\nacc0 = accuracy on the original, acc1 = on the counterfactual, follow = of the items")
    print("it answered correctly before the edit, the share it answers with the NEW value after.")
    print("A reader exploiting a correlate scores high on acc0 and low on follow.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-4bit", action="store_true",
                   help="load in nf4; needed for 7B on 16 GB, and available on "
                        "smaller models so the scale axis holds precision constant")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--out", default="data/cf_3b")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--res", default="runs/cfprobe_3b.json")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--prebuilt", action="store_true",
                   help="take counterfactuals from the dataset manifest instead of re-rendering "
                        "them from a spec; required for the real-image families")
    main(p.parse_args())
