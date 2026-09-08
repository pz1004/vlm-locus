# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Every method on the same 298 held-out items, paired.

One table, one test set, one parser. Rows are grouped by what they need: nothing, boxes, or a
labelled calibration set. Comparing across those groups is only meaningful with the group
stated, which is why it is a column and not a footnote.

`read` is the method under test: take the frozen model's residual stream at the last prompt
token, apply the linear probe fitted on the calibration split, and emit its argmax. It never
runs the decoder for the answer, so it costs one forward pass -- the same pass the base model
already needs.

McNemar is exact-binomial on discordant pairs. `win` counts families where the row beats the
best row that needs strictly less supervision, which is the comparison a reader actually cares
about: is the extra supervision buying anything.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
import numpy as np
from scipy.stats import binomtest

TIER = dict(base="none", steer="labels", prior="none", look="boxes", attn="boxes", cot="none")


def vote(samples):
    s = [x for x in samples if x is not None]
    return Counter(s).most_common(1)[0][0] if s else None


def softmax(x):
    x = np.asarray(x, float); e = np.exp(x - x.max()); return e / e.sum()


def probe_dist(ids):
    """P(class | visual state) at the probe layer, for every test item."""
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz"); meta = json.load(open("runs/states_3b_meta.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V = npz["vis"].astype(np.float32)
    fam = {m["id"]: m["family"] for m in meta}
    out = {}
    for iid in ids:
        pf = P[fam[iid]]; i = pos[iid]
        z = ((V[i, pf["layer"]] - np.array(pf["mean"])) / np.array(pf["scale"])
             - np.array(pf["pca_mean"])) @ np.array(pf["components"]).T
        out[iid] = (pf["classes"], softmax(np.array(pf["coef"]) @ z + np.array(pf["intercept"])))
    return out


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    if n01 + n10 == 0: return 1.0, n01, n10
    return binomtest(n01, n01 + n10, 0.5).pvalue, n01, n10


def main():
    R = [json.loads(l) for l in open("runs/branches6_test.jsonl")]
    ids = [r["id"] for r in R]
    gold = {r["id"]: str(r["answer"]) for r in R}
    fam = {r["id"]: r["family"] for r in R}
    fams = sorted(set(fam.values()))
    M, tier = {}, {}

    for b, t in TIER.items():
        key = "none" if b == "base" else b
        M[b] = {r["id"]: bool(r["ok"][key]) for r in R}; tier[b] = t
    M["read"] = {r["id"]: str(r["probe_pred"]) == gold[r["id"]] for r in R}; tier["read"] = "labels"

    for tag, name in [("bon_3b", "bon8@T0.7"), ("bon_t1_3b", "bon8@T1.0"), ("boncot_3b", "cot-sc@5")]:
        p = f"runs/{tag}.jsonl"
        if not os.path.exists(p): continue
        S = {r["id"]: r["samples"] for r in (json.loads(l) for l in open(p))}
        if set(S) != set(ids):
            print(f"! {name}: item set differs from the test split ({len(set(S)&set(ids))}/{len(ids)}"
                  f" shared) -- skipped", file=sys.stderr); continue
        M[name] = {i: vote(S[i]) == gold[i] for i in ids}; tier[name] = "none"
        M[name + " oracle"] = {i: gold[i] in S[i] for i in ids}; tier[name + " oracle"] = "GOLD"
        if tag == "bon_3b":
            D = probe_dist(ids)
            def pick(i):
                cls, p = D[i]
                c = [x for x in S[i] if x is not None]
                if not c: return None
                return max(c, key=lambda v: p[cls.index(v)] if v in cls else -1.0)
            M["rerank(probe)"] = {i: pick(i) == gold[i] for i in ids}; tier["rerank(probe)"] = "labels"

    # every LoRA budget that has been run: runs/lora_items_<tag>-ep<n>.json. The best epoch per
    # tag is kept, which is oracle model selection and so generous to the competitor.
    import glob, re as _re
    byt = {}
    for f_ in glob.glob("runs/lora_items_*-ep*.json"):
        m = _re.match(r".*lora_items_(.+)-ep(\d+)\.json", f_)
        if not m or m.group(1) == "base": continue
        L = json.load(open(f_))
        acc = np.mean([L[i]["ok"] for i in ids if i in L])
        if acc > byt.get(m.group(1), (-1, None))[0]:
            byt[m.group(1)] = (acc, {i: bool(L[i]["ok"]) for i in ids if i in L})
    NLAB = {"lora": 165, "lora20": 20, "lora40": 40}
    for t, (_, d) in byt.items():
        M[f"{t}({NLAB.get(t, '?')}/fam)"] = d; tier[f"{t}({NLAB.get(t, '?')}/fam)"] = "labels"

    if os.path.exists("runs/readout_3b.json"):
        RO = json.load(open("runs/readout_3b.json"))["per_item"]
        if "linear@final" in RO:
            M["read@final"] = {i: bool(v) for i, v in RO["linear@final"].items() if i in set(ids)}
            tier["read@final"] = "labels"

    order = sorted(M, key=lambda k: -np.mean(list(M[k].values())))
    print(f"{'method':16s}{'needs':8s}" + "".join(f"{f:>10s}" for f in fams)
          + f"{'ALL':>8s}{'vs base':>9s}{'vs read':>9s}")
    for k in order:
        v = M[k]; kid = [i for i in ids if i in v]
        per = [np.mean([v[i] for i in kid if fam[i] == f]) for f in fams]
        allv = np.mean([v[i] for i in kid])
        p1, _, _ = mcnemar([v[i] for i in kid], [M["base"][i] for i in kid])
        p2, _, _ = mcnemar([v[i] for i in kid], [M["read"][i] for i in kid])
        star = "  <-- ours" if k == "read" else ""
        print(f"{k:16s}{tier[k]:8s}" + "".join(f"{100*a:9.0f}%" for a in per)
              + f"{100*allv:7.1f}%{p1:9.2g}{p2:9.2g}{star}")

    print("\nGOLD rows use the gold answer to select and are upper bounds, not methods.")
    best_free = max([k for k in M if tier[k] == "none"], key=lambda k: np.mean(list(M[k].values())))
    w = [f for f in fams if np.mean([M["read"][i] for i in ids if fam[i] == f])
         > np.mean([M[best_free][i] for i in ids if fam[i] == f])]
    print(f"read beats the best supervision-free method ({best_free}) in {len(w)}/{len(fams)} "
          f"families: {', '.join(w) if w else 'none'}")
    for lk in sorted([k for k in M if k.startswith("lora")]):
        w = [f for f in fams if np.mean([M["read"][i] for i in ids if fam[i] == f])
             > np.mean([M[lk][i] for i in ids if fam[i] == f])]
        p, n01, n10 = mcnemar([M["read"][i] for i in ids], [M[lk][i] for i in ids])
        print(f"read beats {lk} (same labels) in {len(w)}/{len(fams)} families: "
              f"{', '.join(w) if w else 'none'}   McNemar p={p:.2g} ({n01} vs {n10} discordant)")
    json.dump({k: {i: bool(v) for i, v in M[k].items()} for k in M},
              open("runs/compare_items.json", "w"))
    print("\nwrote runs/compare_items.json")


if __name__ == "__main__":
    main()
