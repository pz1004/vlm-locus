# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Does LRC survive without labels?

Three feature sets, priced by what they cost to compute:

  gen     mean log-prob of the emitted answer tokens. Rides on the base pass -- FREE.
  vis     + constrained answer-space distribution (p_max, entropy, top-2 gap, agreement with
          the free-form answer). Costs |A| candidate forwards, batchable to ~1.
  vis+bl  + the blindfold contrast (marg_self, blind_max): the plan's evidence margin computed
          on the DECODER rather than the probe. Costs 2|A|, batchable to ~2.

  probe   the labelled-tier features, for reference -- what LRC uses today.

The library is restricted to what each tier can actually reach. B-steer and B-read need the
probe, so they are excluded from every label-free row. B-prior is free in the vis+bl rows
because it IS that computation: the router and the branch share it.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import binomtest

LIB = ["none", "prior", "look", "attn", "cot"]          # label-free library
SETS = {"gen": ["gen_lp"],
        "vis": ["gen_lp", "p_max", "ent", "top2", "agree"],
        "vis+bl": ["gen_lp", "p_max", "ent", "top2", "agree", "marg_self", "blind_max",
                   "prior_gap"]}


def softmax(x):
    x = np.asarray(x, float); e = np.exp(x - x.max()); return e / e.sum()


def feats(u):
    lv, lb = np.array(u["lv"], float), np.array(u["lb"], float)
    pv, pb = softmax(lv), softmax(lb)
    o = np.argsort(-lv)
    k = int(np.argmax(pv)); sp = u["space"]
    j = sp.index(u["gen"]) if u["gen"] in sp else None
    return dict(gen_lp=u["gen_lp"], p_max=float(pv.max()), ent=float(-(pv * np.log(pv + 1e-12)).sum()),
                top2=float(lv[o[0]] - lv[o[1]]) if len(o) > 1 else 10.0,
                agree=float(sp[k] == u["gen"]),
                marg_self=float(lv[j] - lb[j]) if j is not None else -12.0,
                blind_max=float(pb.max()),
                prior_gap=float(pv.max() - pb.max()))


def load(unsup_path, branch_path, with_probe=False):
    U = {json.loads(l)["id"]: json.loads(l) for l in open(unsup_path)}
    out = []
    for line in open(branch_path):
        r = json.loads(line)
        if r["id"] not in U: continue
        f = feats(U[r["id"]])
        out.append(dict(id=r["id"], family=r["family"], ok=r["ok"], f=f))
    return out


def probe_feats(branch_path):
    """The labelled-tier features, rebuilt exactly as run/tiered.py builds them."""
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz"); meta = json.load(open("runs/states_3b_meta.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V, Bl = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
    out = {}
    for line in open(branch_path):
        r = json.loads(line); pf = P[r["family"]]; l = pf["layer"]
        mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
        pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
        coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
        cls = pf["classes"]; i = pos[r["id"]]
        pv = softmax(coef @ (((V[i, l] - mean) / scale - pm) @ comp.T) + inter)
        pb = softmax(coef @ (((Bl[i, l] - mean) / scale - pm) @ comp.T) + inter)
        k = int(np.argmax(pv)); ans = r["out"]["none"]
        j = cls.index(str(ans)) if (ans is not None and str(ans) in cls) else None
        out[r["id"]] = dict(p_ans=float(pv[j]) if j is not None else 0.0,
                            gap=float(pv[k]) - (float(pv[j]) if j is not None else 0.0),
                            sup=float(np.log(max(pv[j], 1e-12)) - np.log(max(pb[j], 1e-12)))
                                if j is not None else -12.0,
                            prior=float(pb.max()), conf=float(pv[k]),
                            margin=float(np.log(max(pv[k], 1e-12)) - np.log(max(pb[k], 1e-12))))
    return out


def route(C, T, keys, fams, lib, cost):
    Xc = np.array([[r["f"][k] for k in keys] + [r["family"] == f for f in fams] for r in C], float)
    Xt = np.array([[r["f"][k] for k in keys] + [r["family"] == f for f in fams] for r in T], float)
    p = np.zeros((len(T), len(lib)))
    for bi, b in enumerate(lib):
        y = np.array([r["ok"][b] for r in C], int)
        if y.min() == y.max(): p[:, bi] = y.max(); continue
        m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=0).fit(Xc, y)
        p[:, bi] = m.predict_proba(Xt)[:, list(m.classes_).index(1)]
    picks = [lib[int(np.argmax(p[i] - 0.0 * np.array([cost[r["family"]][b] for b in lib])))]
             for i, r in enumerate(T)]
    return picks


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    return (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def main():
    C = load("runs/unsup_calib_3b.jsonl", "runs/branches6_calib.jsonl")
    T = load("runs/unsup_test_3b.jsonl", "runs/branches6_test.jsonl")
    pfC, pfT = probe_feats("runs/branches6_calib.jsonl"), probe_feats("runs/branches6_test.jsonl")
    for r in C: r["pf"] = pfC[r["id"]]
    for r in T: r["pf"] = pfT[r["id"]]
    fams = sorted({r["family"] for r in T})
    cost = {f: {b: 1.0 for b in LIB} for f in fams}       # accuracy first; cost priced below
    print(f"calibration {len(C)}   test {len(T)}\n")

    base = np.mean([r["ok"]["none"] for r in T])
    bestfix = max(LIB, key=lambda b: np.mean([r["ok"][b] for r in T]))
    bacc = np.mean([r["ok"][bestfix] for r in T])
    print(f"library (label-free): {LIB}")
    print(f"base {100*base:.1f}%   best fixed branch: {bestfix} {100*bacc:.1f}%   "
          f"oracle {100*np.mean([any(r['ok'][b] for b in LIB) for r in T]):.1f}%\n")

    print(f"{'routing signal':16s}{'needs':10s}" + "".join(f"{f:>10s}" for f in fams)
          + f"{'ALL':>8s}{'vs base':>9s}{'vs best':>9s}")
    rows = []
    for name, keys in SETS.items():
        picks = route(C, T, keys, fams, LIB, cost)
        per = [np.mean([r["ok"][b] for r, b in zip(T, picks) if r["family"] == f]) for f in fams]
        allacc = np.mean([r["ok"][b] for r, b in zip(T, picks)])
        p1 = mcnemar([r["ok"][b] for r, b in zip(T, picks)], [r["ok"]["none"] for r in T])
        p2 = mcnemar([r["ok"][b] for r, b in zip(T, picks)], [r["ok"][bestfix] for r in T])
        print(f"{name:16s}{'no labels':10s}" + "".join(f"{100*a:9.0f}%" for a in per)
              + f"{100*allacc:7.1f}%{p1:9.2g}{p2:9.2g}")
        rows.append((name, float(allacc), float(p1), float(p2)))
    # the labelled-tier router on the same restricted library, for reference
    for r in C: r["f"] = dict(r["f"], **r["pf"])
    for r in T: r["f"] = dict(r["f"], **r["pf"])
    keys = ["p_ans", "gap", "sup", "prior", "conf", "margin"]
    picks = route(C, T, keys, fams, LIB, cost)
    per = [np.mean([r["ok"][b] for r, b in zip(T, picks) if r["family"] == f]) for f in fams]
    allacc = np.mean([r["ok"][b] for r, b in zip(T, picks)])
    p1 = mcnemar([r["ok"][b] for r, b in zip(T, picks)], [r["ok"]["none"] for r in T])
    p2 = mcnemar([r["ok"][b] for r, b in zip(T, picks)], [r["ok"][bestfix] for r in T])
    print(f"{'probe':16s}{'labels':10s}" + "".join(f"{100*a:9.0f}%" for a in per)
          + f"{100*allacc:7.1f}%{p1:9.2g}{p2:9.2g}")
    # B-read: emit the probe's own argmax as the answer. Needs the labelled tier.
    read = {json.loads(l)["id"]: (str(json.loads(l)["probe_pred"]) == str(json.loads(l)["answer"]))
            for l in open("runs/branches6_test.jsonl")}
    keep = [r for r in T if r["id"] in read]
    print(f"\n{'B-read':16s}{'labels':10s}"
          + "".join(f"{100*np.mean([read[r['id']] for r in keep if r['family']==f]):9.0f}%"
                    for f in fams)
          + f"{100*np.mean([read[r['id']] for r in keep]):7.1f}%"
          + f"{mcnemar([read[r['id']] for r in keep], [r['ok']['none'] for r in keep]):9.2g}"
          + f"{mcnemar([read[r['id']] for r in keep], [r['ok'][bestfix] for r in keep]):9.2g}")
    json.dump(rows, open("runs/unsup_router_3b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
