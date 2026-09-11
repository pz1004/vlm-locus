# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""The label-preserving control: does the probe's answer move when the answer does not?

Every other counterfactual in this study changes the attribute, so following one shows the probe
responds to something the edit did -- not that it responds to the attribute rather than to a cue
that travels with it. gen/build_chart_sham.py raises a bar the question does not ask about, by the
same amount as that item's real edit. The queried bar is untouched and the answer is unchanged, so
a reader of the attribute must return the same answer and a reader of the scene may not.

This is deliberately not run through run/cffollow.py. That estimator asks "did the answer track
the edit", and its guard rejects an edit that does not change the answer -- correctly, because on
a sham the quantity of interest is the opposite one. Reported here as stability: the share of
items whose prediction is unchanged. The probe and the model are scored on the same items.

The probe is the canonical one, refitted exactly as run/canonpred.py does, so "the probe" means
the same estimator as everywhere else in the protocol.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon import MIN_CLASS, GEN, pairs_of, split, wilson
from layers import TARGET

FAM = "chart"


def one(tag):
    st = f"runs/states_{tag}.npz"
    cf = f"runs/cfstates_{tag}_sham.npz"
    gen = f"runs/{tag}_sham_gen.jsonl"
    if not all(os.path.exists(p) for p in (st, cf, gen)):
        return None
    npz = np.load(st)
    meta = json.load(open(st.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32)
    L = V.shape[1]
    idx = np.array([i for i, m in enumerate(meta) if m["family"] == FAM])
    y = np.array([TARGET[FAM](meta[i]) for i in idx])
    keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS], dtype=int)
    idx, y = idx[keep], y[keep]
    tr, _, te = split(y, groups=pairs_of(meta, idx), seed=0)
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=min(64, len(tr) - 1), random_state=0),
                         LogisticRegression(max_iter=1000, C=0.5)).fit(V[idx[tr], L - 1], y[tr])

    C = np.load(cf)["vis"].astype(np.float32)
    cm = json.load(open(cf.replace(".npz", "_meta.json")))
    pos = {m["cf_of"]: i for i, m in enumerate(cm)}
    ids = [meta[i]["id"] for i in idx[te]]
    have = [(k, p) for k, p in zip(ids, idx[te]) if k in pos]
    if not have:
        return None
    p0 = pipe.predict(V[[p for _, p in have], L - 1])
    p1 = pipe.predict(C[[pos[k] for k, _ in have], L - 1])

    G = {}
    for line in open(gen):
        r = json.loads(line)
        G[r["id"]] = (r.get("parse_vis") if r.get("parse_vis") is not None else r.get("gen_vis"))
    gold = {meta[i]["id"]: str(TARGET[FAM](meta[i])) for i in idx}

    rows = []
    for (k, _), a, b in zip(have, p0, p1):
        sid = f"{k}_sham"
        rows.append(dict(id=k, gold=gold[k], probe0=str(a), probe1=str(b),
                         model0=None if k not in G else str(G[k]),
                         model1=None if sid not in G else str(G[sid])))

    def stab(pre, post, cond_correct):
        g = [r for r in rows if r[pre] is not None and r[post] is not None]
        if cond_correct:
            g = [r for r in g if r[pre] == r["gold"]]
        n = sum(r[pre] == r[post] for r in g)
        return dict(v=float(n / len(g)) if g else float("nan"), num=n, den=len(g),
                    ci=wilson(n, len(g)) if g else [float("nan")] * 2)

    return dict(tag=tag, n=len(rows),
                probe_stable=stab("probe0", "probe1", False),
                probe_stable_correct=stab("probe0", "probe1", True),
                model_stable=stab("model0", "model1", False),
                model_stable_correct=stab("model0", "model1", True),
                rows=rows)


def main(a):
    out = {}
    print(f"{'tag':26s}{'n':>5s}{'probe stable':>16s}{'LB':>7s}"
          f"{'model stable':>16s}{'LB':>7s}")
    for tag in a.tags:
        r = one(tag)
        if r is None:
            print(f"{tag:26s}  (artefacts missing)")
            continue
        out[tag] = r
        ps, ms = r["probe_stable_correct"], r["model_stable_correct"]
        pf = "%d/%d" % (ps["num"], ps["den"])
        mf = "%d/%d" % (ms["num"], ms["den"])
        print(f"{tag:26s}{r['n']:5d}{pf:>10s}{100*ps['v']:5.0f}%{100*ps['ci'][0]:7.1f}"
              f"{mf:>10s}{100*ms['v']:5.0f}%{100*ms['ci'][0]:7.1f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}  ({len(out)} cell(s))")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="*", default=["realchart_v2", "q3b4_real_chart_v2",
                                                 "q7b_real_chart_v2", "ivl_real_chart_v2",
                                                 "smol_real_chart_v2"])
    p.add_argument("--out", default="runs/sham.json")
    main(p.parse_args())
