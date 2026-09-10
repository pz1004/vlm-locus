# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Is the verdict a property of the data or of one split? Re-derive it on many. No GPU.

Every number in this paper comes from one 55/20/25 split at random_state=0. The honest question
is not whether the accuracies wobble -- they will -- but whether the *verdict* does, and that is
not answered by resampling the probe alone: the permutation null, the blindfold and shuffled
controls, the model's own accuracy, the paired test and the counterfactual statistic are all
computed on the same split and all move together when it changes. So the whole rule is
re-derived per split, exactly as run/canon.py's locus() applies it.

Scope is the cells where the verdict decides something: the readout loci, and the degraded-glyph
controls that calibrate the false-positive rate. Expanding the model count would answer a
different and weaker question, which is the reviewer's point.

The estimator is run/nullcal.py's, condition for condition. One shortcut, and it is exact rather
than approximate: the scaler and the PCA are label-independent, so they are fitted once per
(cell, split) and only the logistic regression is refitted per permutation. Fitting the whole
pipeline per draw gives identical numbers for the cost of a hundredfold more work.

Permutations per split default to 500 rather than nullcal's 2000. The gate is p < 0.05 and the
attainable floor is 1/501, so the decision is unaffected; --nperm restores any value.
"""
from __future__ import annotations
import os
# Pin BLAS threads before numpy loads, for run/nullcal.py's two reasons. The fits here are tiny --
# 64 components, a few hundred rows -- so intra-fit threading is pure oversubscription, and the
# reduction order inside a fit must not depend on the worker count or the null stops being
# reproducible. Parallelism goes across permutation draws instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, warnings
import numpy as np
import joblib
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# The blindfolded states have far fewer distinct rows than items -- a family shares a handful of
# question templates -- so their training split is rank-deficient and PCA's trailing components
# have zero variance. That is the control behaving correctly (it prices a prior that does not vary
# with the image) and the fit is unaffected, so the ratio warning is silenced rather than chased.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon import (FOLLOW_MIN, NULL_ALPHA, G1_BLIND, G1_SHUF, G1_POS, GEN, wilson)

NJOBS = int(os.environ.get("VLM_LOCUS_NJOBS", 24))

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]),
              glyph=lambda m: int(m["attribute"]["value"]))

# the cells whose verdict decides something: the loci, and the controls that calibrate the rate
CELLS = [("3b", "chart"), ("3b", "spatial"), ("smolm", "spatial"),
         ("realchart_v2", "chart"), ("q3b4_real_chart_v2", "chart"),
         ("q7b_real_chart_v2", "chart"), ("ivl_real_chart_v2", "chart"),
         ("real", "glyph"), ("q3b4_real_3b", "glyph"), ("q7b_real_3b", "glyph"),
         ("ivl_real_3b", "glyph"), ("smol_real_3b", "glyph")]


def model_ok(tag):
    f = GEN.get(tag, f"runs/{tag}_gen.jsonl")
    if not os.path.exists(f):
        return {}
    out = {}
    for line in open(f):
        r = json.loads(line)
        out[r["id"]] = bool(r["ok"]["none"] if "ok" in r else r["gen_correct"])
    return out


def reduce(X, tr):
    """Standardise and project, fitted on the training rows only -- label-independent, so one fit
    per split serves the observed probe and every permutation of it."""
    sc = StandardScaler().fit(X[tr])
    pca = PCA(n_components=min(64, len(tr) - 1), random_state=0).fit(sc.transform(X[tr]))
    return lambda Z: pca.transform(sc.transform(Z))


def logprob_margin(Ztr, ytr, Zte, yte, Btr, Bte):
    """Per-item log p(true class), visual probe minus blindfolded probe -- G1's margin conditions."""
    def lp(Za, ya, Zb):
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(Za, ya)
        P, cls = clf.predict_proba(Zb), list(clf.classes_)
        return np.array([np.log(max(P[i, cls.index(v)], 1e-12)) if v in cls else np.log(1e-12)
                         for i, v in enumerate(yte)])
    return lp(Ztr, ytr, Zte) - lp(Btr, ytr, Bte)


def one(V, Bl, C, cpos, cfref, meta, idx, y, seed, nperm, mok):
    tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=seed,
                               stratify=y)
    _, te = train_test_split(rest, test_size=0.55, random_state=seed, stratify=y[rest])
    Xv, Xb = V[idx, -1], Bl[idx, -1]
    pv, pb = reduce(Xv, tr), reduce(Xb, tr)
    Ztr, Zte, Btr, Bte = pv(Xv[tr]), pv(Xv[te]), pb(Xb[tr]), pb(Xb[te])

    clf = LogisticRegression(max_iter=1000, C=0.5).fit(Ztr, y[tr])
    pred = clf.predict(Zte)
    av = float(np.mean(pred == y[te]))
    ab = float(LogisticRegression(max_iter=1000, C=0.5).fit(Btr, y[tr]).score(Bte, y[te]))
    ysh = np.random.default_rng(seed).permutation(y)
    ash = float(LogisticRegression(max_iter=1000, C=0.5)
                .fit(Ztr, ysh[tr]).score(Zte, ysh[te]))
    m = logprob_margin(Ztr, y[tr], Zte, y[te], Btr, Bte)
    g1 = bool(av - ab > G1_BLIND and av - ash > G1_SHUF
              and m.mean() > 0 and (m > 0).mean() > G1_POS)

    def draw(s):
        yy = np.random.default_rng(10_000 * seed + s).permutation(y[tr])
        return float(LogisticRegression(max_iter=1000, C=0.5).fit(Ztr, yy).score(Zte, y[te]))

    null = np.array(joblib.Parallel(n_jobs=NJOBS)(joblib.delayed(draw)(s)
                                                  for s in range(nperm)))
    p_null = float((1 + (null >= av).sum()) / (1 + nperm))

    ids = [meta[i]["id"] for i in idx[te]]
    mt = [mok[k] for k in ids if k in mok]
    macc = float(np.mean(mt)) if mt else float("nan")
    b = sum(p == g and not mok.get(k, False) for k, p, g in zip(ids, pred, y[te]))
    c = sum(mok.get(k, False) and p != g for k, p, g in zip(ids, pred, y[te]))
    p_paired = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0

    jn = jd = 0
    if C is not None:
        have = [(k, i) for k, i in zip(ids, idx[te]) if k in cpos and k in cfref]
        if have:
            p0 = clf.predict(pv(Xv[[list(idx).index(i) for _, i in have]]))
            p1 = clf.predict(pv(C[[cpos[k] for k, _ in have], -1]))
            jn = sum(str(a) == cfref[k]["a0"] and str(bb) == cfref[k]["a1"]
                     for (k, _), a, bb in zip(have, p0, p1))
            jd = len(have)
    lb = 100 * wilson(jn, jd)[0] if jd else float("nan")
    return dict(seed=seed, probe=av, model=macc, gap=100 * (av - macc), g1=g1, p_null=p_null,
                p_paired=p_paired, joint_num=jn, joint_den=jd, joint_lb=lb,
                readout=bool(g1 and p_null < NULL_ALPHA and p_paired < 0.05
                             and av > macc and lb > FOLLOW_MIN))


def main(a):
    out = {}
    print(f"{'cell':26s}{'S':>3s}{'readout':>9s}{'probe':>15s}{'p_null':>9s}"
          f"{'joint LB':>15s}{'G1':>6s}")
    for tag, fam in (a.cells or CELLS):
        st = f"runs/states_{tag}.npz"
        if not os.path.exists(st):
            continue
        z = np.load(st)
        V, Bl = z["vis"].astype(np.float32), z["blind"].astype(np.float32)
        meta = json.load(open(st.replace(".npz", "_meta.json")))
        cf = f"runs/cfstates_{tag}.npz"
        C = cpos = cfref = None
        if os.path.exists(cf):
            C = np.load(cf)["vis"].astype(np.float32)
            cm = json.load(open(cf.replace(".npz", "_meta.json")))
            cpos = {m["cf_of"]: i for i, m in enumerate(cm)}
            cfref = {r["id"]: r for r in json.load(open(f"runs/cfprobe_{tag}.json"))}
        mok = model_ok(tag)
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == fam])
        y = np.array([TARGET[fam](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8], dtype=int)
        idx, y = idx[keep], y[keep]
        rs = [one(V, Bl, C, cpos, cfref, meta, idx, y, s, a.nperm, mok)
              for s in range(a.splits)]
        out[f"{tag}/{fam}"] = rs
        pr = [100 * r["probe"] for r in rs]
        lbs = [r["joint_lb"] for r in rs if r["joint_lb"] == r["joint_lb"]]
        print(f"{tag + '/' + fam:26s}{len(rs):3d}"
              f"{sum(r['readout'] for r in rs):5d}/{len(rs):<3d}"
              f"{np.mean(pr):8.1f} [{min(pr):.0f}-{max(pr):.0f}]"
              f"{max(r['p_null'] for r in rs):9.4f}"
              f"{np.mean(lbs) if lbs else float('nan'):9.1f} [{min(lbs) if lbs else 0:.0f}-{max(lbs) if lbs else 0:.0f}]"
              f"{sum(r['g1'] for r in rs):4d}/{len(rs)}", flush=True)
    prev = {}
    if os.path.exists(a.out):
        prev = json.load(open(a.out))
    merged = {**prev, **out}
    json.dump(merged, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}  ({len(out)} cell(s) updated, {len(merged)} total)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="*", type=lambda s: tuple(s.split("/")),
                   help="tag/family pairs to run instead of the default decision-relevant set")
    p.add_argument("--out", default="runs/splits.json")
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--nperm", type=int, default=500)
    main(p.parse_args())
