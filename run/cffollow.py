# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Counterfactual follow rate at the FINAL layer, from cached states. No GPU.

The published follow rates come from run/cfprobe.py, which reads the layer the serialised probe
was frozen at -- chosen on the selection split. Every other quantity in the locus verdict is a
final-layer quantity, so the gate mixed two estimators. Given run/cfcapture.py's counterfactual
states this can be recomputed at the layer the probe is read from.

Only the probe changes. The items, the true answers before and after the edit, and the model's
own generations are taken verbatim from runs/cfprobe_<tag>.json, so the two files differ in
exactly one thing and the comparison isolates the estimator.

The probe is layers.py's: standardise, 64 PCA components fitted on the training split, logistic
regression at C=0.5, fitted on the same 55% split and read at index -1.

Each record also carries whether its post-edit answer is inside the fitted probe's class support.
`predict` returns a member of `classes_` and nothing else, so an item whose counterfactual answer
was never a training label cannot follow the edit however well the representation encodes it --
the follow rate is then reporting the label space, not the readout. It is not a rare case: the
real counting family holds counts 2-5 and the edit adds an object, so every item at 5 is
unfollowable by construction, and the real chart family never contains the value 100 that raising
a bar can produce. Recorded per item rather than corrected here, because which denominator is
right depends on the claim being made, and run/canon.py is where that choice belongs.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the rare-class filter is a protocol constant, not a local choice: gen/build_chart.py
# uses it to decide which edit targets a probe will be able to emit, so a copy here that
# drifted would produce counterfactuals no probe could follow. One definition, imported.
from canon import MIN_CLASS, pairs_of, split

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]), spatial=lambda m: m["attribute"]["relation"], chart=lambda m: int(m["attribute"]["value"]), tracking=lambda m: int(m["attribute"]["end"]), glyph=lambda m: int(m["attribute"]["value"]))


def main(tag):
    st, cf = f"runs/states_{tag}.npz", f"runs/cfstates_{tag}.npz"
    ref = f"runs/cfprobe_{tag}.json"
    for f in (st, cf, ref):
        if not os.path.exists(f): raise SystemExit(f"missing {f}")
    V = np.load(st)["vis"].astype(np.float32)
    meta = json.load(open(st.replace(".npz", "_meta.json")))
    C = np.load(cf)["vis"].astype(np.float32)
    cmeta = json.load(open(cf.replace(".npz", "_meta.json")))
    cpos = {m["cf_of"]: i for i, m in enumerate(cmeta)}
    base = {r["id"]: r for r in json.load(open(ref))}

    out, report = [], []
    for fam in sorted(TARGET):
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == fam])
        if len(idx) == 0: continue
        y = np.array([TARGET[fam](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS], dtype=int)
        idx, y = idx[keep], y[keep]
        tr, _, te = split(y, groups=pairs_of(meta, idx), seed=0)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=min(64, len(tr) - 1), random_state=0), LogisticRegression(max_iter=1000, C=0.5))
        pipe.fit(V[idx[tr], -1], y[tr])
        support = {str(c) for c in pipe.classes_}

        ids = [meta[i]["id"] for i in idx[te]]
        have = [(k, i) for k, i in zip(ids, idx[te]) if k in cpos and k in base]
        if not have: continue
        p0 = pipe.predict(V[[i for _, i in have], -1])
        p1 = pipe.predict(C[[cpos[k] for k, _ in have], -1])
        for (k, _), a, b in zip(have, p0, p1):
            r = base[k]
            out.append(dict(id=k, family=fam, a0=r["a0"], a1=r["a1"], probe0=str(a), probe1=str(b), model0=r["model0"], model1=r["model1"], a0_in_support=str(r["a0"]) in support, a1_in_support=str(r["a1"]) in support))
        g = [r for r in out if r["family"] == fam]
        ok = [r for r in g if r["probe0"] == r["a0"]]
        fl = np.mean([r["probe1"] == r["a1"] for r in ok]) if ok else float("nan")
        oks = [r for r in ok if r["a1_in_support"]]
        fls = np.mean([r["probe1"] == r["a1"] for r in oks]) if oks else float("nan")
        og = [base[k] for k, _ in have]
        ook = [r for r in og if r["probe0"] == r["a0"]]
        ofl = np.mean([r["probe1"] == r["a1"] for r in ook]) if ook else float("nan")
        report.append((fam, len(g), len(ok), 100 * fl, len(oks), 100 * fls, len(ook), 100 * ofl))

    o = f"runs/cffollow_{tag}.json"
    json.dump(out, open(o, "w"), indent=1)
    print(f"{tag}")
    print(f"  {'family':10s}{'n':>4s}{'den':>6s}{'follow@final':>14s}"
          f"{'den':>6s}{'in-support':>12s}{'den':>6s}{'follow@selected':>17s}")
    for fam, n, d, f_, ds, fs, od, of in report:
        print(f"  {fam:10s}{n:4d}{d:6d}{f_:13.0f}%{ds:6d}{fs:11.0f}%{od:6d}{of:16.0f}%")
    print(f"  wrote {o}")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["realchart"]): main(t)
