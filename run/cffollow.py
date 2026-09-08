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
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]),
              glyph=lambda m: int(m["attribute"]["value"]))


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
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8], dtype=int)
        idx, y = idx[keep], y[keep]
        tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45,
                                    random_state=0, stratify=y)
        _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(64, len(tr) - 1), random_state=0),
                             LogisticRegression(max_iter=1000, C=0.5))
        pipe.fit(V[idx[tr], -1], y[tr])

        ids = [meta[i]["id"] for i in idx[te]]
        have = [(k, i) for k, i in zip(ids, idx[te]) if k in cpos and k in base]
        if not have: continue
        p0 = pipe.predict(V[[i for _, i in have], -1])
        p1 = pipe.predict(C[[cpos[k] for k, _ in have], -1])
        for (k, _), a, b in zip(have, p0, p1):
            r = base[k]
            out.append(dict(id=k, family=fam, a0=r["a0"], a1=r["a1"],
                            probe0=str(a), probe1=str(b),
                            model0=r["model0"], model1=r["model1"]))
        g = [r for r in out if r["family"] == fam]
        ok = [r for r in g if r["probe0"] == r["a0"]]
        fl = np.mean([r["probe1"] == r["a1"] for r in ok]) if ok else float("nan")
        og = [base[k] for k, _ in have]
        ook = [r for r in og if r["probe0"] == r["a0"]]
        ofl = np.mean([r["probe1"] == r["a1"] for r in ook]) if ook else float("nan")
        report.append((fam, len(g), len(ok), 100 * fl, len(ook), 100 * ofl))

    o = f"runs/cffollow_{tag}.json"
    json.dump(out, open(o, "w"), indent=1)
    print(f"{tag}")
    print(f"  {'family':10s}{'n':>4s}{'den':>6s}{'follow@final':>14s}"
          f"{'den':>6s}{'follow@selected':>17s}{'shift':>8s}")
    for fam, n, d, f_, od, of in report:
        print(f"  {fam:10s}{n:4d}{d:6d}{f_:13.0f}%{od:6d}{of:16.0f}%{f_-of:+8.0f}")
    print(f"  wrote {o}")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["realchart"]): main(t)
