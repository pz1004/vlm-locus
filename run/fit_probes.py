# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Fit and freeze the routing probes, and extract a steering direction per class.

The probe is standardise -> PCA -> multinomial logistic regression. For steering we need a
direction in the model's own residual space, so we push the class weight back through both
linear maps:

    logit_c(x) = w_c . PCA((x - mu) / sigma) + b_c
    d logit_c / dx = (components^T w_c) / sigma

which is the raw-space direction that most increases the probe's confidence in class c. It is
unit-normalised at use time and applied at a magnitude matched to the residual norm, so the
intervention is scale-free across layers and models.
"""
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

def main(states, out):
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32)
    g1 = json.load(open(states.replace("states_", "probe_g1_").replace(".npz", ".json")))
    store = {}
    for f, d in g1.items():
        l = d["layer"]
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        y = np.array([TARGET[f](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8])
        idx, y = idx[keep], y[keep]
        tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=0, stratify=y)
        _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(64, len(tr) - 1), random_state=0),
                             LogisticRegression(max_iter=1000, C=0.5))
        pipe.fit(V[idx[tr], l], y[tr])
        sc, pca, lr = pipe[0], pipe[1], pipe[2]
        # raw-space steering direction per class
        Dm = (pca.components_.T @ lr.coef_.T).T / sc.scale_          # [C, D]
        Dm = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
        # sklearn gives a binary LogisticRegression ONE coefficient row and decides on the sign,
        # but every consumer here reconstructs the probe as argmax over coef @ z + intercept,
        # which on a length-1 vector always returns class 0. Store the equivalent symmetric
        # two-row form so the stored probe means the same thing to sklearn and to the consumers;
        # softmax([-s/2, +s/2]) reproduces sklearn's predict_proba exactly.
        C, b = lr.coef_, lr.intercept_
        if C.shape[0] == 1 and len(lr.classes_) == 2:
            C, b = np.vstack([-C / 2, C / 2]), np.array([-b[0] / 2, b[0] / 2])
        store[f] = dict(layer=int(l), classes=[str(c) for c in lr.classes_],
                        mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                        pca_mean=pca.mean_.tolist(), components=pca.components_.tolist(),
                        coef=C.tolist(), intercept=b.tolist(),
                        directions=Dm.tolist(),
                        test_ids=[meta[i]["id"] for i in idx[te]],
                        train_ids=[meta[i]["id"] for i in idx[tr]],
                        test_acc=float(pipe.score(V[idx[te], l], y[te])))
        print(f"  {f:10s} layer {l:2d}  classes {len(lr.classes_):2d}  "
              f"test acc {store[f]['test_acc']:.3f}  |test|={len(te)}")
    # A stored probe is only useful if the consumers' reconstruction reproduces sklearn's own
    # prediction. That silently stopped being true for binary problems; check it rather than
    # trust it, on the same held-out split whose accuracy is reported above.
    for f, d in store.items():
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f and m["id"] in set(d["test_ids"])])
        C, b = np.array(d["coef"]), np.array(d["intercept"])
        got = []
        for i in idx:
            h = V[i, d["layer"]].astype(np.float64)
            z = ((h - np.array(d["mean"])) / np.array(d["scale"]) - np.array(d["pca_mean"])) \
                @ np.array(d["components"]).T
            got.append(d["classes"][int(np.argmax(C @ z + b))])
        acc = float(np.mean([g == str(meta[i]["answer"]) for g, i in zip(got, idx)]))
        if abs(acc - d["test_acc"]) > 0.02:
            raise SystemExit(f"! {f}: stored probe reconstructs to {acc:.3f} but sklearn scored "
                             f"{d['test_acc']:.3f} -- the serialised form does not match the model")
    print("  reconstruction check: stored probes reproduce sklearn's predictions")
    np.save(out, store, allow_pickle=True)
    print(f"wrote {out}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/states_3b.npz",
         sys.argv[2] if len(sys.argv) > 2 else "runs/probes_3b.npy")
