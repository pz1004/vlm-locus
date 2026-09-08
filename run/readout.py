# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""If the answer is in the state, how well can it be read?

B-read used a linear probe because the probe's job in the original design was *diagnosis* --
linearity was the point, since a nonlinear probe can manufacture information that the model's
own linear head could never use. As a diagnostic that constraint is essential and it stays in
place for every claim in sections 1-4.

As a *method* the constraint is arbitrary. This file asks what the same frozen states, the same
labels and the same 1.01x inference cost yield when the readout is allowed to be better:

  linear@best   the diagnostic probe, at the layer chosen on the selection split  (= B-read)
  linear@final  the same at the last layer, no layer search
  concat        three layers concatenated -- late, mid and final
  mlp           one hidden layer on the final state
  mlp@concat    one hidden layer on the concatenated states
  fuse          linear probe combined with the model's own answer-space distribution, one
                mixing weight fitted on the calibration split

Everything is fitted on the probe's own train split and scored on the same held-out items as
every other table. `fuse` needs the model's constrained logits, which exist for the calibration
and test splits only, so its mixing weight is fitted on calibration -- a split disjoint from
both the probe's training data and the test set.
"""
from __future__ import annotations
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]))


def lin(n):
    return make_pipeline(StandardScaler(), PCA(n_components=min(64, n - 1), random_state=0),
                         LogisticRegression(max_iter=1000, C=0.5))


def mlp(n):
    return make_pipeline(StandardScaler(), PCA(n_components=min(64, n - 1), random_state=0),
                         MLPClassifier(hidden_layer_sizes=(128,), max_iter=2000, alpha=1e-2,
                                       random_state=0))


def softmax(x):
    x = np.asarray(x, float); e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    return binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


def main():
    npz = np.load("runs/states_3b.npz"); meta = json.load(open("runs/states_3b_meta.json"))
    V = npz["vis"].astype(np.float32)
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    pos = {m["id"]: i for i, m in enumerate(meta)}
    L = V.shape[1] - 1
    calib = json.load(open("runs/calib_ids.json"))
    U = {}
    for f in ["calib", "test"]:
        for line in open(f"runs/unsup_{f}_3b.jsonl"):
            r = json.loads(line); U[r["id"]] = r

    lora = None
    try: lora = json.load(open("runs/lora_items_lora-ep2.json"))
    except FileNotFoundError: pass

    fams = sorted(TARGET); res = {}; per_item = {}
    for f in fams:
        pf = P[f]; tr = pf["train_ids"]; te = pf["test_ids"]; ca = calib[f]
        y = {m["id"]: str(TARGET[f](m)) for m in meta if m["family"] == f}
        lay = pf["layer"]
        LAYERS = sorted({lay, max(0, L - 8), L})

        def X(ids, layers): return np.concatenate([V[[pos[i] for i in ids], l] for l in layers], 1)
        ytr = np.array([y[i] for i in tr]); yte = np.array([y[i] for i in te])
        out = {}
        out["linear@best"] = lin(len(tr)).fit(X(tr, [lay]), ytr)
        out["linear@final"] = lin(len(tr)).fit(X(tr, [L]), ytr)
        out["concat"] = lin(len(tr)).fit(X(tr, LAYERS), ytr)
        out["mlp"] = mlp(len(tr)).fit(X(tr, [L]), ytr)
        out["mlp@concat"] = mlp(len(tr)).fit(X(tr, LAYERS), ytr)

        res.setdefault(f, {})
        for k, m in out.items():
            pred = m.predict(X(te, [L] if k in ("linear@final", "mlp") else
                                 (LAYERS if "concat" in k else [lay])))
            res[f][k] = float(np.mean(pred == yte))
            per_item.setdefault(k, {}).update({i: bool(p == y[i]) for i, p in zip(te, pred)})

        # fuse: mix the best probe's log-probs with the model's own answer-space log-probs.
        # The weight is chosen on the calibration split, which the probe never saw.
        best = out["mlp@concat"]; feat = lambda ids: X(ids, LAYERS)
        cls = list(best.classes_)
        def blend(ids, w):
            lp = np.log(np.clip(best.predict_proba(feat(ids)), 1e-9, 1))
            hit = []
            for n, i in enumerate(ids):
                u = U.get(i)
                if u is None: hit.append(cls[int(np.argmax(lp[n]))]); continue
                sp, lv = u["space"], np.array(u["lv"], float)
                m2 = np.full(len(cls), -30.0)
                for j, c in enumerate(cls):
                    if c in sp: m2[j] = lv[sp.index(c)]
                m2 = np.log(np.clip(softmax(m2), 1e-9, 1))
                hit.append(cls[int(np.argmax((1 - w) * lp[n] + w * m2))])
            return hit
        ws = np.linspace(0, 1, 11)
        yca = np.array([y[i] for i in ca])
        w = float(ws[int(np.argmax([np.mean(np.array(blend(ca, w_)) == yca) for w_ in ws]))])
        pred = blend(te, w)
        res[f]["fuse"] = float(np.mean(np.array(pred) == yte)); res[f]["fuse_w"] = w
        per_item.setdefault("fuse", {}).update({i: bool(p == y[i]) for i, p in zip(te, pred)})

    keys = ["linear@best", "linear@final", "concat", "mlp", "mlp@concat", "fuse"]
    base = {}
    for line in open("runs/branches6_test.jsonl"):
        r = json.loads(line); base[r["id"]] = bool(r["ok"]["none"])
    ids = list(base)
    n = {f: sum(1 for i in ids if i.split("_")[0] == f) for f in fams}
    print(f"{'readout':14s}" + "".join(f"{f:>10s}" for f in fams) + f"{'ALL':>8s}{'vs B-read':>11s}")
    for k in keys:
        allv = np.average([res[f][k] for f in fams], weights=[n[f] for f in fams])
        p = mcnemar([per_item[k][i] for i in ids], [per_item["linear@best"][i] for i in ids])
        print(f"{k:14s}" + "".join(f"{100*res[f][k]:9.0f}%" for f in fams)
              + f"{100*allv:7.1f}%{p:11.2g}")
    if lora:
        lv = np.average([np.mean([lora[i]["ok"] for i in ids if i.split("_")[0] == f]) for f in fams],
                        weights=[n[f] for f in fams])
        print(f"{'[lora full]':14s}"
              + "".join(f"{100*np.mean([lora[i]['ok'] for i in ids if i.split('_')[0]==f]):9.0f}%"
                        for f in fams) + f"{100*lv:7.1f}%")
    print(f"{'[base]':14s}" + "".join(
        f"{100*np.mean([base[i] for i in ids if i.split('_')[0]==f]):9.0f}%" for f in fams)
        + f"{100*np.mean(list(base.values())):7.1f}%")
    print("\nfuse mixing weight per family (0 = probe only, 1 = model only): "
          + ", ".join(f"{f} {res[f]['fuse_w']:.1f}" for f in fams))
    json.dump({"acc": res, "per_item": {k: {i: bool(v) for i, v in d.items()}
                                        for k, d in per_item.items()}},
              open("runs/readout_3b.json", "w"))
    print("wrote runs/readout_3b.json")


if __name__ == "__main__":
    main()
