"""The label-budget curve again, with the probe's hyperparameters tuned per budget.

run/datasize.py held PCA=64 and C=0.5 fixed at every training-set size, which is unfair at the
small-n end: with 20 labels and an 18-class answer space, 64 components fitted on 20 points is
noise. LoRA was given oracle epoch selection in the same comparison, so the probe gets the
matching courtesy -- a small grid over components and regularisation, chosen on the *calibration*
split, which is disjoint from both the probe's training data and the test set.

If the method still loses after this, it loses on merit.
"""
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]))
SIZES = [10, 20, 40, 80, 120, 165]
GRID = [(k, C) for k in [8, 16, 32, 64] for C in [0.05, 0.5, 5.0]]


def main(states="runs/states_3b.npz", g1="runs/probe_g1.json", out="runs/datasize2_3b.json"):
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32); G = json.load(open(g1))
    calib = json.load(open("runs/calib_ids.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    res = {}
    print(f"{'family':10s}" + "".join(f"{'n='+str(s):>13s}" for s in SIZES))
    for f in sorted(TARGET):
        l = G[f]["layer"]
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        y = np.array([TARGET[f](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8])
        idx, y = idx[keep], y[keep]
        tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=0, stratify=y)
        _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
        Xtr, ytr, Xte, yte = V[idx[tr], l], y[tr], V[idx[te], l], y[te]
        ca = [i for i in calib[f] if i in pos]
        Xca = V[[pos[i] for i in ca], l]
        yca = np.array([str(TARGET[f](meta[pos[i]])) for i in ca])
        row, cur = [], []
        for s in SIZES:
            accs, picks = [], []
            for seed in range(5 if s < len(tr) else 1):
                rng = np.random.default_rng(seed)
                sub = (rng.choice(len(tr), size=s, replace=False) if s < len(tr)
                       else np.arange(len(tr)))
                if len(set(ytr[sub])) < 2: continue
                best = None
                for k, C in GRID:
                    if k >= len(sub): continue
                    p = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                                      LogisticRegression(max_iter=1000, C=C)).fit(Xtr[sub], ytr[sub])
                    v = float(np.mean(p.predict(Xca).astype(str) == yca))   # chosen on calibration
                    if best is None or v > best[0]: best = (v, p, (k, C))
                accs.append(float(best[1].score(Xte, yte))); picks.append(best[2])
            cur.append(dict(n=s, mean=float(np.mean(accs)), sd=float(np.std(accs)),
                            cfg=str(picks[0])))
            row.append(f"{100*np.mean(accs):6.0f}%±{100*np.std(accs):2.0f}")
        res[f] = dict(curve=cur, n_train=int(len(tr)), n_test=int(len(te)))
        print(f"{f:10s}" + "".join(f"{c:>13s}" for c in row))
    nte = {f: res[f]["n_test"] for f in res}; tot = sum(nte.values())
    print(f"{'ALL':10s}" + "".join(
        f"{100*sum(res[f]['curve'][k]['mean']*nte[f] for f in res)/tot:12.1f}%"
        for k in range(len(SIZES))))
    print("\nhyperparameters (PCA components, C) chosen on the calibration split at each budget.")
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
