"""How many labels does reading the latent actually need?

B-read and LoRA draw on the same resource -- a labelled calibration set -- so the honest
comparison is not one accuracy against another but two curves against the label budget. This
sweeps the probe's training-set size on the frozen states and evaluates on the same held-out
split used everywhere else, with 5 resamples per size so the small-n end has an error bar.

Sizes are per family. The full split is 165 items/family, so the left end of this curve is the
regime where fine-tuning a 3B model on 10 examples per family is not a serious proposition.
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


def main(states="runs/states_3b.npz", g1="runs/probe_g1.json",
         branches="runs/branches6_test.jsonl", out="runs/datasize_3b.json"):
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32); G = json.load(open(g1))
    macc = {}
    for line in open(branches):
        r = json.loads(line); macc.setdefault(r["family"], []).append(r["ok"]["none"])
    res = {}
    print(f"{'family':10s}{'model':>7s}" + "".join(f"{'n='+str(s):>12s}" for s in SIZES))
    for f in sorted(TARGET):
        l = G[f]["layer"]
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        y = np.array([TARGET[f](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8])
        idx, y = idx[keep], y[keep]
        tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=0, stratify=y)
        _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
        Xtr, ytr, Xte, yte = V[idx[tr], l], y[tr], V[idx[te], l], y[te]
        row, cur = [], []
        for s in SIZES:
            if s >= len(tr):
                accs = [float(make_pipeline(StandardScaler(),
                                            PCA(n_components=min(64, len(tr) - 1), random_state=0),
                                            LogisticRegression(max_iter=1000, C=0.5))
                              .fit(Xtr, ytr).score(Xte, yte))]
            else:
                accs = []
                for seed in range(5):
                    rng = np.random.default_rng(seed)
                    sub = rng.choice(len(tr), size=s, replace=False)
                    if len(set(ytr[sub])) < 2: continue
                    accs.append(float(make_pipeline(
                        StandardScaler(), PCA(n_components=min(64, s - 1), random_state=0),
                        LogisticRegression(max_iter=1000, C=0.5)).fit(Xtr[sub], ytr[sub])
                        .score(Xte, yte)))
            cur.append(dict(n=s, mean=float(np.mean(accs)), sd=float(np.std(accs))))
            row.append(f"{100*np.mean(accs):7.0f}%±{100*np.std(accs):2.0f}")
        m = float(np.mean(macc[f]))
        res[f] = dict(model=m, curve=cur, n_train=int(len(tr)), n_test=int(len(te)))
        print(f"{f:10s}{100*m:6.0f}%" + "".join(f"{c:>12s}" for c in row))
    print("\nn is labelled items per family; ± is sd over 5 resamples. The rightmost column is")
    print("the full split (single fit). Compare with LoRA, which sees all 165 per family.")
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
