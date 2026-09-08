"""Is the signal ceiling caused by reading ONE layer?

The router reads a single pre-selected layer. But the locus story is explicitly about a
trajectory -- the evidence is present in the middle of the stack and overridden later -- so
the shape of that trajectory should carry information a single layer throws away.

For each family, fit the same capacity-controlled probe at every 3rd layer on the SAME train
split, then for each item read three trajectories across depth:

    p_ans[l]   probe belief in the MODEL's answer          (what it said)
    p_pred[l]  probe belief in its own top class           (what the image says)
    sup[l]     log p_ans on visual minus blindfolded state (what the image adds)

and derive depth features: where support for the model's answer peaks, how much it decays by
the final layer, where the probe's own reading overtakes it. Then re-run the same three
policies. If the learned ceiling rises above 55%, depth carried signal the single layer lost.
"""
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BR = ["none", "steer", "prior", "look"]
TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]))
STEP = 3


def main():
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz"); meta = json.load(open("runs/states_3b_meta.json"))
    V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
    pos = {m["id"]: i for i, m in enumerate(meta)}
    L = V.shape[1]; layers = list(range(0, L, STEP))
    cal = [json.loads(l) for l in open("runs/branches_calib_3b.jsonl")]
    tst = [json.loads(l) for l in open("runs/branches_3b.jsonl")]

    traj = {}
    for fam, pf in P.items():
        tr_ids = pf["train_ids"]
        tr = np.array([pos[i] for i in tr_ids])
        ytr = np.array([TARGET[fam](meta[i]) for i in tr])
        want = [r for r in cal + tst if r["family"] == fam]
        idx = np.array([pos[r["id"]] for r in want])
        pa = np.zeros((len(want), len(layers))); pp = np.zeros_like(pa); sp = np.zeros_like(pa)
        for li, l in enumerate(layers):
            clf = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=0),
                                LogisticRegression(max_iter=1000, C=0.5)).fit(V[tr, l], ytr)
            cls = [str(c) for c in clf.classes_]
            Pv = clf.predict_proba(V[idx, l]); Pb = clf.predict_proba(B[idx, l])
            for k, r in enumerate(want):
                a = r["out"]["none"]
                j = cls.index(str(a)) if (a is not None and str(a) in cls) else None
                pa[k, li] = Pv[k, j] if j is not None else 0.0
                pp[k, li] = Pv[k].max()
                sp[k, li] = (np.log(max(Pv[k, j], 1e-12)) - np.log(max(Pb[k, j], 1e-12))
                             ) if j is not None else -12.0
        for k, r in enumerate(want):
            traj[r["id"]] = dict(pa=pa[k], pp=pp[k], sp=sp[k])
        print(f"  {fam:10s} probes fitted at {len(layers)} depths", flush=True)

    third = max(1, len(layers) // 3)
    def feats(iid):
        t = traj[iid]; pa, pp, sp = t["pa"], t["pp"], t["sp"]
        return [pa.max(), float(np.argmax(pa)) / len(pa), pa[-1], pa[:third].mean(),
                pa[-third:].mean(), pa[-third:].mean() - pa[:third].mean(),
                (pp - pa).max(), float(np.argmax(pp - pa)) / len(pa),
                sp.mean(), sp[-1], pp[-1], pp.max()]
    NAMES = ["pa_max", "pa_argmax", "pa_final", "pa_early", "pa_late", "pa_delta",
             "gap_max", "gap_argmax", "sup_mean", "sup_final", "pp_final", "pp_max"]

    fams = sorted(P)
    def mat(rows):
        return np.array([feats(r["id"]) + [r["family"] == f for f in fams] for r in rows], float)
    Xc, Xt = mat(cal), mat(tst)

    # single-layer ceiling, for reference: the previous run reached 55%
    def learned(Xc, Xt, cal, tst):
        prob = np.zeros((len(tst), len(BR)))
        for bi, b in enumerate(BR):
            y = np.array([r["ok"][b] for r in cal], int)
            if y.min() == y.max(): prob[:, bi] = y.max(); continue
            clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12,
                                         random_state=0).fit(Xc, y)
            prob[:, bi] = clf.predict_proba(Xt)[:, list(clf.classes_).index(1)]
        return [BR[int(np.argmax(prob[i]))] for i in range(len(tst))]

    pick = learned(Xc, Xt, cal, tst)
    acc = {f: np.mean([r["ok"][p] for r, p in zip(tst, pick) if r["family"] == f]) for f in fams}
    allacc = np.mean([r["ok"][p] for r, p in zip(tst, pick)])
    orc = np.mean([any(r["ok"].values()) for r in tst])
    bf = max(np.mean([r["ok"][b] for r in tst]) for b in BR)
    prev = json.load(open("runs/policy_3b.json"))

    print(f"\n{'family':10s}{'best/fam':>10s}{'single-layer':>14s}{'multi-layer':>13s}{'oracle':>8s}")
    for f in fams:
        bfam = max(np.mean([r["ok"][b] for r in tst if r["family"] == f]) for b in BR)
        of = np.mean([any(r["ok"].values()) for r in tst if r["family"] == f])
        print(f"{f:10s}{100*bfam:9.0f}%{'':7s}{'':0s}{100*acc[f]:6.0f}%{100*of:12.0f}%")
    cap = lambda a: 100 * (a - bf) / (orc - bf)
    print(f"\n  best single fixed branch      {100*bf:5.1f}%")
    print(f"  LRC, single layer             {100*prev['B']:5.1f}%   {cap(prev['B']):3.0f}% of headroom")
    print(f"  LRC, depth features           {100*allacc:5.1f}%   {cap(allacc):3.0f}% of headroom")
    print(f"  oracle                        {100*orc:5.1f}%")

    # which depth features the trees actually use
    y = np.array([r["ok"]["none"] for r in cal], int)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=0).fit(Xc, y)
    imp = sorted(zip(NAMES + fams, clf.feature_importances_), key=lambda x: -x[1])[:5]
    print("\n  top features for predicting 'the base answer is already correct':")
    for n, v in imp:
        if v > 0: print(f"     {n:12s} {v:.2f}")
    json.dump(dict(multilayer=float(allacc), single=float(prev["B"]), oracle=float(orc),
                   best_fixed=float(bf)), open("runs/multilayer_3b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
