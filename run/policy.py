"""Is 46% of headroom a limit of the SIGNAL or of the POLICY CLASS?

The shared threshold list captured 46% and lost to always-on `look` on spatial. Two very
different explanations, with opposite consequences:

  * policy-class limit -- the features do identify the winning branch, but a four-threshold
    decision list cannot express the rule. Then a richer policy on the SAME features recovers
    more, and the fix is the policy (plan repair R-d).
  * signal limit -- the features do not identify the winning branch at all. Then no policy
    helps, the ceiling is the estimator, and the honest report is plan outcome C.

Three policies on identical features and identical splits:
  A  shared threshold list                       (what the router does today)
  B  A, plus a family-specific fallback branch chosen on calibration   (repair R-d)
  C  learned: per-branch success predictor, pick argmax   (the signal ceiling on this
     feature set -- if C is also low, the features are the problem)

Everything is fitted on the probe's selection split and evaluated once on the test split.
"""
import itertools, json
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import sys
BR = ["none", "steer", "prior", "look", "attn", "cot"]
FEATS = ["p_ans", "gap", "sup", "prior", "conf", "margin"]


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def build():
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz")
    meta = json.load(open("runs/states_3b_meta.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)

    def rows(path):
        out = []
        for line in open(path):
            r = json.loads(line); fam = r["family"]; pf = P[fam]; l = pf["layer"]
            mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
            pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
            coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
            cls = pf["classes"]; i = pos[r["id"]]
            pv = softmax(coef @ (((V[i, l] - mean) / scale - pm) @ comp.T) + inter)
            pb = softmax(coef @ (((B[i, l] - mean) / scale - pm) @ comp.T) + inter)
            k = int(np.argmax(pv)); ans = r["out"]["none"]
            j = cls.index(str(ans)) if (ans is not None and str(ans) in cls) else None
            sup = (np.log(max(pv[j], 1e-12)) - np.log(max(pb[j], 1e-12))) if j is not None else -12.0
            p_ans = float(pv[j]) if j is not None else 0.0
            out.append(dict(id=r["id"], family=fam, ok=r["ok"], oracle=any(r["ok"].values()),
                            f=dict(p_ans=p_ans, gap=float(pv[k]) - p_ans, sup=float(sup),
                                   prior=float(pb.max()), conf=float(pv[k]),
                                   margin=float(np.log(max(pv[k], 1e-12)) - np.log(max(pb[k], 1e-12))))))
        return out
    CAL = sys.argv[1] if len(sys.argv) > 1 else "runs/branches_calib_3b.jsonl"
    TST = sys.argv[2] if len(sys.argv) > 2 else "runs/branches_3b.jsonl"
    c, t = rows(CAL), rows(TST)
    global BR
    BR = [b for b in BR if b in c[0]["ok"]]
    return c, t


def policy_a(f, th, fallback="none"):
    """The threshold list still names the four original actions; any extra branch enters
    through the family fallback, which is chosen on calibration over the full library."""
    if f["p_ans"] >= th[2]: return "none"
    if f["gap"]   >= th[0]: return "steer"
    if f["prior"] >= th[3]: return "prior"
    if f["sup"]   <  th[1]: return "look"
    return fallback


def fit_a(C, fallbacks=None):
    g = np.array([r["f"]["gap"] for r in C]); s = np.array([r["f"]["sup"] for r in C])
    grid = itertools.product(np.quantile(g, [.3, .5, .7, .85, .95]),
                             np.quantile(s, [.05, .15, .3, .5]),
                             [0.15, 0.3, 0.5, 0.7], [0.5, 0.7, 0.9, 0.97])
    base = np.mean([r["ok"]["none"] for r in C]); best = None
    for th in grid:
        acc = np.mean([r["ok"][policy_a(r["f"], th, (fallbacks or {}).get(r["family"], "none"))]
                       for r in C])
        if acc >= base - 0.005 and (best is None or acc > best[1]): best = (th, acc)
    return best[0]


def main():
    C, T = build()
    print(f"calibration {len(C)}  test {len(T)}\n")
    fams = sorted({r["family"] for r in T})

    # A ---- shared threshold list
    thA = fit_a(C)
    accA = {f: np.mean([r["ok"][policy_a(r["f"], thA)] for r in T if r["family"] == f]) for f in fams}
    allA = np.mean([r["ok"][policy_a(r["f"], thA)] for r in T])

    # B ---- same list, family-specific fallback chosen on calibration (repair R-d)
    fb = {}
    for f in fams:
        g = [r for r in C if r["family"] == f]
        fb[f] = max(BR, key=lambda b: np.mean([r["ok"][b] for r in g]))
    thB = fit_a(C, fb)
    accB = {f: np.mean([r["ok"][policy_a(r["f"], thB, fb[f])] for r in T if r["family"] == f])
            for f in fams}
    allB = np.mean([r["ok"][policy_a(r["f"], thB, fb[r["family"]])] for r in T])

    # C ---- learned per-branch success predictors on the same features
    Xc = np.array([[r["f"][k] for k in FEATS] + [r["family"] == f for f in fams] for r in C], float)
    Xt = np.array([[r["f"][k] for k in FEATS] + [r["family"] == f for f in fams] for r in T], float)
    prob = np.zeros((len(T), len(BR)))
    for bi, b in enumerate(BR):
        y = np.array([r["ok"][b] for r in C], int)
        if y.min() == y.max(): prob[:, bi] = y.max(); continue
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=0).fit(Xc, y)
        prob[:, bi] = clf.predict_proba(Xt)[:, list(clf.classes_).index(1)]
    pickC = [BR[int(np.argmax(prob[i]))] for i in range(len(T))]
    accC = {f: np.mean([r["ok"][p] for r, p in zip(T, pickC) if r["family"] == f]) for f in fams}
    allC = np.mean([r["ok"][p] for r, p in zip(T, pickC)])

    orc = {f: np.mean([r["oracle"] for r in T if r["family"] == f]) for f in fams}
    bfam = {f: max(np.mean([r["ok"][b] for r in T if r["family"] == f]) for b in BR) for f in fams}
    allorc = np.mean([r["oracle"] for r in T]); allbf = max(np.mean([r["ok"][b] for r in T]) for b in BR)

    print(f"{'family':10s}{'best/fam':>10s}{'A shared':>10s}{'B +family':>11s}{'C learned':>11s}"
          f"{'oracle':>8s}")
    for f in fams:
        print(f"{f:10s}{100*bfam[f]:9.0f}%{100*accA[f]:9.0f}%{100*accB[f]:10.0f}%"
              f"{100*accC[f]:10.0f}%{100*orc[f]:7.0f}%")
    print(f"{'ALL':10s}{100*allbf:9.0f}%{100*allA:9.0f}%{100*allB:10.0f}%{100*allC:10.0f}%"
          f"{100*allorc:7.0f}%")
    cap = lambda a: 100 * (a - allbf) / (allorc - allbf)
    print(f"\nheadroom captured vs the best single fixed branch:")
    print(f"   A shared thresholds  {cap(allA):5.0f}%")
    print(f"   B + family fallback  {cap(allB):5.0f}%     (plan repair R-d)")
    print(f"   C learned on the same features  {cap(allC):5.0f}%     <- the signal ceiling")
    print(f"\n  family fallbacks chosen on calibration: {fb}")
    print(f"\nIf C is far above A, the limit was the policy class. If C is also low, the limit is")
    print(f"the estimator, and no policy over these features reaches the oracle.")
    json.dump(dict(A=float(allA), B=float(allB), C=float(allC), oracle=float(allorc),
                   best_fixed=float(allbf), fallbacks=fb, branches=BR),
              open("runs/policy6_3b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
