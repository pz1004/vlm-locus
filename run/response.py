"""Does an INTERVENTION-RESPONSE feature break the 55% ceiling?

Everything the router reads so far describes the item: is the evidence present, does it
support the model's answer. Nothing describes the item-branch interaction -- how this state
would *react* to a correction. That is the information the oracle has and the router does not.

The cheap version costs no GPU at all. The steering branch adds alpha * d to the residual
state; the probe is a function of that state; so the probe's response to steering can be
simulated in numpy:

    resp   = p_probe(top | h + a*d_top) - p_probe(top | h)     how movable this state is
    flip   = argmax p_probe(h + a*d_wrong) != argmax p_probe(h) a per-ITEM puppet-string test
    ans_up = p_probe(model answer | h + a*d_top) - p_probe(model answer | h)

`flip` is the interesting one: the specificity control currently exists only per cell (29% of
spatial items), so a router cannot tell which individual items are puppet strings. This gives
it that, for free.
"""
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier

BR = ["none", "steer", "prior", "look"]
ALPHAS = [0.3, 1.0]


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def main():
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz"); meta = json.load(open("runs/states_3b_meta.json"))
    V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
    pos = {m["id"]: i for i, m in enumerate(meta)}
    cal = [json.loads(l) for l in open("runs/branches_calib_3b.jsonl")]
    tst = [json.loads(l) for l in open("runs/branches_3b.jsonl")]
    fams = sorted(P)

    F = {}
    for fam, pf in P.items():
        l = pf["layer"]
        mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
        pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
        coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
        cls = [str(c) for c in pf["classes"]]; D = np.array(pf["directions"])
        def prob(h):
            return softmax(coef @ (((h - mean) / scale - pm) @ comp.T) + inter)
        for r in [x for x in cal + tst if x["family"] == fam]:
            i = pos[r["id"]]; h = V[i, l]; nrm = float(np.linalg.norm(h))
            pv, pb = prob(h), prob(B[i, l])
            k = int(np.argmax(pv)); worst = int(np.argmin(pv))
            a = r["out"]["none"]
            j = cls.index(str(a)) if (a is not None and str(a) in cls) else None
            f = dict(p_ans=float(pv[j]) if j is not None else 0.0,
                     conf=float(pv[k]), prior=float(pb.max()))
            f["gap"] = f["conf"] - f["p_ans"]
            f["sup"] = (float(np.log(max(pv[j], 1e-12)) - np.log(max(pb[j], 1e-12)))
                        if j is not None else -12.0)
            for al in ALPHAS:
                ptop = prob(h + al * nrm * D[k]); pbad = prob(h + al * nrm * D[worst])
                f[f"resp{al}"] = float(ptop[k] - pv[k])
                f[f"flip{al}"] = float(int(np.argmax(pbad) != k))
                f[f"ansup{al}"] = float((ptop[j] - pv[j]) if j is not None else 0.0)
            F[r["id"]] = f
        print(f"  {fam:10s} response features simulated", flush=True)

    BASE = ["p_ans", "gap", "sup", "prior", "conf"]
    RESP = [f"{n}{a}" for a in ALPHAS for n in ("resp", "flip", "ansup")]

    def mat(rows, names):
        return np.array([[F[r["id"]][n] for n in names] + [r["family"] == f for f in fams]
                         for r in rows], float)

    def learned(names):
        Xc, Xt = mat(cal, names), mat(tst, names)
        prob = np.zeros((len(tst), len(BR)))
        for bi, b in enumerate(BR):
            y = np.array([r["ok"][b] for r in cal], int)
            if y.min() == y.max(): prob[:, bi] = y.max(); continue
            clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12,
                                         random_state=0).fit(Xc, y)
            prob[:, bi] = clf.predict_proba(Xt)[:, list(clf.classes_).index(1)]
        pick = [BR[int(np.argmax(prob[i]))] for i in range(len(tst))]
        return np.mean([r["ok"][p] for r, p in zip(tst, pick)]), pick

    orc = np.mean([any(r["ok"].values()) for r in tst])
    bf = max(np.mean([r["ok"][b] for r in tst]) for b in BR)
    cap = lambda a: 100 * (a - bf) / (orc - bf)
    a1, _ = learned(BASE)
    a2, pick2 = learned(BASE + RESP)
    prev = json.load(open("runs/policy_3b.json"))

    print(f"\n  best single fixed branch          {100*bf:5.1f}%")
    print(f"  LRC, thresholds + family          {100*prev['B']:5.1f}%   {cap(prev['B']):3.0f}% of headroom")
    print(f"  learned, item features only       {100*a1:5.1f}%   {cap(a1):3.0f}%")
    print(f"  learned, + intervention response  {100*a2:5.1f}%   {cap(a2):3.0f}%   <-")
    print(f"  oracle                            {100*orc:5.1f}%")
    for f in fams:
        g = [(r, p) for r, p in zip(tst, pick2) if r["family"] == f]
        bfam = max(np.mean([r["ok"][b] for r, _ in g]) for b in BR)
        print(f"    {f:10s} best/fam {100*bfam:3.0f}%  ->  {100*np.mean([r['ok'][p] for r, p in g]):3.0f}%"
              f"   oracle {100*np.mean([any(r['ok'].values()) for r, _ in g]):3.0f}%")

    # is the per-item puppet-string flag informative about the cell-level rate?
    print("\n  per-item puppet-string flag vs the measured per-cell rate:")
    for f in fams:
        g = [r for r in tst if r["family"] == f]
        fl = np.mean([F[r["id"]]["flip1.0"] for r in g])
        st = np.mean([r["ok"]["steer"] for r in g]); nn = np.mean([r["ok"]["none"] for r in g])
        print(f"    {f:10s} flip rate {100*fl:3.0f}%   steer-vs-base {100*(st-nn):+5.1f} pp")
    json.dump(dict(item_only=float(a1), with_response=float(a2), oracle=float(orc),
                   best_fixed=float(bf)), open("runs/response_3b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
