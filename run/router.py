# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Does the evidence margin predict WHICH correction wins?

Headroom (G3) and signal (G1) are both established. Neither implies a router: they have to be
coupled. This script asks that question in two steps, and reports the negative honestly if it
comes out negative.

  1. Coupling diagnostic. For each branch, the distribution of the routing features on items
     where that branch is the one that works. If the features do not separate, no threshold
     rule can, and nothing below matters.
  2. The router. A decision list over (margin m, probe/base agreement a, blindfold-probe
     confidence s, probe confidence c) with ONE threshold set shared across all four families
     -- per-family thresholds would be four tuning surfaces and the win would mean nothing.
     Thresholds are grid-searched on the probe's *selection* split, which the probe never
     trained on, then frozen and applied once to the test split.

Baselines it must beat, in increasing order of strength:
     base model  <  best single branch globally  <  best branch per family (uses family
     identity, which the router also gets)  <  oracle (the ceiling).
"""
import itertools, json, sys
import numpy as np

BR = ["none", "steer", "prior", "look"]

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def features(states, meta_ids, P, fam, ids):
    """Recompute the frozen probe's outputs for the given ids, on visual and blind states."""
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
    pf = P[fam]; l = pf["layer"]
    mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
    pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
    coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
    cls = pf["classes"]
    out = {}
    for iid in ids:
        i = pos[iid]
        def p_of(X):
            z = ((X[i, l] - mean) / scale - pm) @ comp.T
            return softmax(coef @ z + inter)
        pv, pb = p_of(V), p_of(B)
        k = int(np.argmax(pv))
        out[iid] = dict(pred=cls[k], conf=float(pv[k]), prior=float(pb.max()), pv=pv, pb=pb,
                        cls=cls,
                        margin=float(np.log(max(pv[k], 1e-12)) - np.log(max(pb[k], 1e-12))))
    return out


def support(f, answer):
    """How much the image raises the probe belief in THE MODEL OWN answer. A missing answer
    means the probe class set does not contain it, which is itself evidence against it."""
    if answer is None or str(answer) not in f["cls"]:
        return -12.0, 0.0
    j = f["cls"].index(str(answer))
    return (float(np.log(max(f["pv"][j], 1e-12)) - np.log(max(f["pb"][j], 1e-12))),
            float(f["pv"][j]))

def load(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(path))}

def route(f, th):
    tau_hi, tau_lo, th_c, th_s = th
    if f["p_ans"] >= th_c:   return "none"
    if f["gap"]   >= tau_hi: return "steer"
    if f["prior"] >= th_s:   return "prior"
    if f["sup"]   <  tau_lo: return "look"
    return "none"

def main():
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    cal, tst = load("runs/branches_calib_3b.jsonl"), load("runs/branches_3b.jsonl")
    F = {}
    for fam in P:
        ids = [i for i in list(cal) + list(tst) if (cal.get(i) or tst.get(i))["family"] == fam]
        F.update(features("runs/states_3b.npz", None, P, fam, ids))
    def pack(store):
        rows = []
        for iid, r in store.items():
            f = dict(F[iid]); f["agree"] = (f["pred"] == str(r["out"]["none"]))
            f["sup"], f["p_ans"] = support(f, r["out"]["none"])
            f["gap"] = f["conf"] - f["p_ans"]
            rows.append(dict(id=iid, family=r["family"], feat=f, ok=r["ok"],
                             oracle=any(r["ok"].values())))
        return rows
    C, T = pack(cal), pack(tst)
    print(f"calibration {len(C)} items | test {len(T)} items\n")

    print("=" * 76)
    print("1. COUPLING: do the features separate the branch that works?")
    print("=" * 76)
    print(f"{'branch uniquely correct':26s}{'n':>5s}{'support(ans)':>13s}{'p(ans)':>11s}"
          f"{'gap':>11s}")
    for b in BR:
        sel = [r for r in C + T if r["ok"][b] and sum(r["ok"].values()) == 1]
        if not sel: print(f"{b:26s}{0:5d}{'-':>13s}{'-':>11s}{'-':>12s}"); continue
        m = np.mean([r["feat"]["sup"] for r in sel]); c = np.mean([r["feat"]["p_ans"] for r in sel])
        a = np.mean([r["feat"]["gap"] for r in sel])
        print(f"{b:26s}{len(sel):5d}{m:13.2f}{c:11.2f}{a:11.2f}")
    none_ok = [r for r in C + T if r["ok"]["none"]]
    none_no = [r for r in C + T if not r["ok"]["none"]]
    for lbl, g in (("base correct", none_ok), ("base wrong  ", none_no)):
        print(f"  {lbl} (n={len(g):3d}): probe-margin {np.mean([r['feat']['margin'] for r in g]):+6.2f}"
              f"   support(model answer) {np.mean([r['feat']['sup'] for r in g]):+6.2f}"
              f"   p(model answer) {np.mean([r['feat']['p_ans'] for r in g]):.2f}")

    print()
    print("=" * 76)
    print("2. ROUTER: one threshold set for all families, fitted on calibration only")
    print("=" * 76)
    gaps = np.array([r["feat"]["gap"] for r in C]); sup = np.array([r["feat"]["sup"] for r in C])
    grid = list(itertools.product(np.quantile(gaps, [.3, .5, .7, .85, .95]),
                                  np.quantile(sup, [.05, .15, .3, .5]),
                                  [0.15, 0.3, 0.5, 0.7], [0.5, 0.7, 0.9, 0.97]))
    base_cal = np.mean([r["ok"]["none"] for r in C])
    best = None
    for th in grid:
        acc = np.mean([r["ok"][route(r["feat"], th)] for r in C])
        if acc >= base_cal - 0.005 and (best is None or acc > best[1]):
            best = (th, acc)
    th, cal_acc = best
    print(f"  frozen theta: p(model answer)>={th[2]} -> none | gap>={th[0]:+.2f} -> steer | "
          f"prior>={th[3]} -> prior | support<{th[1]:+.2f} -> look")
    print(f"  calibration accuracy {100*cal_acc:.1f}%  (base {100*base_cal:.1f}%)\n")

    rows = []
    for fam in sorted(P):
        g = [r for r in T if r["family"] == fam]
        acc = {b: np.mean([r["ok"][b] for r in g]) for b in BR}
        rt = np.mean([r["ok"][route(r["feat"], th)] for r in g])
        orc = np.mean([r["oracle"] for r in g])
        bestfam = max(acc.values())
        got = (rt - bestfam) / (orc - bestfam) if orc > bestfam else float("nan")
        rows.append((fam, len(g), acc["none"], bestfam, rt, orc, got))
    print(f"{'family':10s}{'n':>5s}{'base':>7s}{'best/fam':>10s}{'ROUTER':>8s}{'oracle':>8s}"
          f"{'vs best':>9s}{'% of headroom':>15s}")
    for fam, n, b0, bf, rt, orc, got in rows:
        print(f"{fam:10s}{n:5d}{100*b0:6.0f}%{100*bf:9.0f}%{100*rt:7.0f}%{100*orc:7.0f}%"
              f"{100*(rt-bf):+8.1f}pp{(100*got if got==got else float('nan')):14.0f}%")
    allb = np.mean([r["ok"]["none"] for r in T])
    allbf = max(np.mean([r["ok"][b] for r in T]) for b in BR)
    allrt = np.mean([r["ok"][route(r["feat"], th)] for r in T])
    allorc = np.mean([r["oracle"] for r in T])
    print(f"{'ALL':10s}{len(T):5d}{100*allb:6.0f}%{100*allbf:9.0f}%{100*allrt:7.0f}%{100*allorc:7.0f}%"
          f"{100*(allrt-allbf):+8.1f}pp{100*(allrt-allbf)/(allorc-allbf):14.0f}%")
    print(f"\n  branch usage on test: "
          f"{ {b: sum(1 for r in T if route(r['feat'], th) == b) for b in BR} }")
    json.dump(dict(theta=list(map(float, th)), cal_acc=float(cal_acc),
                   test=dict(base=float(allb), best_fixed=float(allbf), router=float(allrt),
                             oracle=float(allorc))), open("runs/router_3b.json", "w"), indent=1)

if __name__ == "__main__":
    main()
