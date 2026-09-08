# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Cost-tiered LRC: spend compute where the signal says the failure needs it.

The 6-branch router captures 51% of headroom, but that number is not cost-matched: B-cot
decodes 140 tokens where the others decode 10, and essentially all of its value is on one
family. A router that reaches for it everywhere is buying an average 14x decode to fix
tracking. The question this asks is whether the locus signal can buy it only where it pays.

Selection rule:  argmax_b  P(correct | x, b)  -  lam * cost_b
with cost_b MEASURED on this GPU (runs/cost_3b.json), normalised to cost(B-none) = 1, and
the probe read charged to every routed item since the router pays it before choosing.

lam sweeps from 0 (accuracy at any price) to large (free tier only), tracing an
accuracy-vs-compute frontier. Three things have to be true for the frontier to mean anything:

  * lam is chosen on CALIBRATION for a target budget and evaluated once on TEST. Sweeping lam
    on test and reporting the best point is test-set fitting with extra steps.
  * it must beat the RANDOM-ESCALATION control: escalate the same fraction of items to the
    same expensive branch, chosen at random. This is the router's puppet-string test -- if
    random spending does as well, the signal is not what is choosing.
  * it must beat what the same compute buys without any locus machinery: Best-of-N
    self-consistency, and majority vote over the whole branch library.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier

BR = ["none", "steer", "prior", "look", "attn", "cot"]
WITH_READ = False
FEATS = ["p_ans", "gap", "sup", "prior", "conf", "margin"]


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def rows(path, P, V, Bl, pos):
    out = []
    for line in open(path):
        r = json.loads(line); fam = r["family"]; pf = P[fam]; l = pf["layer"]
        mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
        pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
        coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
        cls = pf["classes"]; i = pos[r["id"]]
        pv = softmax(coef @ (((V[i, l] - mean) / scale - pm) @ comp.T) + inter)
        pb = softmax(coef @ (((Bl[i, l] - mean) / scale - pm) @ comp.T) + inter)
        k = int(np.argmax(pv)); ans = r["out"]["none"]
        j = cls.index(str(ans)) if (ans is not None and str(ans) in cls) else None
        sup = (np.log(max(pv[j], 1e-12)) - np.log(max(pb[j], 1e-12))) if j is not None else -12.0
        ok, o = dict(r["ok"]), dict(r["out"])
        if WITH_READ:                       # B-read: emit the probe's own argmax as the answer
            ok["read"] = (cls[k] == r["answer"]); o["read"] = cls[k]
        out.append(dict(id=r["id"], family=fam, ok=ok, out=o, answer=r["answer"],
                        puppet=(r["spec_steer"] == r["spec_class"]),
                        f=dict(p_ans=float(pv[j]) if j is not None else 0.0,
                               gap=float(pv[k]) - (float(pv[j]) if j is not None else 0.0),
                               sup=float(sup), prior=float(pb.max()), conf=float(pv[k]),
                               margin=float(np.log(max(pv[k], 1e-12)) - np.log(max(pb[k], 1e-12))))))
    return out


def load_costs(path="runs/cost_3b.json", basis="sec"):
    """Measured cost per branch, by family, normalised to one base-model call.

    Two bases, because neither alone is defensible:
      sec  wall-clock on this GPU -- what a user waits for, but hardware-specific, and it
           rewards one big batched forward over many small ones.
      tok  transformer token-forwards, prefill + ACTUAL decoded tokens (not the budget:
           greedy answers stop after ~1 token, and CoT stops after ~35 of its 140).
           Hardware-independent, roughly FLOP-proportional, blind to kernel efficiency.

    How the router is charged. It cannot route before it has the base answer -- its features
    include the probe's confidence in the model's OWN answer -- so every routed item pays the
    fused base pass (generation and probe read in one forward, measured at 1.01x) and then the
    branch it selects. Routing to B-none therefore costs 1.01, not 0.

    B-prior is charged 2|A| forwards because that is how it is implemented: one per candidate,
    against the image and against the blindfold. Batching the answer space would cut it by
    roughly |A|, so its position on the frontier is an implementation fact about this code,
    not a property of contrastive rescoring.
    """
    rec = json.load(open(path))
    def tok(r):
        prefills = r["prefill_tok"] / max(r["prompt_tok"], 1)
        return r["prefill_tok"] + max(r["fwd"] - prefills, 0)
    key = (lambda r: r["sec"]) if basis == "sec" else tok
    med = defaultdict(dict)
    fams = sorted({r["family"] for r in rec})
    for f in fams:
        for b in set(r["branch"] for r in rec):
            v = [key(r) for r in rec if r["family"] == f and r["branch"] == b]
            if v: med[f][b] = float(np.median(v))
    base = float(np.median([key(r) for r in rec if r["branch"] == "none"]))
    fused = float(np.median([key(r) for r in rec if r["branch"] == "none+probe"])) / base
    cost = {}
    for f in fams:
        cost[f] = {"none": fused}
        for b in BR:
            if b == "read":
                cost[f][b] = fused        # the probe state rides on the base pass; no extra work
            elif b != "none" and b in med[f]:
                cost[f][b] = fused + med[f][b] / base
    return cost, base, med, fused


def fit_predictors(C, fams):
    X = np.array([[r["f"][k] for k in FEATS] + [r["family"] == f for f in fams] for r in C], float)
    models = {}
    for b in BR:
        y = np.array([r["ok"][b] for r in C], int)
        if y.min() == y.max():
            models[b] = float(y.max()); continue
        models[b] = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12,
                                           random_state=0).fit(X, y)
    return models


def predict(models, R, fams):
    X = np.array([[r["f"][k] for k in FEATS] + [r["family"] == f for f in fams] for r in R], float)
    p = np.zeros((len(R), len(BR)))
    for bi, b in enumerate(BR):
        m = models[b]
        p[:, bi] = m if isinstance(m, float) else m.predict_proba(X)[:, list(m.classes_).index(1)]
    return p


def route_guarded(R, p, cost, lam, ban_steer, default, delta):
    """Deviate from the per-family default branch only when the predictor prefers something
    else by more than `delta` in utility.

    Unguarded argmax over 7 per-branch predictors fitted on 239 calibration items loses to the
    single best branch whenever that branch is strong: seven noisy estimates give the maximum a
    positive bias, and the router chases it. Defaulting and requiring a margin is the standard
    abstention fix, and `delta` is fitted on calibration like every other threshold.
    """
    picks = []
    for i, r in enumerate(R):
        f = r["family"]; d = default[f]
        u = np.array([p[i, bi] - lam * cost[f][b] for bi, b in enumerate(BR)])
        if ban_steer.get(f): u[BR.index("steer")] = -1e9
        k = int(np.argmax(u))
        picks.append(BR[k] if u[k] - u[BR.index(d)] > delta else d)
    acc = float(np.mean([r["ok"][b] for r, b in zip(R, picks)]))
    cst = float(np.mean([cost[r["family"]][b] for r, b in zip(R, picks)]))
    return picks, acc, cst


def route(R, p, cost, lam, ban_steer):
    """argmax over utility = P(correct) - lam * cost. Returns picks, accuracy, mean cost."""
    picks = []
    for i, r in enumerate(R):
        u = [p[i, bi] - lam * cost[r["family"]][b]
             for bi, b in enumerate(BR)]
        if ban_steer.get(r["family"]):
            u[BR.index("steer")] = -1e9
        picks.append(BR[int(np.argmax(u))])
    acc = float(np.mean([r["ok"][b] for r, b in zip(R, picks)]))
    cst = float(np.mean([cost[r["family"]][b] for r, b in zip(R, picks)]))
    return picks, acc, cst


def mcnemar_p(a, b):
    from scipy.stats import binomtest
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    return binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


def vote(ss):
    ss = [s for s in ss if s is not None]
    return Counter(ss).most_common(1)[0][0] if ss else None


def main():
    global BR, WITH_READ
    basis = sys.argv[1] if len(sys.argv) > 1 else "sec"
    WITH_READ = "read" in sys.argv
    if WITH_READ: BR = BR + ["read"]
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz")
    meta = json.load(open("runs/states_3b_meta.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V, Bl = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
    C = rows("runs/branches6_calib.jsonl", P, V, Bl, pos)
    T = rows("runs/branches6_test.jsonl", P, V, Bl, pos)
    fams = sorted({r["family"] for r in T})
    cost, base_sec, med, fused = load_costs(basis=basis)

    # steer is discarded per family where it is a puppet string (>50% follow a wrong push)
    ban = {f: np.mean([r["puppet"] for r in C if r["family"] == f]) > 0.5 for f in fams}

    unit = "s" if basis == "sec" else " tokens"
    print(f"measured branch cost [{basis}], relative to B-none (= {base_sec:.3f}{unit});\n"
          f"the fused base+probe pass costs {fused:.2f} and every routed item pays it")
    print(f"{'family':10s}" + "".join(f"{b:>8s}" for b in BR))
    for f in fams:
        print(f"{f:10s}" + "".join(f"{cost[f][b]:8.2f}" for b in BR))
    print(f"{'mean':10s}" + "".join(
        f"{np.mean([cost[f][b] for f in fams]):8.2f}" for b in BR))
    print(f"\nfamilies where steer is a puppet string and is banned: "
          f"{[f for f in fams if ban[f]] or 'none'}\n")

    models = fit_predictors(C, fams)
    pC, pT = predict(models, C, fams), predict(models, T, fams)

    # ---- frontier: lam chosen on CALIBRATION for each budget, evaluated once on TEST
    lams = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 60)])
    cal = [(l,) + route(C, pC, cost, l, ban)[1:] for l in lams]
    print("frontier  (lam fit on calibration, evaluated once on test)")
    print(f"{'budget':>8s}{'lam':>9s}{'cal acc':>9s}{'cal cost':>10s}"
          f"{'TEST acc':>10s}{'TEST cost':>11s}{'cot rate':>10s}{'random ctl':>12s}")
    front = []
    seen = set()
    for target in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 99.0):
        ok = [(l, a, c) for l, a, c in cal if c <= target]
        if not ok: continue
        l, ca, cc = max(ok, key=lambda t: t[1])
        if l in seen: continue
        seen.add(l)
        picks, ta, tc = route(T, pT, cost, l, ban)
        rate = float(np.mean([b == "cot" for b in picks]))
        rctl = random_escalation_control(T, picks, cost, ban, n=200)
        front.append(dict(target=target, lam=float(l), cal_acc=ca, cal_cost=cc,
                          test_acc=ta, test_cost=tc, cot_rate=rate, rand=rctl))
        print(f"{target:8.2f}{l:9.4f}{100*ca:8.0f}%{cc:10.2f}"
              f"{100*ta:9.1f}%{tc:11.2f}{100*rate:9.0f}%{100*rctl:11.1f}%")

    # ---- reference points on the same plane
    print(f"\nreference points (test split, {len(T)} items)")
    print(f"{'policy':28s}{'acc':>8s}{'cost':>8s}")
    for b in BR:
        a = np.mean([r["ok"][b] for r in T])
        c = np.mean([cost[r["family"]][b] for r in T])
        note = "  (inadmissible on tracking)" if b == "steer" and any(ban.values()) else ""
        print(f"{'always ' + b:28s}{100*a:7.1f}%{c:8.2f}{note}")
    allc = np.mean([sum(cost[r["family"]][b] for b in BR
                        if not (b == "steer" and ban[r["family"]])) for r in T])
    va = np.mean([vote([r["out"][b] for b in BR
                        if not (b == "steer" and ban[r["family"]])]) == r["answer"] for r in T])
    print(f"{'vote over all branches':28s}{100*va:7.1f}%{allc:8.2f}")
    orc = np.mean([any(r["ok"][b] for b in BR if not (b == "steer" and ban[r["family"]]))
                   for r in T])
    print(f"{'oracle over library':28s}{100*orc:7.1f}%{allc:8.2f}")
    try:
        bon = [json.loads(l) for l in open("runs/bon_3b.jsonl")]
        bon = {r["id"]: r for r in bon}
        have = [r for r in T if r["id"] in bon]
        print(f"\nBest-of-N self-consistency ({len(have)} of {len(T)} test items)")
        for k in (2, 3, 5, 8):
            a = np.mean([vote(bon[r["id"]]["samples"][:k]) == r["answer"] for r in have])
            print(f"{'  BoN k=' + str(k):28s}{100*a:7.1f}%{float(k):8.2f}")
        a = np.mean([any(s == r["answer"] for s in bon[r["id"]]["samples"]) for r in have])
        print(f"{'  sampling oracle (any/8)':28s}{100*a:7.1f}%{8.0:8.2f}")
    except FileNotFoundError:
        print("\n(Best-of-N not yet run)")

    # where does the router spend? at the operating point nearest 2x
    op = min(front, key=lambda d: abs(d["test_cost"] - 2.0))
    picks, ta, tc = route(T, pT, cost, op["lam"], ban)
    print(f"\nwhere the compute goes at the {tc:.2f}x operating point ({100*ta:.1f}% overall)")
    print(f"{'family':10s}{'base':>7s}{'LRC':>7s}{'delta':>8s}{'cost':>7s}" +
          "".join(f"{b:>8s}" for b in BR))
    for f in fams:
        g = [(r, b) for r, b in zip(T, picks) if r["family"] == f]
        bacc = np.mean([r["ok"]["none"] for r, _ in g])
        racc = np.mean([r["ok"][b] for r, b in g])
        rc = np.mean([cost[f][b] for _, b in g])
        mix = Counter(b for _, b in g)
        print(f"{f:10s}{100*bacc:6.0f}%{100*racc:6.0f}%{100*(racc-bacc):+7.1f}pp{rc:7.2f}" +
              "".join(f"{100*mix[b]/len(g):7.0f}%" for b in BR))

    # ---- guarded routing: does defaulting to the best fixed branch fix the small-n problem?
    default = {f: max(BR, key=lambda b: np.mean([r["ok"][b] for r in C if r["family"] == f]))
               for f in fams}
    print(f"\nguarded routing (default per family, deviate only on a margin)")
    print(f"  calibration defaults: {default}")
    print(f"{'delta':>8s}{'cal acc':>9s}{'TEST acc':>10s}{'TEST cost':>11s}{'deviate':>9s}")
    best_d, best_ca = None, -1
    for delta in (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40):
        _, ca, _ = route_guarded(C, pC, cost, op["lam"], ban, default, delta)
        if ca > best_ca: best_d, best_ca = delta, ca
    for delta in (0.0, 0.05, 0.10, 0.25):
        gp, ga, gc = route_guarded(T, pT, cost, op["lam"], ban, default, delta)
        dev = np.mean([b != default[r["family"]] for r, b in zip(T, gp)])
        _, ca, _ = route_guarded(C, pC, cost, op["lam"], ban, default, delta)
        mark = "  <- delta fitted on calibration" if delta == best_d else ""
        print(f"{delta:8.2f}{100*ca:8.0f}%{100*ga:9.1f}%{gc:11.2f}{100*dev:8.0f}%{mark}")
    gp, ga, gc = route_guarded(T, pT, cost, op["lam"], ban, default, best_d)
    bfacc = np.mean([r["ok"][default[r["family"]]] for r in T])
    bfcost = np.mean([cost[r["family"]][default[r["family"]]] for r in T])
    print(f"  guarded router {100*ga:.1f}% at {gc:.2f}x   vs per-family default "
          f"{100*bfacc:.1f}% at {bfcost:.2f}x   p={mcnemar_p(
              [r['ok'][b] for r, b in zip(T, gp)],
              [r['ok'][default[r['family']]] for r in T]):.2g}")

    # ---- inference at the operating point: the plan's W1/W3/W5 as paired tests
    from scipy.stats import binomtest
    def mcnemar(a, b):
        """exact McNemar on discordant pairs: a beats b."""
        n01 = sum(1 for x, y in zip(a, b) if x and not y)
        n10 = sum(1 for x, y in zip(a, b) if y and not x)
        if n01 + n10 == 0: return 1.0, n01, n10
        return binomtest(n01, n01 + n10, 0.5).pvalue, n01, n10
    bestb = max(BR, key=lambda b: np.mean([r["ok"][b] for r in T]))
    print(f"\npaired tests at the {tc:.2f}x operating point   (best fixed branch: {bestb})")
    print(f"{'family':10s}{'LRC':>7s}{'base':>7s}{'p vs base':>11s}"
          f"{'best fix':>10s}{'p vs best':>11s}{'95% CI on LRC-base':>22s}")
    rng = np.random.default_rng(0)
    for f in fams + ["ALL"]:
        g = [(r, b) for r, b in zip(T, picks) if f == "ALL" or r["family"] == f]
        lrc = [r["ok"][b] for r, b in g]
        bas = [r["ok"]["none"] for r, _ in g]
        bfx = [r["ok"][bestb] for r, _ in g]
        p1, n01, _ = mcnemar(lrc, bas); p2, _, _ = mcnemar(lrc, bfx)
        d = np.array(lrc, float) - np.array(bas, float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{f:10s}{100*np.mean(lrc):6.1f}%{100*np.mean(bas):6.1f}%{p1:11.2g}"
              f"{100*np.mean(bfx):9.1f}%{p2:11.2g}   [{100*lo:+5.1f},{100*hi:+5.1f}] pp")
    # ---- how many test items would the confirmatory run need per cell?
    print(f"\nn required per cell for 80% power at the observed effect (McNemar, alpha=.05)")
    print(f"{'family':10s}{'delta':>8s}{'discord':>9s}{'n needed':>10s}")
    for f in fams:
        g = [(r, b) for r, b in zip(T, picks) if r["family"] == f]
        lrc = [r["ok"][b] for r, b in g]; bas = [r["ok"]["none"] for r, _ in g]
        n01 = sum(1 for x, y in zip(lrc, bas) if x and not y)
        n10 = sum(1 for x, y in zip(lrc, bas) if y and not x)
        d = (n01 - n10) / len(g); pd = (n01 + n10) / len(g)
        if pd == 0 or d == 0:
            print(f"{f:10s}{100*d:+7.1f}pp{100*pd:8.0f}%{'--':>10s}"); continue
        # exact-binomial power approximation: need n*pd discordant pairs to resolve the split
        psi = n01 / (n01 + n10)
        need = (1.96 * 0.5 + 0.84 * np.sqrt(psi * (1 - psi))) ** 2 / ((psi - 0.5) ** 2 * pd)
        print(f"{f:10s}{100*d:+7.1f}pp{100*pd:8.0f}%{need:10.0f}")

    print("\nW3 (do no harm, <=0.5 pp loss vs base): "
          + ("PASS" if all(np.mean([r["ok"][b] for r, b in zip(T, picks) if r["family"] == f])
                           >= np.mean([r["ok"]["none"] for r in T if r["family"] == f]) - 0.005
                           for f in fams) else "FAIL"))
    print("W5 (no single always-on branch matches LRC): "
          + ("PASS" if max(np.mean([r["ok"][b] for r in T]) for b in BR) < ta else "FAIL"))
    print(f"W4 (cost <= the baseline it beats): LRC {tc:.2f}x vs "
          f"always-{bestb} {np.mean([cost[r['family']][bestb] for r in T]):.2f}x -> "
          + ("PASS" if tc <= np.mean([cost[r["family"]][bestb] for r in T]) else "FAIL"))

    json.dump(dict(frontier=front, cost={f: cost[f] for f in fams}, base_sec=base_sec,
                   banned=[f for f in fams if ban[f]]),
              open(f"runs/tiered_3b_{basis}.json", "w"), indent=1)


def random_escalation_control(T, picks, cost, ban, n=200):
    """Same number of escalations to the same branches, but on random items.

    If this matches the router, the compute was what mattered and the signal was not.
    """
    rng = np.random.default_rng(0)
    per_fam = defaultdict(Counter)
    for r, b in zip(T, picks):
        per_fam[r["family"]][b] += 1
    accs = []
    for _ in range(n):
        tot = 0.0; cnt = 0
        for f, cc in per_fam.items():
            g = [r for r in T if r["family"] == f]
            bag = [b for b, k in cc.items() for _ in range(k)]
            rng.shuffle(bag)
            tot += sum(r["ok"][b] for r, b in zip(g, bag)); cnt += len(g)
        accs.append(tot / cnt)
    return float(np.mean(accs))


if __name__ == "__main__":
    main()
