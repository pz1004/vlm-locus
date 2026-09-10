# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Every reported number, from ONE probe protocol.

The study accumulated two probe pipelines. `run/fit_probes.py` selects a layer on a held-out
selection split and serialises the probe for downstream consumers (cfprobe, lora comparison);
`run/layers.py` fits the same capacity-controlled probe at every layer with fixed
hyperparameters and reports the FINAL layer. They disagree in 9 of 28 (dataset, family) cells,
by up to 21 pp -- not a bug in either, but two different estimators, and a paper cannot ship
both.

Every reported number uses `layers.py` at the final layer, for three reasons:
  1. it is the quantity the readout claim is *about* -- the vector the LM head itself consumes,
     with no layer search to defend;
  2. it is the more conservative of the two on the headline cells and never benefits from
     best-of-L selection, which the 3 px glyph control prices at 8.5 pp (peak) vs 2.5 pp (final);
  3. it gives the *stronger* prediction result (lower bound 12/12 vs 11/12), so the choice is
     not self-serving in the direction of the claim.
The four headline chart cells are identical to 0.0 pp under either estimator, so no conclusion
depends on this decision; `--audit` prints the full disagreement table so a reader can check.

Counterfactual follow rates come from `cfprobe_*.json`, which was run with the serialised
(selected-layer) probe. That is reported rather than papered over: re-running
the counterfactual pass with the final-layer probe needs the GPU, and the follow rate is a
*conditional* statistic whose role is to separate ~85% from ~20%, a distinction no 3 pp shift in
probe accuracy can touch.
"""
from __future__ import annotations
import argparse, glob, json, os, re

# Output locations. Defaults keep everything inside the repository so a fresh
# clone runs standalone; a caller may override them to write straight
# into paper/.
JSON = os.environ.get("VLM_LOCUS_JSON", "out/canon.json")
TEX  = os.environ.get("VLM_LOCUS_TEX",  "out/tables")
import numpy as np
from scipy.stats import spearmanr, binomtest, ttest_rel, f as f_dist

# (tag, model label, dataset label) -> the layer sweep and its counterfactual pass
REAL = [("real",            "q3b",  "Qwen-3B bf16"),
        ("realchart_v2",    "q3b",  "Qwen-3B bf16"),
        ("q3b4_real_3b",    "q3b4", "Qwen-3B nf4"),
        ("q3b4_real_chart_v2", "q3b4", "Qwen-3B nf4"),
        ("q7b_real_3b",     "q7b",  "Qwen-7B nf4"),
        ("q7b_real_chart_v2", "q7b",  "Qwen-7B nf4"),
        ("ivl_real_3b",     "ivl",  "InternVL3-2B"),
        ("ivl_real_chart_v2", "ivl",  "InternVL3-2B"),
        ("smol_real_3b",    "smol", "SmolVLM"),
        ("smol_real_chart_v2", "smol", "SmolVLM")]
SYNTH = [("3b", "q3b", "Qwen-3B"), ("smolm", "smol", "SmolVLM")]

# The artefact floor used to be a constant here (2.5 pp, priced once on the designed-absence
# family). It is now per cell, from run/nullcal.py's label-permutation null: a cell must clear
# the 95th percentile of what its own probe scores when trained on permuted labels. The null
# ranges from 9.3% to 61.3% across this grid as class count and n vary, so no single constant
# could have served.
#
# FOLLOW_MIN is a substantive design threshold rather than an artefact estimate -- a reader that
# tracks the attribute should track an edit to it nearly always, not half the time -- so it is
# not calibrated against a null. But it is applied to the *lower confidence bound* on the follow
# rate, not to the point estimate, for two reasons. The denominator is conditional (items the
# reader answered correctly before the edit) and on the designed-absence family it falls to 3-9
# of 75, where a point estimate carries almost no information: 1/5 reads as "20% follow". And a
# point-estimate gate decided one cell by two items (synthetic counting, 30/57 = 52.6%), which
# made the threshold's exact value load-bearing for a verdict -- the same objection this protocol
# raises against a constant artefact floor. With the bound, the seven surviving loci clear the
# threshold by 16.5 to 44.9 pp and no verdict turns on the gate's precise position.
FOLLOW_MIN, NULL_ALPHA = 50.0, 0.05

# The rare-class filter: a label needs this many items in a family to be fitted at all. It is a
# protocol constant and it lives here, but the analysis path still carries it as the literal 8 in
# layers.py, nullcal.py, cffollow.py, splits.py and headread.py -- verify_protocol.py check 13
# asserts those agree with this, because a filter that differs between the probe and the data
# builder silently produces edit targets the probe cannot emit. gen/build_chart.py imports it.
MIN_CLASS = 8

# Which counterfactual statistic the verdict is gated on. The forward follow rate is conditional
# on the probe being right before the edit, so its denominator is a probe-dependent subset -- and
# it is silently truncated by the probe's class support, because predict() cannot return a label
# it never trained on and the edits routinely produce one (see run/cffollow.py). The joint
# two-endpoint accuracy is prespecified instead, on three grounds that are fixed before any of
# them is computed: its denominator is n, it conditions on no probe outcome, and it is symmetric
# in the two endpoints. Forward, in-support and reverse are computed and reported for every cell
# beside it; exactly one cell's verdict differs between them, and the manuscript names it.
GATE_CI = "probe_joint_ci"

# The presence test's own thresholds. G1 is a mandatory condition of the locus verdict, so these
# are as much part of the decision rule as FOLLOW_MIN and owe the reader the same sensitivity
# sweep -- which g1_sensitivity() computes. run/probe.py's docstring gives the reasons for each.
G1_BLIND, G1_SHUF, G1_POS = 0.05, 0.10, 0.75


def wilson(k, m, z=1.96):
    """Wilson score interval; the normal approximation is unusable at these denominators."""
    if not m: return [float("nan"), float("nan")]
    ph, d = k / m, 1 + z * z / m
    c, h = (ph + z * z / (2 * m)) / d, z * np.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def follow(tag):
    """probe/model accuracy before the edit and follow rate, per family, from the paired pass.

    Prefers runs/cffollow_<tag>.json, which run/cffollow.py computes with the FINAL-layer probe
    from run/cfcapture.py's counterfactual states -- the same estimator as every other quantity
    in the locus verdict. runs/cfprobe_<tag>.json, the selected-layer version, is still read so
    that both can be reported; the estimator shift moves individual cells by up to 25 pp.
    """
    fin, sel = f"runs/cffollow_{tag}.json", f"runs/cfprobe_{tag}.json"
    f = fin if os.path.exists(fin) else sel
    if not os.path.exists(f): return {}
    alt = {}
    if f == fin and os.path.exists(sel):
        for fam in {r["family"] for r in json.load(open(sel))}:
            g2 = [r for r in json.load(open(sel)) if r["family"] == fam]
            ok2 = [r for r in g2 if r["probe0"] == r["a0"]]
            alt[fam] = (float(np.mean([r["probe1"] == r["a1"] for r in ok2]))
                        if ok2 else float("nan"), len(ok2))
    out = {}
    for fam in {r["family"] for r in json.load(open(f))}:
        g = [r for r in json.load(open(f)) if r["family"] == fam]
        def st(pre, post):
            # follow is conditioned on being correct before the edit, so the denominator is
            # not n and differs between the probe and the model. Reporting the ratio alone
            # hides that two readers were scored on two different subsets.
            ok = [r for r in g if r[pre] == r["a0"]]
            num = sum(r[post] == r["a1"] for r in ok)
            return (float(np.mean([r[pre] == r["a0"] for r in g])),
                    float(num / len(ok)) if ok else float("nan"), int(num), int(len(ok)))
        p0, pf, pn, pd = st("probe0", "probe1"); m0, mf, mn, md = st("model0", "model1")

        def rate(num, den):
            return dict(v=float(num / den) if den else float("nan"), num=int(num), den=int(den),
                        ci=wilson(num, den))

        # forward, restricted to items whose post-edit answer the probe could actually emit
        sup = [r for r in g if r.get("a1_in_support", True)]
        oks = [r for r in sup if r["probe0"] == r["a0"]]
        fs = rate(sum(r["probe1"] == r["a1"] for r in oks), len(oks))
        # reverse: given the probe reads the edited image correctly, does it read the original?
        # in support by construction, since probe1 == a1 requires a1 in classes_
        rev = [r for r in g if r["probe1"] == r["a1"]]
        rv = rate(sum(r["probe0"] == r["a0"] for r in rev), len(rev))
        # joint: both endpoints right, denominator n, no conditioning at all
        jt = rate(sum(r["probe0"] == r["a0"] and r["probe1"] == r["a1"] for r in g), len(g))
        # probe against model on one shared denominator. The unrestricted comparison is not
        # like-for-like: the model generates freely and the probe cannot leave its class support,
        # so the model is scored on items where the probe's answer is unavailable by construction.
        mt = [r for r in sup if r["probe0"] == r["a0"] and r["model0"] == r["a0"]]
        mp = rate(sum(r["probe1"] == r["a1"] for r in mt), len(mt))
        mm = rate(sum(r["model1"] == r["a1"] for r in mt), len(mt))
        out[fam] = dict(n=len(g), probe_acc0=p0, probe_follow=pf, model_acc0=m0, model_follow=mf,
                        probe_follow_num=pn, probe_follow_den=pd,
                        model_follow_num=mn, model_follow_den=md,
                        probe_follow_ci=wilson(pn, pd), model_follow_ci=wilson(mn, md),
                        probe_unsupported=int(len(g) - len(sup)),
                        probe_follow_sup=fs["v"], probe_follow_sup_num=fs["num"],
                        probe_follow_sup_den=fs["den"], probe_follow_sup_ci=fs["ci"],
                        probe_reverse=rv["v"], probe_reverse_num=rv["num"],
                        probe_reverse_den=rv["den"], probe_reverse_ci=rv["ci"],
                        probe_joint=jt["v"], probe_joint_num=jt["num"],
                        probe_joint_den=jt["den"], probe_joint_ci=jt["ci"],
                        matched_den=mp["den"], probe_matched=mp["v"],
                        probe_matched_num=mp["num"], probe_matched_ci=mp["ci"],
                        model_matched=mm["v"], model_matched_num=mm["num"],
                        model_matched_ci=mm["ci"],
                        follow_layer="final" if f == fin else "selected",
                        probe_follow_sel=alt.get(fam, (float("nan"), 0))[0],
                        probe_follow_sel_den=alt.get(fam, (float("nan"), 0))[1])
    return out


G1_FILE = {"3b": "runs/probe_g1.json"}   # the first sweep predates the per-tag naming


def nullcal(tag, fam):
    f = f"runs/null_{tag}.json"
    if not os.path.exists(f): return None
    return json.load(open(f)).get(fam)


def g1(tag, fam):
    """G1 at the final layer, under the full gate run/probe.py defines.

    Two things were wrong before. The verdict came from runs/probe_g1_*.json, computed at the
    layer selected on the selection split, while the table beside it printed the final-layer
    probe -- which is how cells_real.tex once carried a 6.7% probe marked as beating its
    shuffled control
    by more than 10 pp. And the gate dropped run/probe.py's two margin conditions, which flips
    11 of 36 cells. Both are fixed by reading run/nullcal.py's final-layer file, which computes
    all four conditions at the layer the probe is read from.
    """
    d = nullcal(tag, fam)
    if d is not None:
        return dict(d, passes=bool(d["g1"]), layer="final")
    f = G1_FILE.get(tag, f"runs/probe_g1_{tag}.json")      # fallback: no null file yet
    if not os.path.exists(f): return None
    d = json.load(open(f)).get(fam)
    if d is None: return None
    return dict(d, passes=bool(100 * (d["vis"] - d["blind"]) > 5
                               and 100 * (d["vis"] - d["shuffled"]) > 10
                               and d.get("mean_margin", 0) > 0
                               and d.get("pos_margin", 0) > 0.75), layer="selected")


# the scored generations each layer sweep was aligned to, for the paired probe-vs-model test
GEN = {"3b": "runs/branches6_test.jsonl", "smolm": "runs/smolm_gen.jsonl"}


def g1_at(nc, blind=G1_BLIND, shuf=G1_SHUF, pos=G1_POS):
    """The presence test, recomputed from the stored final-layer statistics.

    run/nullcal.py already stores the verdict as `g1`; this reproduces it from its components so
    the thresholds can be swept. emit() asserts the two agree in every cell, which is what stops
    this from drifting into a second, quieter definition of the gate.
    """
    return bool(nc["vis"] - nc["blind_refit"] > blind and nc["vis"] - nc["shuffled"] > shuf
                and nc["mean_margin"] > 0 and nc["pos_margin"] > pos)


def locus(r, blind=G1_BLIND, shuf=G1_SHUF, pos=G1_POS, alpha=NULL_ALPHA, follow=FOLLOW_MIN):
    """The complete decision rule. Every condition and every constant, stated once.

    A cell is a readout locus if all four hold, each against its own reference:

      (1) presence.  The probe beats a blindfolded refit and its own shuffled-label control at
          the final layer, on accuracy and on per-item log-probability margin (g1_at). This
          prices the language prior, which a label permutation does not: permuting labels
          destroys the label-representation relation but leaves the image in place.
      (2) the null.  Its accuracy rejects its own label-permutation null (Eq. 1 of the paper).
      (3) the model. It beats the model on the same items under an exact paired test, in the
          direction of the probe -- the test is two-sided, so `gap > 0` supplies the direction.
      (4) the counterfactual.  The lower bound on GATE_CI's statistic clears `follow`.

    (1) is mandatory rather than auxiliary and was previously missing from this conjunction while
    the manuscript called it necessary. Adding it changes no verdict on this grid -- every locus
    already passed it -- but a protocol whose stated rule is not the one it runs has no rule.
    """
    nc, pq = r.get("nullcal"), r.get("paired")
    if nc is None or pq is None:
        return False
    return bool(g1_at(nc, blind, shuf, pos)
                and nc["p_null"] < alpha
                and pq["p"] < 0.05 and r["gap"] > 0
                and 100 * r.get(GATE_CI, [float("nan")])[0] > follow)


def paired(tag, fam):
    """McNemar (exact binomial on discordant pairs), canonical probe against the model."""
    pf = f"runs/canonpred_{tag}.json"
    gf = GEN.get(tag, f"runs/{tag}_gen.jsonl")
    if not (os.path.exists(pf) and os.path.exists(gf)): return None
    d = json.load(open(pf)).get(fam)
    if d is None: return None
    G = {}
    for line in open(gf):
        r = json.loads(line)
        G[r["id"]] = bool(r["ok"]["none"] if "ok" in r else r["gen_correct"])
    pr = dict(zip(d["ids"], d["pred"])); go = dict(zip(d["ids"], d["gold"]))
    ids = [i for i in d["ids"] if i in G]
    b = sum(pr[i] == go[i] and not G[i] for i in ids)
    c = sum(G[i] and pr[i] != go[i] for i in ids)
    return dict(probe_only=b, model_only=c, n=len(ids),
                p=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)


def agreement(tags, fam):
    """Item-level agreement between the canonical probes of different models, same items."""
    P = []
    for t in tags:
        f = f"runs/canonpred_{t}.json"
        if not os.path.exists(f): return None
        d = json.load(open(f)).get(fam)
        if d is None: return None
        P.append(dict(zip(d["ids"], d["pred"])))
    ids = sorted(set.intersection(*[set(p) for p in P]))
    out = {}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            out[f"{tags[i]}|{tags[j]}"] = dict(
                n=len(ids), agree=float(np.mean([P[i][k] == P[j][k] for k in ids])))
    return out


def bands(tag="realchart_v2", fam="chart", edges=(0, 25, 50, 75, 101)):
    """Where in the value range the gap lives -- the obvious follow-up question on charts."""
    pf, gf = f"runs/canonpred_{tag}.json", f"runs/{tag}_gen.jsonl"
    if not (os.path.exists(pf) and os.path.exists(gf)): return None
    d = json.load(open(pf))[fam]
    G = {r["id"]: r for r in (json.loads(l) for l in open(gf))}
    pr = dict(zip(d["ids"], d["pred"])); go = dict(zip(d["ids"], d["gold"]))
    out = []
    for lo, hi in zip(edges, edges[1:]):
        ids = [i for i in d["ids"] if lo <= float(G[i]["answer"]) < hi]
        if not ids: continue
        out.append(dict(lo=lo, hi=hi - 1, n=len(ids),
                        model=float(np.mean([bool(G[i]["gen_correct"]) for i in ids])),
                        probe=float(np.mean([pr[i] == go[i] for i in ids]))))
    return out


def cells(spec):
    rows = []
    for tag, model, label in spec:
        lf = f"runs/layers_{tag}.json"
        if not os.path.exists(lf): continue
        L, F = json.load(open(lf)), follow(tag)
        for fam, v in sorted(L.items()):
            gap = 100 * (v["final"] - v["model"])
            fl = F.get(fam, {}).get("probe_follow", float("nan"))
            nc, pq = nullcal(tag, fam), paired(tag, fam)
            # The generator declares an answer-space size in the item metadata and run/layers.py
            # copies it through as `chance`. That is not the probe's answer space: the labels the
            # generator actually emitted, and the rare-class filter, both narrow it. The declared
            # value therefore understates chance -- by 3.4 pp on synthetic counting, where 1/13 is
            # declared against 9 emitted classes, and 5.0 pp on real counting, 1/5 against 4. It
            # differs in 14 of 28 cells. Chance is uniform over the classes the probe is actually
            # fitted on; the declared value is kept beside it so the discrepancy stays auditable.
            declared = v["chance"]
            chance = 1 / nc["classes"] if nc is not None else declared
            r = dict(tag=tag, model=model, label=label, family=fam,
                     n_test=v["n_test"], chance=chance, chance_declared=declared,
                     classes=nc["classes"] if nc is not None else None, model_acc=v["model"],
                     model_all=v["model_all"], probe=v["final"], peak=v["peak"],
                     peak_layer=v["peak_layer"], gap=gap, follow=fl,
                     curve=v["vis"], blindcurve=v["blind"],
                     g1=g1(tag, fam), nullcal=nc,
                     **{k: v2 for k, v2 in F.get(fam, {}).items()},
                     paired=pq)
            # the verdict is locus()'s and only locus()'s -- see it for the four conditions
            r["readout"] = locus(r)
            rows.append(r)
    return rows


def prediction(real):
    """Probe gain vs fine-tuning gain, plus the free predictors it has to beat."""
    M = {(r["model"], r["family"]): r for r in json.load(open("runs/p3_lora_matched.json"))}
    idx = {(r["model"], r["family"]): r for r in real if r["model"] in {"q3b", "q7b", "ivl", "smol"}}
    rows = []
    for k, m in M.items():
        c = idx.get(k)
        if c is None: continue
        rows.append(dict(model=k[0], family=k[1], n=m["n"], base=c["model_acc"],
                         probe=c["probe"], lora=m["lora"], chance=c["chance"]))
    fam = [r["family"] for r in rows]
    def dm(x):
        x = np.array(x, float); o = x.copy()
        for f in set(fam):
            k = np.array([g == f for g in fam]); o[k] = x[k] - x[k].mean()
        return o
    # Rank on integer success counts. Accuracies are k/n, so mathematically equal differences
    # land on different floats and scipy assigns them distinct instead of averaged ranks --
    # which inflated the published coefficient. Counting restores the true tie structure.
    def ct(key): return [round(r[key] * r["n"]) for r in rows]
    kb, kp, kl = ct("base"), ct("probe"), ct("lora")
    lg = [l - b for l, b in zip(kl, kb)]
    cand = {"probe_gain":  [p - b for p, b in zip(kp, kb)],
            "probe_acc":   kp,
            "base_acc":    kb,
            "headroom":    [r["n"] - b for r, b in zip(rows, kb)],
            "above_chance":[b - r["n"] * r["chance"] for r, b in zip(rows, kb)]}
    stats = {}
    for k, v in cand.items():
        r1, p1 = spearmanr(v, lg); r2, p2 = spearmanr(dm(v), dm(lg))
        stats[k] = dict(rho=float(r1), p=float(p1), rho_dm=float(r2), p_dm=float(p2))
    inf = [r for r in rows if r["family"] != "glyph"]
    stats["lower_bound"] = dict(
        all=[int(sum(r["probe"] <= r["lora"] + 1e-9 for r in rows)), len(rows)],
        informative=[int(sum(r["probe"] <= r["lora"] + 1e-9 for r in inf)), len(inf)],
        # the bound is vacuous where the probe sits below the base model, so report the
        # non-vacuous form too: does adaptation beat the better of the two?
        strict=[int(sum(max(r["probe"], r["base"]) <= r["lora"] + 1e-9 for r in inf)), len(inf)],
        helped=[int(sum(r["lora"] > r["base"] for r in inf)), len(inf)])
    stats["coupling_null"] = coupling_null(kp, kb, kl, fam)
    stats["oos"] = oos(rows)
    # The same thing with the designed control dropped. On this grid the family-held-out
    # difference is -3.6 pp with the control in and +1.1 pp with it out, because adaptation
    # cannot help on a family with nothing to read and the probe correctly says so -- a fold
    # where the answer is "nothing happens, and the probe knows it" is not forecasting skill.
    # One of four folds should not be able to carry a headline, so both are reported.
    stats["oos_noctrl"] = oos([r for r in rows if r["family"] != "glyph"])
    return rows, stats


def coupling_null(kp, kb, kl, fam):
    """Both gains are measured against the same estimated baseline, so Cov(P-B, T-B) picks up
    +Var(B) even when P and T are unrelated. Permuting the probe within family, with (base,
    adapted) pairs intact, keeps that coupling in the null and prices it."""
    kp, kb, kl = np.array(kp, float), np.array(kb, float), np.array(kl, float)
    fam = np.array(fam)
    def dmv(x):
        o = x.copy()
        for f in set(fam.tolist()): o[fam == f] = x[fam == f] - x[fam == f].mean()
        return o
    def rho(p): return spearmanr(dmv(p - kb), dmv(kl - kb)).statistic
    obs, rng, draws = rho(kp), np.random.default_rng(0), []
    groups = [np.where(fam == f)[0] for f in sorted(set(fam.tolist()))]
    for _ in range(20000):
        q = kp.copy()
        for g in groups: q[g] = kp[rng.permutation(g)]
        draws.append(rho(q))
    draws = np.array(draws)
    return dict(rho=float(obs), null_mean=float(draws.mean()), null_sd=float(draws.std()),
                p=float((1 + (draws >= obs - 1e-12).sum()) / (1 + len(draws))), nperm=len(draws))


def oos(rows):
    """Does probe gain improve *out-of-sample* prediction of adapted accuracy over a baseline
    of family effects plus base accuracy? A correlation cannot answer this; held-out error can.
    Reported as mean absolute error in percentage points, leaving out one model or one family."""
    def design(rs, use_probe, fams):
        X = [[1.0] + [1.0 * (r["family"] == f) for f in fams[1:]] + [r["base"]]
             + ([r["probe"]] if use_probe else []) for r in rs]
        return np.array(X), np.array([r["lora"] for r in rs])
    out = {}
    for key in ("model", "family"):
        err = {False: [], True: []}; fold = {}
        for g in sorted({r[key] for r in rows}):
            tr = [r for r in rows if r[key] != g]; te = [r for r in rows if r[key] == g]
            fams = sorted({r["family"] for r in tr})
            e = {}
            for up in (False, True):
                Xtr, ytr = design(tr, up, fams); Xte, yte = design(te, up, fams)
                bb, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
                e[up] = np.abs(Xte @ bb - yte); err[up] += list(e[up])
            fold[g] = 100 * (e[True].mean() - e[False].mean())
        # A mean-absolute-error difference is not a result without an interval. On 4 folds and
        # 16 held-out cells a 0.2 pp difference is indistinguishable from zero, so report the
        # paired test and the interval and let the direction speak only if it is resolved.
        a, b = np.array(err[False]), np.array(err[True])
        dl = 100 * (b - a)
        se = float(dl.std(ddof=1) / np.sqrt(len(dl))) if len(dl) > 1 else float("nan")
        out[f"heldout_{key}"] = dict(mae_baseline=float(100 * a.mean()),
                                     mae_with_probe=float(100 * b.mean()),
                                     delta=float(dl.mean()), delta_se=se,
                                     ci=[float(dl.mean() - 1.96 * se), float(dl.mean() + 1.96 * se)],
                                     p_paired=float(ttest_rel(b, a).pvalue) if len(dl) > 1 else float("nan"),
                                     per_fold={g: float(d) for g, d in fold.items()},
                                     helps=bool(b.mean() < a.mean()),
                                     resolved=bool(ttest_rel(b, a).pvalue < 0.05) if len(dl) > 1 else False)
    # in-sample incremental F test, for contrast with the held-out result
    fams = sorted({r["family"] for r in rows})
    X0, y = design(rows, False, fams); X1, _ = design(rows, True, fams)
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    s0, s1 = float(r0 @ r0), float(r1 @ r1)
    df2 = len(y) - X1.shape[1]
    F = ((s0 - s1) / 1) / (s1 / df2) if df2 > 0 and s1 > 0 else float("nan")
    out["insample"] = dict(F=float(F), df=[1, int(df2)],
                           p=float(1 - f_dist.cdf(F, 1, df2)) if df2 > 0 else float("nan"))
    return out


def g1_sensitivity(rows):
    """Over what range of each of the presence test's three free constants is the locus set fixed?

    The follow threshold is not the protocol's only prespecified number: G1 is mandatory, so its
    5 pp blindfold margin, 10 pp shuffled margin and 0.75 positive-margin share are decision
    constants too. Each is swept alone, holding the other two and everything else fixed, which is
    what "no verdict turns on its precise position" actually asks. The mean-margin condition has
    no free constant -- zero is the only value that means anything -- so it is not swept.
    """
    base = frozenset(f"{r['tag']}/{r['family']}" for r in rows if locus(r))
    out = {}
    # the grids run the full admissible range of each constant, so a reported edge is the point
    # where a verdict actually moves rather than the end of the sweep
    for name, kw, grid in [("blind", "blind", [i / 200 for i in range(0, 201)]),
                           ("shuf", "shuf", [i / 200 for i in range(0, 201)]),
                           ("pos", "pos", [i / 200 for i in range(0, 200)])]:
        ok = [t for t in grid
              if frozenset(f"{r['tag']}/{r['family']}" for r in rows
                           if locus(r, **{kw: t})) == base]
        out[name] = dict(lo=min(ok), hi=max(ok), default=dict(blind=G1_BLIND, shuf=G1_SHUF,
                                                              pos=G1_POS)[kw])
    out["n"] = len(base)
    return out


def follow_sensitivity(rows, alpha=0.05):
    """Over what range of the one prespecified threshold is the verdict set unchanged?

    A prespecified constant owes the reader this sweep, on whichever statistic GATE_CI names.
    It is computed rather than asserted because the obvious guess is wrong: the stable range does
    not extend to zero, since below it the synthetic counting cell is admitted.
    """
    def at(t):
        return frozenset(f"{r['tag']}/{r['family']}" for r in rows
                         if locus(r, alpha=alpha, follow=t))
    base = at(FOLLOW_MIN)
    ts = [t for t in range(0, 101) if at(t) == base]
    lo, hi = min(ts), max(ts)
    # the cells immediately outside the stable range, and which way they move
    below = sorted(at(lo - 1) - base) if lo > 0 else []
    above = sorted(base - at(hi + 1)) if hi < 100 else []
    return dict(lo=lo, hi=hi, n=len(base), admitted_below=below, dropped_above=above,
                bounds={f"{r['tag']}/{r['family']}": 100 * r[GATE_CI][0]
                        for r in rows if f"{r['tag']}/{r['family']}" in base})


def multiplicity(rows, alpha=0.05):
    """Do the locus verdicts survive testing 28 cells?

    The permutation null's p is at its 1/(1+nperm) floor for every locus, so the binding
    quantity is the paired probe-vs-model test. Holm controls the family-wise error rate and is
    the conservative reading; Benjamini-Hochberg controls the false discovery rate and is the
    appropriate one for a grid whose purpose is to find where a gap lives.
    """
    ps = sorted(((f"{r['label']} {r['family']}", r["paired"]["p"], bool(r["readout"]))
                 for r in rows if r.get("paired")), key=lambda t: t[1])
    m = len(ps)
    holm, still = set(), True
    for i, (name, pv, _) in enumerate(ps):
        if still and pv <= alpha / (m - i): holm.add(name)
        else: still = False
    bh = set()
    for i, (name, pv, _) in enumerate(ps):
        if pv <= alpha * (i + 1) / m: bh = {n for n, _, _ in ps[:i + 1]}
    loci = {n for n, _, ro in ps if ro}
    return dict(n=m, holm=sorted(holm), bh=sorted(bh),
                loci_holm=sorted(loci & holm), loci_bh=sorted(loci & bh),
                loci=sorted(loci),
                loci_holm_fail=sorted(loci - holm), loci_bh_fail=sorted(loci - bh))


def audit():
    print(f"\n{'dataset':22}{'family':10}{'final':>8}{'peak':>8}{'selected':>10}{'diff':>8}")
    big = []
    for lf in sorted(glob.glob("runs/layers_*.json")):
        tag = os.path.basename(lf)[7:-5]
        pf = f"runs/probes_{tag}.npy"
        if not os.path.exists(pf): continue
        L, P = json.load(open(lf)), np.load(pf, allow_pickle=True).item()
        for fam in sorted(set(L) & set(P)):
            d = 100 * (P[fam]["test_acc"] - L[fam]["final"])
            if abs(d) >= 5: big.append((tag, fam, d))
            print(f"{tag:22}{fam:10}{100*L[fam]['final']:7.1f}%{100*L[fam]['peak']:7.1f}%"
                  f"{100*P[fam]['test_acc']:9.1f}%{d:+8.1f}")
    print(f"\n{len(big)} cells differ by >=5 pp between the two estimators; "
          f"worst {max(abs(d) for *_, d in big):.1f} pp")
    return big


def main(a):
    synth, real = cells(SYNTH), cells(REAL)
    pred_rows, pred = prediction(real)
    out = dict(synthetic=synth, real=real, prediction=dict(rows=pred_rows, stats=pred),
               multiplicity=multiplicity(synth + real),
               follow_sensitivity=follow_sensitivity(synth + real),
               g1_sensitivity=g1_sensitivity(synth + real),
               bands=bands(),
               agreement=agreement(["realchart_v2", "q3b4_real_chart_v2", "q7b_real_chart_v2",
                                    "ivl_real_chart_v2"], "chart"),
               protocol=dict(probe="layers.py final layer, fixed hyperparameters",
                             null_alpha=NULL_ALPHA, follow_min_pct=FOLLOW_MIN))
    os.makedirs(os.path.dirname(JSON) or ".", exist_ok=True)
    json.dump(out, open(JSON, "w"), indent=1, default=float)

    def show(rows, title):
        print(f"\n### {title}")
        print(f"{'model':22}{'family':10}{'n':>4}{'chance':>8}{'model':>8}{'probe':>8}"
              f"{'gap':>8}{'follow':>8}{'G1':>5}{'verdict':>9}")
        for r in rows:
            g = r["g1"]
            print(f"{r['label']:22}{r['family']:10}{r['n_test']:4d}{100*r['chance']:7.1f}%"
                  f"{100*r['model_acc']:7.1f}%{100*r['probe']:7.1f}%{r['gap']:+8.1f}"
                  f"{(f'{100*r[chr(102)+chr(111)+chr(108)+chr(108)+chr(111)+chr(119)]:6.0f}%' if not np.isnan(r['follow']) else '     --'):>8}"
                  f"{('PASS' if g and g['passes'] else ('fail' if g else '--')):>5}"
                  f"{('READOUT' if r['readout'] else '-'):>9}")
    show(synth, "synthetic families")
    show(real, "real-image families")

    print("\n### prediction: what forecasts fine-tuning gain (n=%d pairs)" % len(pred_rows))
    print(f"{'predictor':26}{'raw rho':>9}{'p':>9}{'demeaned rho':>14}{'p':>9}")
    NAME = {"probe_gain": "probe gain (probe-base)", "probe_acc": "probe accuracy",
            "base_acc": "base accuracy (free)", "headroom": "headroom 1-base (free)",
            "above_chance": "base above chance (free)"}
    for k, n in NAME.items():
        s = pred[k]
        print(f"{n:26}{s['rho']:+9.3f}{s['p']:9.4f}{s['rho_dm']:+14.3f}{s['p_dm']:9.4f}")
    print("\n### paired probe-vs-model (McNemar, canonical protocol)")
    for r in synth + real:
        q = r["paired"]
        if q and q["probe_only"] + q["model_only"] > 0:
            print(f"{r['label']:22}{r['family']:10} probe-only {q['probe_only']:3d}  "
                  f"model-only {q['model_only']:3d}  p={q['p']:.3g}")
    print("\n### real-chart probe agreement between models (same items)")
    for k, v in (out["agreement"] or {}).items():
        print(f"  {k:34} {100*v['agree']:5.1f}%  (n={v['n']})")
    print("\n### where the real-chart gap lives")
    for b in out["bands"] or []:
        print(f"  {b['lo']:3d}-{b['hi']:<3d} n={b['n']:3d}  model {100*b['model']:5.1f}%  "
              f"probe {100*b['probe']:5.1f}%  gap {100*(b['probe']-b['model']):+6.1f}")
    lb = pred["lower_bound"]
    print(f"\nprobe <= fine-tune: {lb['all'][0]}/{lb['all'][1]} pairs, "
          f"{lb['informative'][0]}/{lb['informative'][1]} excluding the glyph floor")
    emit(synth, real, pred_rows, pred, out)
    if a.audit: audit()
    print(f"\nwrote {JSON}")


# ---------------------------------------------------------------------------------------------
# LaTeX emission, for callers that typeset these. Generated from the same dict as canon.json,
# so a table cannot disagree with it.


FAMNAME = dict(chart="chart", counting="counting", spatial="spatial",
               tracking="tracking$^\\dagger$", glyph="glyph 3\\,px$^\\dagger$")
MODNAME = dict(q3b="Qwen2.5-VL-3B", q7b="Qwen2.5-VL-7B", ivl="InternVL3-2B", smol="SmolVLM")


def _pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


DECOMP = "runs/p0_3b.json"


BUDGET_PROBE = "runs/datasize_3b.json"
BUDGET_LORA = {20: "runs/lora20_3b_ep*.json", 40: "runs/lora40_3b_ep*.json",
               165: "runs/lora_3b_ep*.json"}


def budget():
    """The label-budget comparison: a probe and LoRA at 20, 40 and 165 labels per family.

    Both columns were typed into the manuscript, which is how one of them went wrong. The probe
    curve pools item-weighted over the four families and reproduces what was typed. The LoRA
    column did not: at 20 labels the manuscript quoted the best of three epochs, 75.2, where the
    prespecified last-epoch rule of the robustness appendix gives 74.2. That is the selection
    rule the paper argues against, surviving here because these numbers sat outside this file's
    coverage. The peak is returned alongside so the correction can be stated rather than hidden.
    """
    if not os.path.exists(BUDGET_PROBE):
        return None
    d = json.load(open(BUDGET_PROBE))
    fams = sorted(d)
    nt = {f: d[f]["n_test"] for f in fams}
    tot = sum(nt.values())
    rows = []
    for b in sorted(BUDGET_LORA):
        probe = sum(next(p["mean"] for p in d[f]["curve"] if p["n"] == b) * nt[f]
                    for f in fams) / tot
        eps = {}
        for f in glob.glob(BUDGET_LORA[b]):
            e = int(re.search(r"_ep(\d+)\.json$", f).group(1))
            j = json.load(open(f))
            eps[e] = (j.get("acc") or j["final"])["ALL"]
        if not eps:
            return None
        # at the top budget the cap is the family's own training split, which the rare-class
        # filter has already shortened -- 161 chart items, not 165, hence 656 and not 660
        items = sum(min(b, d[f]["n_train"]) for f in fams)
        rows.append(dict(labels=b, n_train=items, probe=float(probe),
                         lora=float(eps[max(eps)]), lora_peak=float(max(eps.values())),
                         epoch=max(eps), epochs=len(eps)))
    return rows


SPLITS = "runs/splits.json"


def splits(rows):
    """run/splits.py's table: the whole verdict re-derived on resampled splits, per condition.

    Reported per condition rather than as one stability number, because the conditions do not
    fail together: the probe, the presence test and the null hold in every split of every locus,
    and it is the two comparative conditions -- the paired test and the counterfactual bound --
    that move, both of them limited by n rather than by the size of the effect.
    """
    if not os.path.exists(SPLITS):
        return None, None
    d = json.load(open(SPLITS))
    lab = {(r["tag"], r["family"]): (r["label"], FAMNAME.get(r["family"], r["family"]))
           for r in rows}
    body = []
    for k, rs in d.items():
        tag, fam = k.split("/")
        model, famn = lab.get((tag, fam), (tag, fam))
        S = len(rs)
        pr = [100 * r["probe"] for r in rs]
        body.append(
            f"{model} & {famn} & \\textbf{{{sum(r['readout'] for r in rs)}/{S}}} "
            f"& {min(pr):.0f}--{max(pr):.0f} "
            f"& {sum(r['g1'] for r in rs)}/{S} "
            f"& {sum(r['p_null'] < NULL_ALPHA for r in rs)}/{S} "
            f"& {sum(r['p_paired'] < 0.05 for r in rs)}/{S} "
            f"& {sum(r['joint_lb'] > FOLLOW_MIN for r in rs)}/{S}")
    return body, d


HEADREAD = "runs/headread.json"


def headread(rows):
    """run/headread.py's table: the model's own head given the probe's answer set.

    The exact rows are the families whose answers have distinct first tokens; the rest are
    scored on the first token alone and marked, because the state is captured at the final
    prompt token and a one-step read cannot separate "20" from "25". Labels come from the
    canonical rows so this table names cells the way Tables 4 and 6 do.
    """
    if not os.path.exists(HEADREAD):
        return None, None
    d = json.load(open(HEADREAD))
    lab = {(r["tag"], r["family"]): (r["label"], FAMNAME.get(r["family"], r["family"]))
           for r in rows}
    body = []
    for r in d:
        model, fam = lab.get((r["tag"], r["family"]), (r["tag"], r["family"]))
        body.append(f"{model} & {fam} & {r['n']} & {r['classes']} & {_pct(r['model'])} "
                    f"& {_pct(r['head'])} & {_pct(r['probe'])} & "
                    + ("exact" if r["exact"] else f"first token ({r['head_classes']})"))
    return body, d


def decomp(synth):
    """S7's locus decomposition, from run/p0.py's artefact instead of typed into main.tex.

    Four of the five rows are p0's own accuracy table, so the base row is the same measurement
    as Table 3's model column by construction (manifest.py asserts that shared source). The
    probe row is the canonical Qwen-3B probe: its per-family cells come from cells_synthetic
    and its ALL from p0's pooled `readout` over the same 298 items.
    """
    d = json.load(open(DECOMP))
    acc, fams = d["acc"], ("chart", "counting", "spatial", "tracking")
    probe = {f: next(r["probe"] for r in synth if r["tag"] == "3b" and r["family"] == f)
             for f in fams}
    probe["ALL"] = d["readout"]
    # the two single-stack rows are bolded: the point of the table is that either stack alone
    # recovers most of what both together do
    plan = [("base model", acc["base"], False),
            ("supervised linear probe, VLM weights frozen", probe, False),
            ("LoRA \\texttt{lang} (vision frozen)", acc["lang"], True),
            ("LoRA \\texttt{vis} (language frozen)", acc["vis"], True),
            ("LoRA \\texttt{both} (reference)", acc["both"], False)]
    rows = []
    for name, a, bold in plan:
        cells = [f"{100 * a[f]:.1f}" for f in fams]
        allc = f"{100 * a['ALL']:.1f}"
        rows.append(f"{name} & " + " & ".join(cells) + " & "
                    + (f"\\textbf{{{allc}}}" if bold else allc))
    return rows, d["n"]


def _p(x):
    return "--" if x is None else ("$<$0.001" if x < 1e-3 else f"{x:.3f}")


def _tab(path, colspec, header, rows, pre=""):
    """A complete, self-contained tabular. Emitting the whole environment (rather than rows for
    a caller to \\input inside one) avoids booktabs' lookahead colliding with \\input in
    alignment context, and means a table's column count can change without editing main.tex."""
    with open(path, "w") as f:
        f.write("% generated by run/canon.py -- do not edit\n")
        if pre: f.write(pre + "\n")
        f.write("\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n" % (colspec, header))
        for r in rows: f.write(r + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def emit(synth, real, pred_rows, pred, out):
    os.makedirs(TEX, exist_ok=True)
    gs = g1_sensitivity(synth + real)
    # g1_at() exists so the presence thresholds can be swept; if it ever disagreed with
    # run/nullcal.py's stored verdict it would be a second definition of the gate instead
    bad = [f"{r['tag']}/{r['family']}" for r in synth + real
           if r["nullcal"] is not None and g1_at(r["nullcal"]) != bool(r["nullcal"]["g1"])]
    if bad:
        raise SystemExit(f"g1_at disagrees with nullcal's stored verdict: {bad}")
    # `follow` is the conditional forward rate the prose discusses, `joint` the two-endpoint
    # accuracy the verdict is gated on, and `LB` the joint rate's Wilson lower bound. The gate is
    # applied to the bound, so the column the caption names has to be in the table beside it.
    CELLHDR = (r"Model & Family & $n$ & chance & model & probe & gap & $p$ & follow & joint & LB "
               r"& G1 & locus")
    for name, rows in [("synthetic", synth), ("real", real)]:
        body = []
        for r in rows:
            q = r["paired"] or {}
            g = r["g1"]
            body.append(f"{r['label']} & {FAMNAME.get(r['family'], r['family'])} & {r['n_test']} "
                        f"& {_pct(r['chance'])} & {_pct(r['model_acc'])} & {_pct(r['probe'])} "
                        f"& {r['gap']:+.1f} & {_p(q.get('p'))} & {_pct(r['follow'], 0)} "
                        f"& {_pct(r.get('probe_joint'), 0)} "
                        f"& {_pct(r.get(GATE_CI, [None])[0], 0)} "
                        f"& {'--' if g is None else ('pass' if g['passes'] else 'fail')} "
                        f"& {r'\textbf{readout}' if r['readout'] else '--'}")
        # 13 columns overflow the text block at 3.4pt, which is what the joint and LB columns
        # cost; 2.9pt fits both tables with no overfull box and keeps them visually identical
        _tab(f"{TEX}/cells_{name}.tex", "@{}llrrrrrrrrrrl@{}", CELLHDR, body,
             pre="\\footnotesize\\setlength{\\tabcolsep}{2.9pt}")

    NAME = [("probe_gain", "probe gain (probe $-$ base)", "one probe fit"),
            ("probe_acc", "probe accuracy", "one probe fit"),
            ("base_acc", "base accuracy", "free"),
            ("headroom", "headroom ($1-$base)", "free"),
            ("above_chance", "base $-$ chance", "free")]
    body = []
    for k, n, c in NAME:
        s = pred[k]
        bf = (lambda x: "\\textbf{%s}" % x) if s["p_dm"] < .05 else (lambda x: x)
        body.append(f"{n} & {c} & {s['rho']:+.3f} & {_p(s['p'])} "
                    f"& {bf('%+.3f' % s['rho_dm'])} & {bf(_p(s['p_dm']))}")
    _tab(f"{TEX}/prediction.tex", "@{}llrrrr@{}",
         "\\multirow{2}{*}{Predictor} & \\multirow{2}{*}{cost} "
         "& \\multicolumn{2}{c}{raw} & \\multicolumn{2}{c}{family-demeaned} \\\\\n"
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
         " & & $\\rho$ & $p$ & $\\rho$ & $p$", body)

    body = [f"{MODNAME[r['model']]} & {FAMNAME.get(r['family'], r['family'])} "
            f"& {_pct(r['base'])} & {_pct(r['probe'])} & {_pct(r['lora'])} "
            f"& {100 * (r['probe'] - r['base']):+.1f} & {100 * (r['lora'] - r['base']):+.1f}"
            for r in sorted(pred_rows, key=lambda r: (r["family"], r["model"]))]
    _tab(f"{TEX}/pairs.tex", "@{}llrrrrr@{}",
         "Model & Family & base & probe & fine-tuned & probe gain & FT gain", body)

    body = [f"{b['lo']}--{b['hi']} & {b['n']} & {_pct(b['model'])} & {_pct(b['probe'])} "
            f"& {100 * (b['probe'] - b['model']):+.1f}" for b in out["bands"]]
    _tab(f"{TEX}/bands.tex", "@{}lrrrr@{}", "Value band & $n$ & model & probe & gap", body)

    body, _ = decomp(synth)
    bd = budget()
    if bd is not None:
        _tab(f"{TEX}/budget.tex", "@{}rrrrr@{}",
             r"labels/family & items & probe & LoRA & LoRA $-$ probe",
             [f"{r['labels']} & {r['n_train']} & {100 * r['probe']:.1f} & "
              f"{100 * r['lora']:.1f} & "
              f"{round(100 * r['lora'], 1) - round(100 * r['probe'], 1):+.1f}" for r in bd])

    sb, sd = splits(synth + real)
    if sb is not None:
        _tab(f"{TEX}/splits.tex", "@{}llrrrrrr@{}",
             "Model & Family & locus & probe & presence & null & paired & bound", sb,
             pre="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    hb, hd = headread(synth + real)
    if hb is not None:
        _tab(f"{TEX}/headread.tex", "@{}llrrrrrl@{}",
             "Model & Family & $n$ & $k$ & model & head & probe & comparison", hb,
             pre="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    _tab(f"{TEX}/decomp.tex", "@{}lrrrrr@{}",
         "Configuration & chart & counting & spatial & tracking & ALL", body)

    # single-source-of-truth macros for numbers quoted in running prose
    with open(f"{TEX}/facts.tex", "w") as f:
        f.write("% generated by run/canon.py -- do not edit\n")
        ag, lb = out["agreement"] or {}, pred["lower_bound"]
        rc = [r for r in real if r["family"] == "chart"]
        gl = [r for r in real if r["family"] == "glyph" and r["nullcal"]]
        fa = {"AgreeMin": f"{100 * min(v['agree'] for v in ag.values()):.1f}",
              "AgreeMax": f"{100 * max(v['agree'] for v in ag.values()):.1f}",
              "NPairs": str(len(pred_rows)),
              "LbAll": f"{lb['all'][0]}/{lb['all'][1]}",
              "LbInf": f"{lb['informative'][0]}/{lb['informative'][1]}",
              "RhoGain": f"{pred['probe_gain']['rho_dm']:+.3f}",
              "PGain": f"{pred['probe_gain']['p_dm']:.4f}",
              "RhoGainRaw": f"{pred['probe_gain']['rho']:+.3f}",
              # the free predictor that looks significant raw and does not survive demeaning; the
              # prose names the coefficient, so it is generated rather than typed
              "RhoAboveChanceRaw": f"{pred['above_chance']['rho']:+.3f}",
              "PAboveChanceRaw": f"{pred['above_chance']['p']:.3f}",
              "PGainRaw": f"{pred['probe_gain']['p']:.4f}",
              "ReadoutReal": str(sum(r["readout"] for r in real)),
              "NReal": str(len(real)),
              "ReadoutSynth": str(sum(r["readout"] for r in synth)),
              "NSynth": str(len(synth)),
              "ProbeChartMin": f"{100 * min(r['probe'] for r in rc if r['readout']):.1f}",
              "ProbeChartMax": f"{100 * max(r['probe'] for r in rc if r['readout']):.1f}",
              # same population as ProbeChartMin/Max: the cells with a readout locus. Without
              # the filter the two ranges were drawn from different sets of cells.
              "ModelChartMin": f"{100 * min(r['model_acc'] for r in rc if r['readout']):.1f}",
              "ModelChartMax": f"{100 * max(r['model_acc'] for r in rc if r['readout']):.1f}",
              # the calibrated null, and the false-positive rate measured on the family whose
              # attribute is absent by construction
              "NullQMin": f"{100 * min(r['nullcal']['null_q95'] for r in real + synth if r['nullcal']):.1f}",
              "NullQMax": f"{100 * max(r['nullcal']['null_q95'] for r in real + synth if r['nullcal']):.1f}",
              "NullPerm": str(next(r['nullcal']['nperm'] for r in real if r['nullcal'])),
              # the test's own parameters. The manuscript states the decision rule, so alpha and
              # the attainable p-floor are generated here rather than typed: the rule is
              # p_null < NULL_ALPHA, and with B draws no p smaller than 1/(B+1) exists.
              "NullAlpha": f"{NULL_ALPHA:g}",
              "NullPFloor": f"1/{next(r['nullcal']['nperm'] for r in real if r['nullcal']) + 1}",
              # what a layer search buys on the designed control, over reading the final layer
              "PeakPremium": f"{np.mean([100 * (r['peak'] - r['probe']) for r in gl]):.1f}",
              "FollowShiftMax": f"{max(abs(100 * (r['probe_follow'] - r['probe_follow_sel'])) for r in real + synth if r.get('probe_follow_sel') == r.get('probe_follow_sel')):.0f}",
              # follow-rate ranges quoted in prose, so they cannot drift from the tables
              "FollowChartMin": f"{min(100 * r['follow'] for r in rc if r['readout']):.0f}",
              "FollowChartMax": f"{max(100 * r['follow'] for r in rc if r['readout']):.0f}",
              "FollowCountMin": f"{min(100 * r['follow'] for r in real if r['family'] == 'counting'):.0f}",
              "FollowCountMax": f"{max(100 * r['follow'] for r in real if r['family'] == 'counting'):.0f}",
              "FollowSynthChart": f"{100 * next(r['follow'] for r in synth if r['tag'] == '3b' and r['family'] == 'chart'):.0f}",
              "FollowSynthSpatial": f"{100 * next(r['follow'] for r in synth if r['tag'] == 'smolm' and r['family'] == 'spatial'):.0f}",
              # the individual cells the prose names. These were typed by hand and seven of
              # them survived Stage 2's move to the final-layer estimator as stale numbers
              # contradicting the tables on the same page; they are generated now.
              "FollowSynthSpatialQwen": f"{100 * next(r['follow'] for r in synth if r['tag'] == '3b' and r['family'] == 'spatial'):.0f}",
              "FollowSynthTracking": f"{100 * next(r['follow'] for r in synth if r['tag'] == '3b' and r['family'] == 'tracking'):.0f}",
              "FollowSynthChartSmol": f"{100 * next(r['follow'] for r in synth if r['tag'] == 'smolm' and r['family'] == 'chart'):.0f}",
              "FollowChartNfSmall": f"{100 * next(r['follow'] for r in real if r['tag'] == 'q3b4_real_chart_v2'):.0f}",
              "FollowChartNfLarge": f"{100 * next(r['follow'] for r in real if r['tag'] == 'q7b_real_chart_v2'):.0f}",
              "FollowRealChartSmol": f"{100 * next(r['follow'] for r in real if r['tag'] == 'smol_real_chart_v2'):.0f}",
              "PeakPremiumMin": f"{min(100 * (r['peak'] - r['probe']) for r in gl):.1f}",
              "PeakPremiumMax": f"{max(100 * (r['peak'] - r['probe']) for r in gl):.1f}",
              "FprCtrl": f"{sum(1 for r in gl if r['nullcal']['p_null'] < NULL_ALPHA)}/{len(gl)}",
              # A 1-of-5 rate is not evidence of miscalibration and must not be reported as
              # though it were: state what a nominal test predicts, and the interval five cells
              # can actually support.
              "FprExp": f"{len(gl) * NULL_ALPHA:.2f}",
              "FprP": f"{binomtest(sum(1 for r in gl if r['nullcal']['p_null'] < NULL_ALPHA), len(gl), NULL_ALPHA).pvalue:.2f}",
              "FprCiLo": f"{100 * binomtest(sum(1 for r in gl if r['nullcal']['p_null'] < NULL_ALPHA), len(gl), NULL_ALPHA).proportion_ci(0.95).low:.1f}",
              "FprCiHi": f"{100 * binomtest(sum(1 for r in gl if r['nullcal']['p_null'] < NULL_ALPHA), len(gl), NULL_ALPHA).proportion_ci(0.95).high:.0f}",
              # the composed verdict, which is what the protocol actually issues
              "FprVerdict": f"{sum(1 for r in gl if r['readout'])}/{len(gl)}",
              # follow-rate lower bounds: the loci clear the gate by these margins
              "FollowLbMin": f"{100 * min(r['probe_follow_ci'][0] for r in synth + real if r['readout']):.0f}",
              "FollowLbMax": f"{100 * max(r['probe_follow_ci'][0] for r in synth + real if r['readout']):.0f}",
              # the label-budget comparison. Names carry no digits, so the budgets are words.
              **({} if bd is None else {
                  f"Budget{w}{k}": f"{100 * r[v]:.1f}"
                  for w, r in zip(("Twenty", "Forty", "Full"), bd)
                  for k, v in [("Probe", "probe"), ("Lora", "lora")]}),
              **({} if bd is None else {
                  # what the oracle epoch rule would have reported at the tightest budget, which
                  # is the second instance of the effect the robustness appendix argues from
                  "BudgetLoraTwentyPeak": f"{100 * bd[0]['lora_peak']:.1f}",
                  "BudgetLoraTwentyEpochs": str(bd[0]["epochs"])}),
              # the repeated-split analysis. The point is that the conditions do not fail
              # together, so the macros separate the ones that hold in every split from the two
              # that do not, and name the control result the calibration claim rests on.
              **({} if sd is None else {
                  "SplitS": str(len(next(iter(sd.values())))),
                  "SplitSynLoci": "/".join(
                      str(x) for x in [min(sum(r["readout"] for r in rs) for k, rs in sd.items()
                                           if k in ("3b/chart", "3b/spatial")),
                                       len(next(iter(sd.values())))]),
                  # the real-image chart cells, selected against the canonical tag list rather
                  # than by matching tag spellings -- the spelling changed once and this broke
                  **{f"SplitReal{e}": str(f([sum(r["readout"] for r in rs)
                                             for k, rs in sd.items()
                                             if k.endswith("/chart")
                                             and k.split("/")[0] in {t for t, _, _ in REAL}]))
                     for e, f in [("Min", min), ("Max", max)]},
                  "SplitPairedMin": str(min(sum(r["p_paired"] < 0.05 for r in rs)
                                            for k, rs in sd.items() if "glyph" not in k)),
                  "SplitBoundMin": str(min(sum(r["joint_lb"] > FOLLOW_MIN for r in rs)
                                           for k, rs in sd.items() if "glyph" not in k)),
                  "SplitCtrlLoci": str(sum(sum(r["readout"] for r in rs)
                                           for k, rs in sd.items() if "glyph" in k)),
                  "SplitCtrlN": str(sum(len(rs) for k, rs in sd.items() if "glyph" in k)),
                  "SplitProbeMin": f"{100 * min(r['probe'] for k, rs in sd.items() if 'glyph' not in k for r in rs):.0f}",
                  "SplitProbeMax": f"{100 * max(r['probe'] for k, rs in sd.items() if 'glyph' not in k for r in rs):.0f}",
                  # on the canonical split every locus's bound sits above its resampled mean,
                  # which the manuscript states rather than leaves for a reviewer to find
                  "SplitSeedAbove": str(sum(
                      1 for k, rs in sd.items() if "glyph" not in k
                      and next(r["joint_lb"] for r in rs if r["seed"] == 0)
                      > float(np.mean([r["joint_lb"] for r in rs])))),
                  "SplitLoci": str(sum(1 for k in sd if "glyph" not in k))}),
              # the constrained-head comparison. Its point is which gaps close when the model's
              # own head is given the probe's candidate set and which do not, so the macros are
              # the three readouts side by side on the cells where that question is decided.
              **({} if hd is None else {
                  **{f"HeadCount{k}{e}": f"{100 * f([r[v] for r in hd if r['family'] == 'counting' and r['exact']]):.1f}"
                     for k, v in [("Model", "model"), ("Head", "head"), ("Probe", "probe")]
                     for e, f in [("Min", min), ("Max", max)]},
                  **{f"HeadSpat{k}{e}": f"{100 * f([r[v] for r in hd if r['family'] == 'spatial' and r['tag'] in ('3b', 'smolm')]):.1f}"
                     for k, v in [("Model", "model"), ("Head", "head"), ("Probe", "probe")]
                     for e, f in [("Min", min), ("Max", max)]},
                  **{f"HeadChartSyn{k}": f"{100 * next(r[v] for r in hd if r['tag'] == '3b' and r['family'] == 'chart'):.1f}"
                     for k, v in [("Model", "model"), ("Head", "head"), ("Probe", "probe")]},
                  **{f"HeadChartReal{k}{e}": f"{100 * f([r[v] for r in hd if r['family'] == 'chart' and r['tag'] in ('realchart_v2', 'ivl_real_chart_v2')]):.1f}"
                     for k, v in [("Head", "head"), ("Probe", "probe")]
                     for e, f in [("Min", min), ("Max", max)]},
                  "HeadExact": str(sum(1 for r in hd if r["exact"])),
                  "HeadCells": str(len(hd))}),
              # how many adaptation runs stand behind the pairs. Sixteen (model, family) pairs
              # come from fewer runs than that, because three real families are trained jointly
              # and chart alone, and the manuscript never said so.
              **{k: str(len({r[f] for r in json.load(open("runs/p3_lora_matched.json"))}))
                 for k, f in [("AdaptRuns", "tag"), ("AdaptConfigs", "model")]},
              # the probe's effective rank. The pipeline asks for 64 components but fits
              # min(64, n_train - 1), so on a small family the requested capacity is not the
              # delivered capacity, and the manuscript's "capacity is held fixed" needs the
              # exception named rather than assumed.
              "PcaRankMin": str(min(min(64, r['nullcal']['n_train'] - 1)
                                    for r in real + synth if r['nullcal'])),
              "PcaCapped": str(sum(1 for r in real + synth
                                   if r['nullcal'] and r['nullcal']['n_train'] - 1 < 64)),
              "PcaCells": str(sum(1 for r in real + synth if r['nullcal'])),
              # the presence test's three free constants, and how far each can move before a
              # verdict does. Macro names carry no digits, so G1 is spelled out.
              "GOneBlindHi": f"{100 * gs['blind']['hi']:.0f}",
              "GOneShufHi": f"{100 * gs['shuf']['hi']:.0f}",
              "GOnePosHi": f"{gs['pos']['hi']:.2f}",
              # how many of the real chart configurations are loci, and the near miss among
              # them. The manuscript said "four of five" for two revisions; it is generated now
              # so a recount cannot be missed in prose the way it was.
              # the same cells under the selected-layer estimator, for the robustness appendix
              **{f"FollowChartSel{e}": f"{f([100 * r['probe_follow_sel'] for r in rc if r['readout']]):.0f}"
                 for e, f in [("Min", min), ("Max", max)]},
              # out-of-sample, with the designed control excluded; and the single fold that
              # drives the difference when it is included
              **{f"OosNoCtrl{a}{b}": f"{v:+.2f}" if b == "Delta" else f"{v:.2f}"
                 for a, k in [("Model", "heldout_model"), ("Fam", "heldout_family")]
                 for b, v in [("Delta", pred['oos_noctrl'][k]['delta']),
                              ("P", pred['oos_noctrl'][k]['p_paired'])]},
              "OosCtrlFold": f"{pred['oos']['heldout_family']['per_fold']['glyph']:+.1f}",
              # the precision contrast on identical weights and identical items: with the
              # larger chart set it no longer widens the gap, it decides the verdict, so the
              # four numbers the claim rests on are generated rather than typed
              **{f"Quant{w}{p_}": f"{100 * next(r[k] for r in rc if r['label'] == lbl):.1f}"
                 for w, k in [("Model", "model_acc"), ("Probe", "probe")]
                 for p_, lbl in [("Bf", "Qwen-3B bf16"), ("Nf", "Qwen-3B nf4")]},
              **{f"ScaleGap{e}": f"{next(r['gap'] for r in rc if r['label'] == lbl):+.1f}"
                 for e, lbl in [("Small", "Qwen-3B nf4"), ("Large", "Qwen-7B nf4")]},
              "ChartLoci": str(sum(1 for r in rc if r['readout'])),
              "ChartCells": str(len(rc)),
              **{f"ChartNear{k}": v for k, v in [
                  ("Gap", f"{next(r['gap'] for r in rc if not r['readout'] and r['gap'] > 0):+.1f}"),
                  ("P", f"{next(r['paired']['p'] for r in rc if not r['readout'] and r['gap'] > 0):.2f}"),
                  ("Lb", f"{100 * next(r[GATE_CI][0] for r in rc if not r['readout'] and r['gap'] > 0):.0f}")]},
              # the gated statistic: the loci's range, and the highest cell that is not a locus.
              # The gap between them is what makes the threshold's exact position immaterial.
              "JointLbMin": f"{100 * min(r[GATE_CI][0] for r in synth + real if r['readout']):.0f}",
              "JointLbMax": f"{100 * max(r[GATE_CI][0] for r in synth + real if r['readout']):.0f}",
              # the nearest miss *on this axis*: the highest bound among cells that satisfy every
              # other condition and fail only the bound. Comparing against all non-loci was
              # wrong once a cell failed the paired test while reading a high bound -- that says
              # nothing about whether the threshold is well placed, which is the claim it backs.
              "JointLbNonMax": f"{100 * max([r[GATE_CI][0] for r in synth + real
                                             if not r['readout'] and locus(r, follow=0.0)],
                                            default=float('nan')):.0f}",
              # how much of each conditional forward denominator the class support removes, on the
              # families where it bites: predict() cannot emit a count the training split never
              # held, nor a bar value of 100 that no original chart carries
              "UnsupCountMin": str(min(r['probe_follow_den'] - r['probe_follow_sup_den']
                                       for r in synth + real if r['family'] == 'counting')),
              "UnsupCountMax": str(max(r['probe_follow_den'] - r['probe_follow_sup_den']
                                       for r in synth + real if r['family'] == 'counting')),
              "UnsupCountPctMax": f"{100 * max((r['probe_follow_den'] - r['probe_follow_sup_den']) / r['probe_follow_den'] for r in synth + real if r['family'] == 'counting'):.0f}",
              "UnsupChartMax": str(max(r['probe_follow_den'] - r['probe_follow_sup_den']
                                       for r in real if r['family'] == 'chart')),
              # the one cell whose verdict depends on which counterfactual statistic is used
              **{f"Flip{k}": f"{100 * next(r[v][0] for r in synth if r['tag'] == '3b' and r['family'] == 'counting'):.0f}"
                 for k, v in [("Fwd", "probe_follow_ci"), ("Sup", "probe_follow_sup_ci"),
                              ("Rev", "probe_reverse_ci"), ("Joint", "probe_joint_ci")]},
              # and its like-for-like comparison against the model, on one shared denominator
              **{f"Flip{k}": str(next(r[v] for r in synth
                                      if r['tag'] == '3b' and r['family'] == 'counting'))
                 for k, v in [("MatchDen", "matched_den"), ("MatchProbe", "probe_matched_num"),
                              ("MatchModel", "model_matched_num")]},
              "FollowDenMin": str(min(r['probe_follow_den'] for r in real
                                      if r['family'].startswith('glyph'))),
              "FollowDenMax": str(max(r['probe_follow_den'] for r in real
                                      if r['family'].startswith('glyph'))),
              # multiplicity
              # counting is the protocol's cleanest negative and the claim is exceptionless, so
              # it is counted rather than described with a range that held only for the real cells
              "CountCells": str(sum(1 for r in synth + real if r['family'] == 'counting')),
              "CountNullClear": str(sum(1 for r in synth + real if r['family'] == 'counting'
                                        and r['nullcal']['p_null'] < NULL_ALPHA)),
              "CountLoci": str(sum(1 for r in synth + real
                                   if r['family'] == 'counting' and r['readout'])),
              "CountFollowMin": f"{100 * min(r['follow'] for r in synth + real if r['family'] == 'counting'):.0f}",
              "CountFollowMax": f"{100 * max(r['follow'] for r in synth + real if r['family'] == 'counting'):.0f}",
              "MultN": str(out['multiplicity']['n']),
              "FollowStableLo": str(out['follow_sensitivity']['lo']),
              "FollowStableHi": str(out['follow_sensitivity']['hi']),
              "LociBh": f"{len(out['multiplicity']['loci_bh'])}/{len(out['multiplicity']['loci'])}",
              "LociHolm": f"{len(out['multiplicity']['loci_holm'])}/{len(out['multiplicity']['loci'])}",
              # out-of-sample: the interval, not just the direction
              "OosModelDelta": f"{pred['oos']['heldout_model']['delta']:+.2f}",
              "OosModelP": f"{pred['oos']['heldout_model']['p_paired']:.2f}",
              "OosModelCiLo": f"{pred['oos']['heldout_model']['ci'][0]:+.1f}",
              "OosModelCiHi": f"{pred['oos']['heldout_model']['ci'][1]:+.1f}",
              "OosFamDelta": f"{pred['oos']['heldout_family']['delta']:+.2f}",
              "OosFamP": f"{pred['oos']['heldout_family']['p_paired']:.2f}",
              "OosFamCiLo": f"{pred['oos']['heldout_family']['ci'][0]:+.1f}",
              "OosFamCiHi": f"{pred['oos']['heldout_family']['ci'][1]:+.1f}",
              "OosModel": f"{pred['oos']['heldout_model']['mae_baseline']:.1f}",
              "OosModelProbe": f"{pred['oos']['heldout_model']['mae_with_probe']:.1f}",
              "OosFam": f"{pred['oos']['heldout_family']['mae_baseline']:.1f}",
              "OosFamProbe": f"{pred['oos']['heldout_family']['mae_with_probe']:.1f}",
              "InsampleF": f"{pred['oos']['insample']['F']:.2f}",
              "InsampleP": f"{pred['oos']['insample']['p']:.3f}",
              "CouplingP": f"{pred['coupling_null']['p']:.4f}",
              "LbStrict": f"{lb['strict'][0]}/{lb['strict'][1]}",
              "LbHelped": f"{lb['helped'][0]}/{lb['helped'][1]}"}
        for k, v in fa.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(f"wrote {TEX}/{{cells_synthetic,cells_real,prediction,pairs,bands,facts}}.tex")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true",
                   help="print the two estimators side by side for every cell")
    main(p.parse_args())
