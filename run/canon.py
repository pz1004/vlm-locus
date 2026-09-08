"""Every number in the manuscript, from ONE probe protocol.

The study accumulated two probe pipelines. `run/fit_probes.py` selects a layer on a held-out
selection split and serialises the probe for downstream consumers (cfprobe, lora comparison);
`run/layers.py` fits the same capacity-controlled probe at every layer with fixed
hyperparameters and reports the FINAL layer. They disagree in 9 of 28 (dataset, family) cells,
by up to 21 pp -- not a bug in either, but two different estimators, and a paper cannot ship
both.

The manuscript uses `layers.py` at the final layer throughout, for three reasons:
  1. it is the quantity the readout claim is *about* -- the vector the LM head itself consumes,
     with no layer search to defend;
  2. it is the more conservative of the two on the headline cells and never benefits from
     best-of-L selection, which the 3 px glyph control prices at 8.5 pp (peak) vs 2.5 pp (final);
  3. it gives the *stronger* prediction result (lower bound 12/12 vs 11/12), so the choice is
     not self-serving in the direction of the claim.
The four headline chart cells are identical to 0.0 pp under either estimator, so no conclusion
depends on this decision; `--audit` prints the full disagreement table so a reader can check.

Counterfactual follow rates come from `cfprobe_*.json`, which was run with the serialised
(selected-layer) probe. That is stated in the manuscript rather than papered over: re-running
the counterfactual pass with the final-layer probe needs the GPU, and the follow rate is a
*conditional* statistic whose role is to separate ~85% from ~20%, a distinction no 3 pp shift in
probe accuracy can touch.
"""
from __future__ import annotations
import argparse, glob, json, os

# Output locations. Defaults keep everything inside the repository so a fresh
# clone runs standalone; the manuscript build overrides them to write straight
# into paper/.
JSON = os.environ.get("VLM_LOCUS_JSON", "out/canon.json")
TEX  = os.environ.get("VLM_LOCUS_TEX",  "out/tables")
import numpy as np
from scipy.stats import spearmanr, binomtest

# (tag, model label, dataset label) -> the layer sweep and its counterfactual pass
REAL = [("real",            "q3b",  "Qwen-3B bf16"),
        ("realchart",       "q3b",  "Qwen-3B bf16"),
        ("q3b4_real_3b",    "q3b4", "Qwen-3B nf4"),
        ("q3b4_real_chart", "q3b4", "Qwen-3B nf4"),
        ("q7b_real_3b",     "q7b",  "Qwen-7B nf4"),
        ("q7b_real_chart",  "q7b",  "Qwen-7B nf4"),
        ("ivl_real_3b",     "ivl",  "InternVL3-2B"),
        ("ivl_real_chart",  "ivl",  "InternVL3-2B"),
        ("smol_real_3b",    "smol", "SmolVLM"),
        ("smol_real_chart", "smol", "SmolVLM")]
SYNTH = [("3b", "q3b", "Qwen-3B"), ("smolm", "smol", "SmolVLM")]

# the layer-selection artefact, priced on the designed negative control (3 px glyph, attribute
# provably absent): best-of-L yields ~8.5 pp of apparent gap, the final layer ~2.5 pp.
ARTEFACT_FINAL, FOLLOW_MIN = 2.5, 50.0


def follow(tag):
    """probe/model accuracy before the edit and follow rate, per family, from the paired pass."""
    f = f"runs/cfprobe_{tag}.json"
    if not os.path.exists(f): return {}
    out = {}
    for fam in {r["family"] for r in json.load(open(f))}:
        g = [r for r in json.load(open(f)) if r["family"] == fam]
        def st(pre, post):
            ok = [r for r in g if r[pre] == r["a0"]]
            return (float(np.mean([r[pre] == r["a0"] for r in g])),
                    float(np.mean([r[post] == r["a1"] for r in ok])) if ok else float("nan"))
        p0, pf = st("probe0", "probe1"); m0, mf = st("model0", "model1")
        out[fam] = dict(n=len(g), probe_acc0=p0, probe_follow=pf, model_acc0=m0, model_follow=mf)
    return out


G1_FILE = {"3b": "runs/probe_g1.json"}   # the first sweep predates the per-tag naming


def g1(tag, fam):
    f = G1_FILE.get(tag, f"runs/probe_g1_{tag}.json")
    if not os.path.exists(f): return None
    d = json.load(open(f)).get(fam)
    if d is None: return None
    # G1: beat blindfold by >5 pp AND the shuffled-label control by >10 pp
    d = dict(d, passes=bool(100 * (d["vis"] - d["blind"]) > 5
                            and 100 * (d["vis"] - d["shuffled"]) > 10))
    return d


# the scored generations each layer sweep was aligned to, for the paired probe-vs-model test
GEN = {"3b": "runs/branches6_test.jsonl", "smolm": "runs/smolm_gen.jsonl"}


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


def bands(tag="realchart", fam="chart", edges=(0, 25, 50, 75, 101)):
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
            rows.append(dict(tag=tag, model=model, label=label, family=fam,
                             n_test=v["n_test"], chance=v["chance"], model_acc=v["model"],
                             model_all=v["model_all"], probe=v["final"], peak=v["peak"],
                             peak_layer=v["peak_layer"], gap=gap, follow=fl,
                             curve=v["vis"], blindcurve=v["blind"],
                             g1=g1(tag, fam), **{k: v2 for k, v2 in F.get(fam, {}).items()},
                             paired=paired(tag, fam),
                             readout=bool(gap > ARTEFACT_FINAL
                                          and not np.isnan(fl) and 100 * fl > FOLLOW_MIN)))
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
    lg = [r["lora"] - r["base"] for r in rows]
    cand = {"probe_gain":  [r["probe"] - r["base"] for r in rows],
            "probe_acc":   [r["probe"] for r in rows],
            "base_acc":    [r["base"] for r in rows],
            "headroom":    [1 - r["base"] for r in rows],
            "above_chance":[r["base"] - r["chance"] for r in rows]}
    stats = {}
    for k, v in cand.items():
        r1, p1 = spearmanr(v, lg); r2, p2 = spearmanr(dm(v), dm(lg))
        stats[k] = dict(rho=float(r1), p=float(p1), rho_dm=float(r2), p_dm=float(p2))
    inf = [r for r in rows if r["family"] != "glyph"]
    stats["lower_bound"] = dict(
        all=[int(sum(r["probe"] <= r["lora"] + 1e-9 for r in rows)), len(rows)],
        informative=[int(sum(r["probe"] <= r["lora"] + 1e-9 for r in inf)), len(inf)])
    return rows, stats


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
               bands=bands(),
               agreement=agreement(["realchart", "q3b4_real_chart", "q7b_real_chart",
                                    "ivl_real_chart"], "chart"),
               protocol=dict(probe="layers.py final layer, fixed hyperparameters",
                             artefact_final_pp=ARTEFACT_FINAL, follow_min_pct=FOLLOW_MIN))
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
# LaTeX emission. The manuscript \inputs these, so a table cannot disagree with canon.json.


FAMNAME = dict(chart="chart", counting="counting", spatial="spatial",
               tracking="tracking$^\\dagger$", glyph="glyph 3\\,px$^\\dagger$")
MODNAME = dict(q3b="Qwen2.5-VL-3B", q7b="Qwen2.5-VL-7B", ivl="InternVL3-2B", smol="SmolVLM")


def _pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def _p(x):
    return "--" if x is None else ("$<$0.001" if x < 1e-3 else f"{x:.3f}")


def _tab(path, colspec, header, rows, pre=""):
    """A complete, self-contained tabular. Emitting the whole environment (rather than rows for
    the manuscript to \\input inside one) avoids booktabs' lookahead colliding with \\input in
    alignment context, and means a table's column count can change without editing main.tex."""
    with open(path, "w") as f:
        f.write("% generated by run/canon.py -- do not edit\n")
        if pre: f.write(pre + "\n")
        f.write("\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n" % (colspec, header))
        for r in rows: f.write(r + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def emit(synth, real, pred_rows, pred, out):
    os.makedirs(TEX, exist_ok=True)
    CELLHDR = (r"Model & Family & $n$ & chance & model & probe & gap & $p$ & follow & G1 "
               r"& locus")
    for name, rows in [("synthetic", synth), ("real", real)]:
        body = []
        for r in rows:
            q = r["paired"] or {}
            g = r["g1"]
            body.append(f"{r['label']} & {FAMNAME.get(r['family'], r['family'])} & {r['n_test']} "
                        f"& {_pct(r['chance'])} & {_pct(r['model_acc'])} & {_pct(r['probe'])} "
                        f"& {r['gap']:+.1f} & {_p(q.get('p'))} & {_pct(r['follow'], 0)} "
                        f"& {'--' if g is None else ('pass' if g['passes'] else 'fail')} "
                        f"& {r'\textbf{readout}' if r['readout'] else '--'}")
        _tab(f"{TEX}/cells_{name}.tex", "@{}llrrrrrrrrl@{}", CELLHDR, body,
             pre="\\footnotesize\\setlength{\\tabcolsep}{3.4pt}")

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

    # single-source-of-truth macros for numbers quoted in running prose
    with open(f"{TEX}/facts.tex", "w") as f:
        f.write("% generated by run/canon.py -- do not edit\n")
        ag, lb = out["agreement"] or {}, pred["lower_bound"]
        rc = [r for r in real if r["family"] == "chart"]
        fa = {"AgreeMin": f"{100 * min(v['agree'] for v in ag.values()):.1f}",
              "AgreeMax": f"{100 * max(v['agree'] for v in ag.values()):.1f}",
              "NPairs": str(len(pred_rows)),
              "LbAll": f"{lb['all'][0]}/{lb['all'][1]}",
              "LbInf": f"{lb['informative'][0]}/{lb['informative'][1]}",
              "RhoGain": f"{pred['probe_gain']['rho_dm']:+.3f}",
              "PGain": f"{pred['probe_gain']['p_dm']:.4f}",
              "RhoGainRaw": f"{pred['probe_gain']['rho']:+.3f}",
              "PGainRaw": f"{pred['probe_gain']['p']:.4f}",
              "ReadoutReal": str(sum(r["readout"] for r in real)),
              "NReal": str(len(real)),
              "ReadoutSynth": str(sum(r["readout"] for r in synth)),
              "NSynth": str(len(synth)),
              "ProbeChartMin": f"{100 * min(r['probe'] for r in rc if r['readout']):.1f}",
              "ProbeChartMax": f"{100 * max(r['probe'] for r in rc if r['readout']):.1f}",
              "ModelChartMin": f"{100 * min(r['model_acc'] for r in rc):.1f}",
              "ModelChartMax": f"{100 * max(r['model_acc'] for r in rc):.1f}",
              "ArtefactFinal": f"{ARTEFACT_FINAL:.1f}"}
        for k, v in fa.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(f"wrote {TEX}/{{cells_synthetic,cells_real,prediction,pairs,bands,facts}}.tex")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true",
                   help="print the two estimators side by side for every cell")
    main(p.parse_args())
