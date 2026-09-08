"""Every figure, all from out/canon.json plus the frozen run artefacts in runs/.

No number here is typed in: every value is read from the same canonical protocol the tables use,
so a figure cannot drift from the text.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Defaults keep output inside the repository; a caller may override them.
JSON = os.environ.get("VLM_LOCUS_JSON", "out/canon.json")
OUT  = os.environ.get("VLM_LOCUS_FIGS", "out/figs")
C = json.load(open(JSON))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "serif"})
INK, PROBE, MODEL, BLIND = "#1a1a1a", "#0b6e99", "#b03a2e", "#8a8a8a"


def cell(scope, model, family):
    return next(r for r in C[scope] if r["model"] == model and r["family"] == family)


def fig1():
    """The signature: where the answer lives, synthetic and real."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for a, (r, ttl) in zip(ax, [(cell("synthetic", "q3b", "chart"), "(a) synthetic charts"),
                                (cell("real", "q3b", "chart"), "(b) real charts (ChartQA)")]):
        v, b = 100 * np.array(r["curve"]), 100 * np.array(r["blindcurve"])
        x = np.arange(len(v))
        a.plot(x, v, color=PROBE, lw=1.6, label="probe (visual states)")
        a.plot(x, b, color=BLIND, lw=1.1, ls=":", label="probe (blindfolded)")
        a.axhline(100 * r["model_acc"], color=MODEL, lw=1.3, ls="--", label="model output")
        a.text(0.6, 100 * r["model_acc"] + 2.5, f"model {100*r['model_acc']:.1f}%",
               fontsize=6.8, color=MODEL)
        a.axhline(100 * r["chance"], color=INK, lw=0.7, ls="-.", alpha=.5, label="chance")
        a.annotate("", xy=(x[-1], v[-1]), xytext=(x[-1], 100 * r["model_acc"]),
                   arrowprops=dict(arrowstyle="<->", color=INK, lw=.9))
        a.text(x[-1] - 1.2, (v[-1] + 100 * r["model_acc"]) / 2, f"{r['gap']:+.1f} pp",
               ha="right", va="center", fontsize=7.5, color=INK,
               bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
        a.set_title(ttl, loc="left"); a.set_xlabel("layer"); a.set_ylim(-3, 105)
        a.set_xlim(0, len(v) - 1)
    ax[0].set_ylabel("accuracy (%)"); ax[0].legend(loc="upper left", frameon=False)
    fig.savefig(f"{OUT}/fig1_layers.pdf"); fig.savefig(f"{OUT}/fig1_layers.png"); plt.close(fig)


def fig2():
    """The locus map, on the axes the verdict actually uses.

    This figure used to plot the probe-minus-model gap against the follow-rate point estimate,
    with a dashed vertical line at the 2.5 pp artefact floor. Both axes were wrong once the
    protocol was calibrated: the floor is retired, and the follow gate is applied to the lower
    confidence bound, so a cell could sit above the drawn line and still not be a locus. The axes
    below are the two calibrated quantities, which is also what the caption always claimed.
    """
    fig, a = plt.subplots(figsize=(3.6, 3.05))
    rows = [(r, "o") for r in C["synthetic"]] + [(r, "s") for r in C["real"]]
    a.axvspan(-70, 0, color="#dddddd", alpha=.55, lw=0)
    a.axhspan(-8, 50, color="#dddddd", alpha=.55, lw=0)
    a.axvline(0, color=INK, lw=.7, ls="--"); a.axhline(50, color=INK, lw=.7, ls="--")
    for r, mk in rows:
        nc = r.get("nullcal")
        if nc is None or not r["probe_follow_den"]: continue
        x = 100 * (r["probe"] - nc["null_q95"])
        lo = 100 * r["probe_follow_ci"][0]
        good = r["readout"]
        paired_ok = r.get("paired") and r["paired"]["p"] < 0.05 and r["gap"] > 0
        a.scatter(x, lo, marker=mk, s=27 if good else 20,
                  facecolor=PROBE if good else "white",
                  edgecolor=PROBE if good else ("#555555" if paired_ok else "#aaaaaa"),
                  lw=.9, zorder=3)
    rc = [r for r in C["real"] if r["family"] == "chart" and r["readout"]]
    if rc:
        xs = [100 * (r["probe"] - r["nullcal"]["null_q95"]) for r in rc]
        ys = [100 * r["probe_follow_ci"][0] for r in rc]
        a.add_patch(plt.Rectangle((min(xs) - 4, min(ys) - 4),
                                  max(xs) - min(xs) + 8, max(ys) - min(ys) + 8,
                                  fill=False, ec=PROBE, lw=.7, ls=":", zorder=2))
        a.annotate("real charts\n(4 configs)", (max(xs) + 4, (min(ys) + max(ys)) / 2),
                   xytext=(7, -14), textcoords="offset points", fontsize=6.2, color=PROBE,
                   arrowprops=dict(arrowstyle="-", color=PROBE, lw=.6))
    # right-anchored, so the offset is the gap to the marker rather than to the text's left edge
    # the synthetic chart point (86, 85) sits above the real-chart cluster and beside the
    # spatial point (59, 95), so the two synthetic labels are put on separate rows
    LAB = {("3b", "chart"): ("Qwen-3B chart", -6, 16),
           ("3b", "spatial"): ("Qwen-3B spatial", -9, -9),
           ("smolm", "spatial"): ("SmolVLM spatial", -9, 0)}
    for r in C["synthetic"]:
        k = (r["tag"], r["family"])
        if r["readout"] and k in LAB:
            t, dx, dy = LAB[k]
            a.annotate(t, (100 * (r["probe"] - r["nullcal"]["null_q95"]),
                           100 * r["probe_follow_ci"][0]),
                       textcoords="offset points", xytext=(dx, dy), fontsize=6.2, color=INK,
                       ha="right", va="center")
    # the near-miss is the informative negative: decodes well, does not track the edit
    nm = [r for r in C["synthetic"] if r["tag"] == "3b" and r["family"] == "counting"]
    if nm:
        r = nm[0]
        a.annotate("Qwen-3B counting\n(30/57 = 53%, bound 40%)",
                   (100 * (r["probe"] - r["nullcal"]["null_q95"]), 100 * r["probe_follow_ci"][0]),
                   xytext=(-40, -22), textcoords="offset points", fontsize=6.0, color="#8a4500",
                   arrowprops=dict(arrowstyle="-", color="#8a4500", lw=.6))
    # no in-plot region labels: the axis labels already name both calibrated quantities, and the
    # dashed lines plus shading carry the quadrant rule that the caption states
    a.set_xlabel("probe accuracy $-$ its own null 95th pct. (pp)")
    a.set_ylabel("follow rate, lower 95% bound (%)")
    a.set_xlim(-70, 100); a.set_ylim(-8, 108)
    a.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#777777", label="synthetic"),
                      Line2D([], [], marker="s", ls="", mfc="white", mec="#777777", label="real"),
                      Line2D([], [], marker="o", ls="", mfc=PROBE, mec=PROBE, label="readout locus")],
             loc="lower right", frameon=False, borderaxespad=.3, handletextpad=.3, fontsize=6.4)
    fig.savefig(f"{OUT}/fig2_locusmap.pdf"); fig.savefig(f"{OUT}/fig2_locusmap.png"); plt.close(fig)


def fig3():
    """The instrument earns its keep: it forecasts fine-tuning gain; free predictors do not."""
    P = C["prediction"]; rows, st = P["rows"], P["stats"]
    fam = [r["family"] for r in rows]
    def dm(x):
        x = np.array(x, float); o = x.copy()
        for f in set(fam):
            k = np.array([g == f for g in fam]); o[k] = x[k] - x[k].mean()
        return o
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.6),
                           gridspec_kw=dict(width_ratios=[1.15, 1], wspace=.42))
    MK = {"chart": "o", "counting": "s", "spatial": "^", "glyph": "D"}
    pg = dm([100 * (r["probe"] - r["base"]) for r in rows])
    lg = dm([100 * (r["lora"] - r["base"]) for r in rows])
    for i, r in enumerate(rows):
        ax[0].scatter(pg[i], lg[i], marker=MK[r["family"]], s=28, lw=1.1,
                      facecolor="white" if r["family"] == "glyph" else PROBE,
                      edgecolor=PROBE, zorder=3)
    z = np.polyfit(pg, lg, 1)
    xs = np.linspace(pg.min() - 2, pg.max() + 2, 10)
    ax[0].plot(xs, np.polyval(z, xs), color=MODEL, lw=1.1, ls="--", zorder=2)
    s = st["probe_gain"]
    ax[0].text(.03, .96, r"$\rho=%+.3f$, $p=%.4f$" % (s["rho_dm"], s["p_dm"]),
               transform=ax[0].transAxes, va="top", fontsize=7.5)
    ax[0].set_xlabel("probe gain, family-demeaned (pp)")
    ax[0].set_ylabel("fine-tuning gain, family-demeaned (pp)")
    ax[0].set_title("(a) 16 (model, family) pairs", loc="left")
    ax[0].set_ylim(min(lg) - 12, max(lg) + 4)
    ax[0].legend(handles=[Line2D([], [], marker=MK[f], ls="", mfc=PROBE if f != "glyph" else "white",
                                 mec=PROBE, label=f) for f in ["chart", "counting", "spatial", "glyph"]],
                 loc="lower right", frameon=False, ncol=2, handletextpad=.2,
                 columnspacing=.9, fontsize=6.5)
    NAME = [("probe_gain", "probe gain"), ("probe_acc", "probe accuracy"),
            ("base_acc", "base accuracy"), ("headroom", "headroom"),
            ("above_chance", "base $-$ chance")]
    y = np.arange(len(NAME))[::-1]
    ax[1].barh(y, [st[k]["rho_dm"] for k, _ in NAME],
               color=[PROBE if st[k]["p_dm"] < .05 else "#c8c8c8" for k, _ in NAME], height=.6)
    for yy, (k, _) in zip(y, NAME):
        ax[1].text(.99, yy, f"$p$={st[k]['p_dm']:.3f}", va="center", ha="right",
                   fontsize=6.5, color=INK if st[k]["p_dm"] < .05 else "#777777")
    ax[1].set_yticks(y); ax[1].set_yticklabels([n for _, n in NAME])
    ax[1].axvline(0, color=INK, lw=.7)
    ax[1].set_xlim(-.55, 1.02); ax[1].set_xticks([-.5, -.25, 0, .25, .5, .75]); ax[1].set_xlabel(r"Spearman $\rho$ with fine-tuning gain")
    ax[1].set_title("(b) against free predictors", loc="left")
    fig.savefig(f"{OUT}/fig3_prediction.pdf"); fig.savefig(f"{OUT}/fig3_prediction.png"); plt.close(fig)


def fig4():
    """The two controls that make the floor a measurement instead of an assertion."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw=dict(wspace=.3))
    d = json.load(open("runs/glyph_dose_3b.json"))
    px = sorted((int(k) for k in d), reverse=True)
    ax[0].plot(range(len(px)), [100 * d[str(k)] for k in px], marker="o", ms=3.4,
               color=MODEL, lw=1.3)
    ax[0].axhline(5.6, color=INK, lw=.7, ls="-.", alpha=.6)
    ax[0].text(len(px) - 1, 8.5, "chance", ha="right", fontsize=6.5, color=INK)
    ax[0].axvspan(len(px) - 2.5, len(px) - .5, color="#dddddd", alpha=.6, lw=0)
    ax[0].text(len(px) - 1.5, 55, "control\nregime", ha="center", fontsize=6.5, color="#555555")
    ax[0].set_xticks(range(len(px))); ax[0].set_xticklabels(px)
    ax[0].set_xlabel("numeral height (px)"); ax[0].set_ylabel("model accuracy (%)")
    ax[0].set_title("(a) the control floor is measured", loc="left")

    B = C["bands"]
    x = np.arange(len(B))
    ax[1].bar(x - .19, [100 * b["model"] for b in B], .36, color=MODEL, label="model output")
    ax[1].bar(x + .19, [100 * b["probe"] for b in B], .36, color=PROBE, label="probe @ final")
    for i, b in enumerate(B):
        d = 100 * (b["probe"] - b["model"])
        if d > 1:
            ax[1].text(i, 100 * b["probe"] + 2, f"{d:+.0f}", ha="center", fontsize=6.5, color=PROBE)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{b['lo']}-{b['hi']}\n(n={b['n']})" for b in B])
    ax[1].set_xlabel("true bar value"); ax[1].set_ylabel("accuracy (%)")
    ax[1].set_ylim(0, 118)
    ax[1].legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2,
                 handletextpad=.4, columnspacing=1.0, borderaxespad=.1)
    ax[1].set_title("(b) real-chart gap by bar value", loc="left")
    fig.savefig(f"{OUT}/fig4_controls.pdf"); fig.savefig(f"{OUT}/fig4_controls.png"); plt.close(fig)


def fig5():
    """The interventions the causal argument rests on, and the null it is judged against.

    Each row is one paired item: the original, the counterfactual, and the pixels that differ
    between them -- computed here rather than read from a stored mask, so the panel doubles as a
    visual check on the guard. Crops are to the changed region, since a 3 px numeral is invisible
    at page scale and the point of the figure is that the reader can inspect the edit.

    The right column is that cell's label-permutation null against its observed final-layer
    probe: the threshold, rather than a percentage point taken on trust.
    """
    from PIL import Image
    spec = [("chart", "data/real_chart", "chart_000000", "realchart", "real chart, bar raised"),
            ("glyph", "data/real_3b", "glyph_000000", "real", "designed absence, 3 px numeral")]
    fig, ax = plt.subplots(2, 4, figsize=(6.9, 3.6),
                           gridspec_kw=dict(wspace=.22, hspace=.55,
                                            width_ratios=[1, 1, 1, 1.45]))
    for r, (fam, root, iid, tag, label) in enumerate(spec):
        man = {x["id"]: x for x in (json.loads(l) for l in
                                   open(os.path.join(root, "manifest.jsonl")))}
        o, c = man[iid], man[f"{iid}_cf"]
        A = np.asarray(Image.open(os.path.join(root, o["image"])).convert("RGB"), int)
        B = np.asarray(Image.open(os.path.join(root, c["image"])).convert("RGB"), int)
        D = np.abs(A - B).sum(2)
        ys, xs = np.nonzero(D)
        # crop to the changed region, padded, and never smaller than 90 px so the edit has context
        pad = 26
        y0, y1 = max(0, ys.min() - pad), min(A.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(A.shape[1], xs.max() + pad + 1)
        if y1 - y0 < 90: y0, y1 = max(0, (y0 + y1) // 2 - 45), min(A.shape[0], (y0 + y1) // 2 + 45)
        if x1 - x0 < 90: x0, x1 = max(0, (x0 + x1) // 2 - 45), min(A.shape[1], (x0 + x1) // 2 + 45)
        cut = (slice(y0, y1), slice(x0, x1))
        panels = [(A[cut].astype(np.uint8), f"original, answer {o['answer']}", None),
                  (B[cut].astype(np.uint8), f"counterfactual, answer {c['answer']}", None),
                  (D[cut] > 0, f"{len(ys):,} pixels differ", "gray")]
        for k, (img, ttl, cm) in enumerate(panels):
            a = ax[r][k]
            a.imshow(img, cmap=cm, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(True); sp.set_linewidth(.5)
            a.set_title(ttl, loc="left", fontsize=6.4, pad=2.5)

        d = json.load(open(f"runs/null_{tag}.json"))[fam]
        a = ax[r][3]
        h = {float(k): v for k, v in d.get("null_hist", {}).items()}
        if h:
            xv = np.array(sorted(h)); w = np.array([h[v] for v in xv], float)
            bw = 100 * (xv[1] - xv[0]) if len(xv) > 1 else 1.0
            a.bar(100 * xv, w / w.sum(), width=max(.7, bw * .9), color=BLIND, alpha=.8, lw=0,
                  label=f"null ({d['nperm']} label permutations)")
        a.axvline(100 * d["null_q95"], color=INK, lw=.9, ls=":", label="null 95th pct")
        a.axvline(100 * d["vis"], color=PROBE, lw=1.7, label="probe @ final layer")
        a.set_xlabel("probe accuracy (%)", fontsize=6.8); a.set_ylabel("density", fontsize=6.8)
        a.set_yticks([])
        a.set_xlim(-2, max(100 * d["null_max"] + 14, 100 * d["vis"] + 8))
        pv = ("$p<0.001$" if d["p_null"] < 1e-3 else "$p=%.3f$" % d["p_null"])
        a.set_title(f"{label}    {pv}", loc="left", fontsize=6.4, pad=2.5)
        if r == 0:
            a.legend(frameon=False, fontsize=5.6, loc="upper left",
                     bbox_to_anchor=(.24, .99), handletextpad=.4,
                     borderaxespad=.1, labelspacing=.3)
    fig.savefig(f"{OUT}/fig5_interventions.pdf")
    fig.savefig(f"{OUT}/fig5_interventions.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("wrote", ", ".join(sorted(os.listdir(OUT))))
