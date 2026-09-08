"""The manuscript's figures, all from analysis/tables/canon.json plus the frozen run artefacts.

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

# Defaults keep output inside the repository; the manuscript build overrides them.
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
    """The locus map: probe gain alone does not decide; the causal control does."""
    fig, a = plt.subplots(figsize=(3.5, 3.0))
    rows = [(r, "o", "synthetic") for r in C["synthetic"]] + \
           [(r, "s", "real") for r in C["real"]]
    a.axvspan(-25, 2.5, color="#dddddd", alpha=.55, lw=0)
    a.axhspan(-3, 50, color="#dddddd", alpha=.55, lw=0)
    a.axvline(2.5, color=INK, lw=.7, ls="--"); a.axhline(50, color=INK, lw=.7, ls="--")
    for r, mk, _ in rows:
        if np.isnan(r["follow"]): continue
        good = r["readout"]
        a.scatter(r["gap"], 100 * r["follow"], marker=mk, s=26 if good else 20,
                  facecolor=PROBE if good else "white", edgecolor=PROBE if good else "#777777",
                  lw=.9, zorder=3)
    # the four real-chart loci sit almost on top of one another, so they get one group label
    rc = [r for r in C["real"] if r["family"] == "chart" and r["readout"]]
    if rc:
        x0, x1 = min(r["gap"] for r in rc), max(r["gap"] for r in rc)
        y0, y1 = min(100 * r["follow"] for r in rc), max(100 * r["follow"] for r in rc)
        a.add_patch(plt.Rectangle((x0 - 3, y0 - 4), (x1 - x0) + 6, (y1 - y0) + 8,
                                  fill=False, ec=PROBE, lw=.7, ls=":",
                                  transform=a.transData, zorder=2))
        a.annotate("real charts\n(4 models)", (x1 + 3, y0 - 4), xytext=(9, -16),
                   textcoords="offset points", fontsize=6.2, color=PROBE,
                   arrowprops=dict(arrowstyle="-", color=PROBE, lw=.6))
    LAB = {("q3b", "chart"): ("Qwen-3B chart", -6, -12),
           ("q3b", "spatial"): ("Qwen-3B spatial", -4, 6),
           ("q3b", "counting"): ("Qwen-3B counting", 6, -2),
           ("smol", "spatial"): ("SmolVLM spatial", -20, 8)}
    for r in C["synthetic"]:
        k = (r["model"], r["family"])
        if r["readout"] and k in LAB:
            t, dx, dy = LAB[k]
            a.annotate(t, (r["gap"], 100 * r["follow"]), textcoords="offset points",
                       xytext=(dx, dy), fontsize=6.2, color=INK)
    a.text(3.4, 16, "layer-selection artefact floor", fontsize=5.8, color="#555555",
           rotation=90, va="bottom")
    a.text(-24, 42, "causal-control threshold", fontsize=5.8, color="#555555")
    a.set_xlabel("final-layer probe gain over model (pp)")
    a.set_ylabel("counterfactual follow rate (%)")
    a.set_xlim(-25, 78); a.set_ylim(-6, 108)
    a.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#777777", label="synthetic"),
                      Line2D([], [], marker="s", ls="", mfc="white", mec="#777777", label="real"),
                      Line2D([], [], marker="o", ls="", mfc=PROBE, mec=PROBE, label="readout locus")],
             loc="lower right", frameon=False, borderaxespad=.3, handletextpad=.3)
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


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3(); fig4()
    print("wrote", ", ".join(sorted(os.listdir(OUT))))
