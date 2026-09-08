"""Checks for the self-review pass: the seven places the reasoning was thin.

Each check corresponds to a specific reviewer objection that the revision now answers, and each
fails loudly if the paper drifts back. Run from the repository root.

    .venv/bin/python run/verify_review.py
"""
from __future__ import annotations
import json, math, os, re, subprocess, sys
import numpy as np
from scipy.stats import binomtest, ttest_rel

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = ".venv/bin/python"
ck = []


def chk(name, ok, note=""):
    ck.append((name, ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {note}" if note else ""))


subprocess.run([PY, "run/canon.py"], capture_output=True,
               env=dict(os.environ, VLM_LOCUS_TEX="../paper/tables"))
C = json.load(open("out/canon.json"))
rows = C["synthetic"] + C["real"]
tex = open("../paper/main.tex").read()
facts = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open("../paper/tables/facts.tex").read()))

# 1 -- the follow gate is on the bound, and no verdict is decided by a couple of items
loci = [r for r in rows if r["readout"]]
chk("every locus clears the follow gate on its LOWER bound",
    all(100 * r["probe_follow_ci"][0] > 50 for r in loci), f"{len(loci)} loci")
margins = [100 * r["probe_follow_ci"][0] - 50 for r in loci]
chk("no locus clears the follow gate by less than 10 pp of bound",
    min(margins) > 10, f"min margin {min(margins):.1f} pp")
chk("the two verdict paths agree on the locus count",
    len(loci) == int(subprocess.run([PY, "run/p3.py"], capture_output=True, text=True)
                     .stdout.count("READOUT")), f"canon {len(loci)}")

# 2 -- the sensitivity range is computed, stated, and does not reach zero
fs = C["follow_sensitivity"]
chk("the stated stable threshold range matches the computed one",
    facts["FollowStableLo"] == str(fs["lo"]) and facts["FollowStableHi"] == str(fs["hi"]),
    f"[{fs['lo']},{fs['hi']}]")
chk("the stable range does NOT extend to zero (so 'any low threshold' is not a defence)",
    fs["lo"] > 0, f"lo={fs['lo']}, admitted below: {fs['admitted_below']}")

# 3 -- the FPR is reported with its arithmetic, not as a bare rate
gl = [r for r in C["real"] if r["family"].startswith("glyph")]
k = sum(1 for r in gl if r["nullcal"]["p_null"] < 0.05)
bt = binomtest(k, len(gl), 0.05)
chk("1-of-5 is consistent with a nominal 5% test", bt.pvalue > 0.05, f"exact p={bt.pvalue:.3f}")
chk("the paper states the binomial p and the interval, not just the rate",
    all(f"\\Fpr{x}" in tex for x in ("P", "CiLo", "CiHi", "Exp")))
chk("the composed verdict's FPR on the control is reported separately",
    facts["FprVerdict"] == f"{sum(1 for r in gl if r['readout'])}/{len(gl)}",
    f"presence {facts['FprCtrl']}, verdict {facts['FprVerdict']}")

# 4 -- the out-of-sample claim carries its interval and is not stated as harm
for key in ("model", "family"):
    o = C["prediction"]["stats"]["oos"][f"heldout_{key}"]
    chk(f"leave-one-{key}-out difference is reported as unresolved",
        not o["resolved"] and o["ci"][0] < 0 < o["ci"][1],
        f"{o['delta']:+.2f} pp, p={o['p_paired']:.2f}, CI [{o['ci'][0]:+.1f},{o['ci'][1]:+.1f}]")
chk("the paper no longer asserts the probe is 'worse' without an interval",
    "Both directions are worse with the probe" not in tex)

# 5 -- multiplicity is addressed
m = C["multiplicity"]
chk("every locus survives Benjamini-Hochberg across the whole grid",
    not m["loci_bh_fail"], f"{len(m['loci_bh'])}/{len(m['loci'])} at n={m['n']}")
chk("the Holm result is stated including its one exception",
    facts["LociHolm"] == f"{len(m['loci_holm'])}/{len(m['loci'])}"
    and "Holm" in tex, f"Holm {facts['LociHolm']}, fails: {m['loci_holm_fail']}")

# 6 -- counting is exceptionless
cn = [r for r in rows if r["family"] == "counting"]
chk("all counting cells clear their own null and none is a locus",
    all(r["nullcal"]["p_null"] < 0.05 for r in cn) and not any(r["readout"] for r in cn),
    f"{len(cn)} cells")
chk("no surviving prose claims counting follows on 'under 30%'",
    "under 30\\% of items" not in tex)

# 7 -- the figure matches the calibrated protocol
# only the drawing calls count -- the docstring deliberately explains what was removed
figs = open("run/figs.py").read()
fig2 = figs[figs.index("def fig2():"):figs.index("def fig3():")]
code = "\n".join(l for l in fig2.split("\n")
                 if l.strip().startswith(("a.", "fig", "for", "if", "rc", "xs", "ys", "LAB", "nm")))
chk("the locus map draws no constant gate line",
    not re.search(r"ax[vh]line\(\s*2\.5", code) and "artefact floor" not in code)
chk("its gate lines are the calibrated ones (null at 0, follow bound at 50)",
    "axvline(0" in code and "axhline(50" in code)
chk("the locus map plots the follow lower bound, not the point estimate",
    "probe_follow_ci" in code and '100 * r["follow"]' not in code)

# 8 -- the head assertion covers the quantised configurations
chk("the paper reports the head check on all five configurations, not three",
    "all three architectures" not in tex and "five\nconfigurations" in tex.replace("\r", ""))

# 9 -- the novelty claim is scoped
chk("the permutation null is not claimed as a technique",
    "Nor do we claim the permutation null as a technique" in tex)
chk("'absent by construction' is gone in favour of a measured floor",
    "absent by construction" not in tex)

print(f"\n{sum(1 for _, o in ck if o)}/{len(ck)} checks pass")
sys.exit(0 if all(o for _, o in ck) else 1)
