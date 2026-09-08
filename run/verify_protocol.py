"""Protocol-integrity checks: the properties the locus verdict is supposed to have.

Each check corresponds to an objection the protocol has to answer, and fails loudly if a change
to the analysis code breaks it. These run against the repository's own artefacts and generated
outputs only -- no path outside the repository, so they work from a fresh clone.

    python3 run/verify_protocol.py
"""
from __future__ import annotations
import json, math, os, re, subprocess, sys
import numpy as np
from scipy.stats import binomtest, ttest_rel

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable          # not a hardcoded venv path: a fresh clone has no .venv,
                             # and the analysis path needs no GPU dependency anyway
ck = []


def chk(name, ok, note=""):
    ck.append((name, ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {note}" if note else ""))


subprocess.run([PY, "run/canon.py"], capture_output=True)
subprocess.run([PY, "run/results_md.py"], capture_output=True)
C = json.load(open(os.environ.get("VLM_LOCUS_JSON", "out/canon.json")))
rows = C["synthetic"] + C["real"]
TEX = os.environ.get("VLM_LOCUS_TEX", "out/tables")
facts = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                        open(f"{TEX}/facts.tex").read()))
doc = open(os.environ.get("VLM_LOCUS_DOCS", "docs/RESULTS.md")).read()

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
chk("the generated results state the arithmetic, not just the rate",
    "exact binomial against alpha" in doc and "cannot resolve the rate" in doc
    and f"p = {bt.pvalue:.3f}" in doc)
chk("the composed verdict's FPR on the control is reported separately",
    facts["FprVerdict"] == f"{sum(1 for r in gl if r['readout'])}/{len(gl)}",
    f"presence {facts['FprCtrl']}, verdict {facts['FprVerdict']}")

# 4 -- the out-of-sample claim carries its interval and is not stated as harm
for key in ("model", "family"):
    o = C["prediction"]["stats"]["oos"][f"heldout_{key}"]
    chk(f"leave-one-{key}-out difference is reported as unresolved",
        not o["resolved"] and o["ci"][0] < 0 < o["ci"][1],
        f"{o['delta']:+.2f} pp, p={o['p_paired']:.2f}, CI [{o['ci'][0]:+.1f},{o['ci'][1]:+.1f}]")
chk("the generated results report the interval and decline to claim harm",
    "no demonstrable benefit" in doc and "95% CI" in doc
    and all(f"{C['prediction']['stats']['oos'][f'heldout_{k}']['p_paired']:.2f}" in doc
            for k in ("model", "family")))

# 5 -- multiplicity is addressed
m = C["multiplicity"]
chk("every locus survives Benjamini-Hochberg across the whole grid",
    not m["loci_bh_fail"], f"{len(m['loci_bh'])}/{len(m['loci'])} at n={m['n']}")
chk("the Holm result is stated including its one exception",
    facts["LociHolm"] == f"{len(m['loci_holm'])}/{len(m['loci'])}"
    and "Holm" in doc and all(n in doc for n in m["loci_holm_fail"]),
    f"Holm {facts['LociHolm']}, fails: {m['loci_holm_fail']}")

# 6 -- counting is exceptionless
cn = [r for r in rows if r["family"] == "counting"]
chk("all counting cells clear their own null and none is a locus",
    all(r["nullcal"]["p_null"] < 0.05 for r in cn) and not any(r["readout"] for r in cn),
    f"{len(cn)} cells")
chk("the generated results carry the counting counts the artefacts support",
    f"**{sum(1 for r in cn if r['nullcal']['p_null'] < 0.05)} of {len(cn)}**" in doc
    and f"**{sum(1 for r in cn if r['readout'])}** survive" in doc)

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

# 8 -- the head assertion can be run on the quantised configurations too, since quantisation
#      changes the numerics the assertion measures and two configurations are quantised
ah = open("run/assert_head.py").read()
chk("the head assertion is runnable under 4-bit quantisation", "--load-4bit" in ah)
chk("...and its tolerance is discriminative rather than an equality test",
    "100 * r_post" in ah)

# 9 -- the repository is self-contained: nothing reads a path outside it
outside = []
for d, _, fs in os.walk("."):
    if any(x in d for x in (".git", ".venv", "out/", "runs/superseded")): continue
    for f in fs:
        if not f.endswith((".py", ".sh", ".md")): continue
        fp = os.path.join(d, f)
        for i, ln in enumerate(open(fp, errors="ignore"), 1):
            # any "../" is suspect: a self-contained repository never needs to leave its own
            # root, and the four stage scripts that used ../vlm-locus/.venv broke for anyone who
            # cloned to a differently named directory
            if "../" in ln and not ln.lstrip().startswith("#") and "os.path" not in ln:
                outside.append(f"{fp}:{i}")
chk("no tracked source or doc reads a path outside the repository", not outside,
    f"offenders={outside}")

# 10 -- the README is the repository's front page and carries hardcoded numbers, which is the
#       drift the generated results document exists to avoid. It cannot be generated (it is
#       prose), so it is checked instead.
R = open("README.md").read()
q95 = [100 * r["nullcal"]["null_q95"] for r in rows]
readme = {
    "the null range it quotes": f"{min(q95):.1f}% to {max(q95):.1f}%" in R,
    "the artefact count it promises": f"All {len(json.loads(subprocess.run([PY, 'run/manifest.py', '--json'], capture_output=True, text=True).stdout)['reads'])} artefacts" in R,
    "the follow gate it states": f"{C['protocol']['follow_min_pct']:.0f}%" in R and "lower 95% bound" in R,
    "its claim that the composed verdict never fires on the control":
        (sum(r["readout"] for r in gl) == 0) == ("never fires" in R),
    "only scripts that exist": all(os.path.exists(f) for f in
                                   re.findall(r"run/[a-z_0-9]+\.(?:py|sh)", R)),
    # the third-party-data statement names exact counts; a regenerated index would silently
    # falsify them, and a licence statement is the wrong thing to let drift
    "the ChartQA-derived entry counts it discloses": all(
        f"({len(json.load(open(f)))}" in R.replace(" charts)", ")")
        for f in ("runs/chart_index.json", "runs/chart_index_nolabel.json")),
    "that no COCO annotation content is committed": not any(
        re.search(r'"(bbox|segmentation)"', open(f, errors="ignore").read())
        for f in subprocess.run(["git", "ls-files", "runs/"], capture_output=True,
                                text=True).stdout.split()),
}
for what, good in readme.items():
    chk(f"README: {what} matches the artefacts", good)

# 11 -- the generated results document is byte-stable, so diffing it verifies it
d1 = open(os.environ.get("VLM_LOCUS_DOCS", "docs/RESULTS.md")).read()
subprocess.run([PY, "run/results_md.py"], capture_output=True)
chk("the generated results document is byte-stable across regeneration",
    d1 == open(os.environ.get("VLM_LOCUS_DOCS", "docs/RESULTS.md")).read())

print(f"\n{sum(1 for _, o in ck if o)}/{len(ck)} checks pass")
sys.exit(0 if all(o for _, o in ck) else 1)
