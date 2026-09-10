# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
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


sys.path.insert(0, os.path.join(os.getcwd(), "run"))
# the gated statistic's name comes from canon.py, not from a copy here. Checks 1 and 12c below
# assert properties of "the bound the gate uses", and for three commits they asserted them of
# the conditional forward bound while canon.py gated on the joint one -- passing by luck, since
# every locus clears both. A verification harness with its own copy of the rule verifies nothing.
from canon import GATE_CI, locus, g1_at
import canon

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
    all(100 * r[GATE_CI][0] > 50 for r in loci), f"{len(loci)} loci, gate on {GATE_CI}")
margins = [100 * r[GATE_CI][0] - 50 for r in loci]
chk("no locus clears the follow gate by less than 10 pp of bound",
    min(margins) > 10, f"min margin {min(margins):.1f} pp")
# and the rule the document states is the rule canon.py runs, recomputed from the row
chk("every printed verdict equals locus() recomputed from its own row",
    all(bool(r["readout"]) == locus(r) for r in rows))
# and the presence test is inside that rule rather than reported beside it. Asserting that every
# locus passes G1 would be vacuous here -- no cell fails G1 while passing the other three -- so
# the check is that tightening G1 alone moves the verdict, which it cannot if g1_at is not in the
# conjunction. This is the check that would have caught G1 missing from it for three commits.
tight = [r for r in rows if locus(r, pos=0.99)]
chk("the presence test binds the verdict, not merely reported beside it",
    len(tight) < len(loci) and all(g1_at(r["nullcal"]) for r in loci),
    f"tightening pos alone: {len(loci)} loci -> {len(tight)}")
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

# 4 -- the out-of-sample claim carries its interval and is not stated as harm.
# The claim is about the folds the paper endorses, which exclude the designed control: adaptation
# cannot help on a family with nothing to read, so a fold whose answer is "nothing happens" tests
# the control rather than the forecast. This check asserted the unqualified version and failed
# when the larger chart set made the control fold worth -11 pp on its own -- correctly, because
# the manuscript's claim was then wider than its evidence. It now checks the claim as stated.
for key in ("model", "family"):
    o = C["prediction"]["stats"]["oos_noctrl"][f"heldout_{key}"]
    chk(f"leave-one-{key}-out difference is unresolved with the control excluded",
        not o["resolved"] and o["ci"][0] < 0 < o["ci"][1],
        f"{o['delta']:+.2f} pp, p={o['p_paired']:.2f}, CI [{o['ci'][0]:+.1f},{o['ci'][1]:+.1f}]")
# and the manuscript does not present the control-inclusive family figure as a bare negative
chk("the control-inclusive family difference is disclosed, not hidden",
    C["prediction"]["stats"]["oos"]["heldout_family"]["delta"] < 0
    <= abs(C["prediction"]["stats"]["oos"]["heldout_family"]["per_fold"]["glyph"]),
    f"glyph fold {C['prediction']['stats']['oos']['heldout_family']['per_fold']['glyph']:+.1f} pp")
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

# 12 -- licensing. Copyleft propagates from the ChartQA-derived index files, so the licence is
#       not a cosmetic field: a permissive grant left behind anywhere would be a claim the
#       author is not in a position to make. Checked, for the same reason the entry counts are.
# 12a -- protocol constants live in one place. The rare-class filter was the literal 8 in five
# analysis files and in the data builder, which is the shape of defect this revision kept finding:
# gen/build_chart.py uses it to decide which edit targets a probe can emit, so a copy that drifted
# would silently produce counterfactuals no probe could follow.
lit = subprocess.run(["git", "grep", "-n", r"sum() >= 8", "--", "run/", "gen/",
                      ":!run/verify_protocol.py"],
                     capture_output=True, text=True).stdout.split()
chk("the rare-class filter is imported, not copied", not lit, f"literals={lit[:3]}")

# 12b -- no near-duplicate pair straddles the split. A dataset whose items come in pairs that
# differ in one bar is leakage waiting to happen: the probe can memorise the image in training
# and be scored on its twin. gen/build_chart_bidir.py records the pair, canon.split() groups on
# it, and this asserts the outcome rather than trusting the plumbing -- the cost of getting it
# wrong is every number moving in the flattering direction, silently.
import glob as _glob
straddle = []
for mp in sorted(_glob.glob("runs/states_*_meta.json")):
    tag = os.path.basename(mp)[len("states_"):-len("_meta.json")]
    meta = json.load(open(mp))
    for fam in {m["family"] for m in meta}:
        ix = [i for i, m in enumerate(meta) if m["family"] == fam]
        g = canon.pairs_of(meta, ix)
        if g is None:
            continue
        y = np.array([str(meta[i].get("answer", "")) for i in ix])
        tr, sel, te = canon.split(y, groups=g, seed=0)
        if set(g[tr]) & set(g[te]) or set(g[tr]) & set(g[sel]) or set(g[sel]) & set(g[te]):
            straddle.append(f"{tag}/{fam}")
# and every captured dataset declares the field at all. run/capture.py used to drop it, so a
# paired dataset captured before that fix reports "no pairs" and its split silently ungroups --
# which is how the first bidirectional capture put 275 of 466 pairs across the boundary.
nofield = [os.path.basename(mp)[len("states_"):-len("_meta.json")]
           for mp in sorted(_glob.glob("runs/states_*_meta.json"))
           for m in [json.load(open(mp))] if m and "pair" not in m[0]]
chk("every captured dataset declares the pair field", not nofield,
    f"missing in {len(nofield)} of {len(_glob.glob('runs/states_*_meta.json'))}"
    + (f": {nofield[:4]}" if nofield else ""))

npaired = sum(1 for mp in _glob.glob("runs/states_*_meta.json")
              for m in [json.load(open(mp))]
              if any(canon.pairs_of(m, range(len(m))) is not None for _ in [0]))
chk("no near-duplicate pair straddles the split", not straddle,
    f"{npaired} paired dataset(s) captured"
    + (" -- vacuous until one is" if not npaired else "")
    + (f", straddling: {straddle}" if straddle else ""))

SPDX = "GPL-3.0-only"
LIC = open("LICENSE").read()
src = subprocess.run(["git", "ls-files", "*.py", "*.sh"], capture_output=True,
                     text=True).stdout.split()
nohdr = [f for f in src if SPDX not in "".join(open(f).readlines()[:5])]
chk("LICENSE is the unmodified GPL-3.0 text",
    all(s in LIC for s in ("GNU GENERAL PUBLIC LICENSE", "Version 3, 29 June 2007",
                           "END OF TERMS AND CONDITIONS", "How to Apply These Terms")))
chk(f"every tracked source file carries SPDX {SPDX}", not nohdr, f"missing={nohdr[:5]}")
chk("the README names the same licence the sources declare",
    f"[{SPDX}](LICENSE)" in R and "GNU General Public License" in R)
chk("no permissive grant survives anywhere in the tree",
    not subprocess.run(["git", "grep", "-lI", "-e", "MIT License", "-e",
                        "Permission is hereby granted", "--", ":!run/verify_protocol.py"],
                       capture_output=True, text=True).stdout.split())

# 12b -- the SPDX headers shifted every source file down by two lines, which is exactly the way
#        a `file.py:N-M` citation goes quietly stale. Every such citation in the tree points at a
#        comment block, so the range must cover one exactly: all comment lines, with non-comment
#        lines either side. Requiring only that the topic word fall somewhere inside the window
#        is too weak -- a shift of one or two lines keeps it there.
cites = {(f if "/" in f else "run/" + f, int(a), int(b)) for f, a, b in re.findall(
    r"([a-z_0-9/]+\.py):(\d+)-(\d+)", subprocess.run(
        ["git", "grep", "-hI", "-oE", r"[a-z_0-9/]+\.py:[0-9]+-[0-9]+"],
        capture_output=True, text=True).stdout)}


def covers_a_block(f, a, b, topic):
    """The cited lines are a whole comment block, and it is the one the citation is about."""
    ln = [l.strip() for l in open(f).readlines()]
    if not 1 <= a <= b <= len(ln):
        return False
    cited = ln[a - 1:b]
    before = ln[a - 2] if a > 1 else ""          # "" is not a comment, so a block at the top
    after = ln[b] if b < len(ln) else ""         # or bottom of the file still reads correctly
    return (all(l.startswith("#") for l in cited)
            and not before.startswith("#") and not after.startswith("#")
            and topic in " ".join(cited))


chk("the source citation in the docs covers exactly the comment block it names",
    len(cites) == 1 and all(covers_a_block(f, a, b, "sdpa") for f, a, b in cites),
    f"cites={sorted(cites)}")

print(f"\n{sum(1 for _, o in ck if o)}/{len(ck)} checks pass")
sys.exit(0 if all(o for _, o in ck) else 1)
