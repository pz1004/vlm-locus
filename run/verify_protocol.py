# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Protocol-integrity checks: the properties the locus verdict is supposed to have.

Each check corresponds to an objection the protocol has to answer, and fails loudly if a change
to the analysis code breaks it. These run against the repository's own artefacts and generated
outputs only -- no path outside the repository, so they work from a fresh clone.

    python3 run/verify_protocol.py
"""
from __future__ import annotations
import ast, json, math, os, re, subprocess, sys
from importlib.metadata import PackageNotFoundError, packages_distributions, version
import numpy as np
from PIL import Image
from scipy.stats import binomtest, ttest_rel

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable          # not a hardcoded venv path: a fresh clone has no .venv,
                             # and the analysis path needs no GPU dependency anyway
ck = []


def chk(name, ok, note=""):
    ck.append((name, ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {note}" if note else ""))


# The analysis environment, read once. requirements-analysis.txt pins the interpreter every
# committed number was emitted under; check 27 asserts the pin covers what the analysis path
# imports, and check 18 uses this to say why a reproduction failure is probably not the artefacts.
PIN = "requirements-analysis.txt"
PINS = {}
for _ln in open(PIN):
    _ln = _ln.split("#")[0].strip()
    if "==" in _ln:
        _d, _v = _ln.split("==")
        PINS[_d.strip().lower()] = _v.strip()
ENVDRIFT = []
for _d, _v in sorted(PINS.items()):
    try:
        _have = version(_d)
    except PackageNotFoundError:
        _have = "absent"
    if _have != _v:
        ENVDRIFT.append(f"{_d} {_have} != {_v}")


sys.path.insert(0, os.path.join(os.getcwd(), "run"))
# the gated statistic's name comes from canon.py, not from a copy here. Check 1 below asserts
# properties of "the bound the gate uses", and for three commits it asserted them of
# the conditional forward bound while canon.py gated on the joint one -- passing by luck, since
# every locus clears both. A verification harness with its own copy of the rule verifies nothing.
from canon import GATE_CI, locus, g1_at
import canon

# The generators run first and everything below reads what they wrote, so a generator that exits
# must be reported as itself. canon.py can exit deliberately -- prediction() refuses to pair a
# probe and an adaptation scored on different item sets -- and discarding its output turned that
# into a JSONDecodeError on a file it never wrote, which names neither the cause nor the script.
for _s in ("run/canon.py", "run/results_md.py"):
    _r = subprocess.run([PY, _s], capture_output=True, text=True)
    if _r.returncode:
        msg = (_r.stderr or _r.stdout or "").strip().split("\n")
        sys.exit(f"{_s} exited {_r.returncode}, so there is nothing to verify:\n  "
                 + "\n  ".join(msg[-3:]))
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
# Only the body counts: the docstring deliberately explains what was removed, and says "artefact
# floor" while doing so. It used to be excluded by an allowlist of line prefixes, which silently
# dropped every line the list did not anticipate -- `lo = 100 * r[GATE_CI][0]`, the one that sets
# the y value, began with neither. Cutting the docstring instead keeps every code line.
figs = open("run/figs.py").read()
fig2 = figs[figs.index("def fig2():"):figs.index("def fig3():")]
code = fig2[fig2.index('"""', fig2.index('"""') + 3) + 3:]
chk("the locus map draws no constant gate line",
    not re.search(r"ax[vh]line\(\s*2\.5", code) and "artefact floor" not in code)
chk("its gate lines are the calibrated ones (null at 0, follow bound at 50)",
    "axvline(0" in code and "axhline(50" in code)
# and it plots the bound the verdict is gated on, named from canon rather than copied. It plotted
# probe_follow_ci against a line at 50 for several commits while the gate was probe_joint_ci; no
# cell straddled 50 on one and not the other, so it passed on luck rather than on agreement.
chk("the locus map plots the gated bound, not a second copy of the rule",
    "GATE_CI" in code and not re.search(r'"probe_\w+_ci"', code)
    and '100 * r["follow"]' not in code)

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

# 12 -- protocol constants live in one place. The rare-class filter was the literal 8 in five
# analysis files and in the data builder, which is the shape of defect this revision kept finding:
# gen/build_chart.py uses it to decide which edit targets a probe can emit, so a copy that drifted
# would silently produce counterfactuals no probe could follow.
lit = subprocess.run(["git", "grep", "-n", r"sum() >= 8", "--", "run/", "gen/",
                      ":!run/verify_protocol.py"],
                     capture_output=True, text=True).stdout.split()
chk("the rare-class filter is imported, not copied", not lit, f"literals={lit[:3]}")

# 13 -- no near-duplicate pair straddles the split. A dataset whose items come in pairs that
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

# 14 -- the reverse follow rate is only worth quoting where it can differ from the forward one.
# Both statistics count the same numerator, items whose two endpoints the probe both reads
# correctly, and differ only in denominator: forward divides by the items read correctly before
# the edit, reverse by those read correctly after it. On a set closed under pair reversal every
# image appears once as a base and once as a counterfactual, so the two denominators range over
# one multiset of states and are equal by construction -- the reverse rate is then identically
# the forward rate and carries no information at all. The statistic exists to answer "does the
# probe only track increases?", and it is exactly the bidirectional dataset built to ask that
# question on which it cannot answer. So two things are asserted: that the identity does hold
# where the geometry says it must (it fails if base and counterfactual captures ever diverge on
# the same image, which is the skew run/capture.py's header documents), and that no reverse-rate
# macro is sourced from a cell where it is degenerate.
def reversal_closed(recs, pair_of):
    """Every pair contributes both of its directions, with the endpoints swapped."""
    grp = {}
    for r in recs:
        p = pair_of.get(r["id"])
        if p is None:
            return False
        grp.setdefault(p, []).append(r)
    return bool(grp) and all(len(v) == 2 and (v[0]["a0"], v[0]["a1"]) == (v[1]["a1"], v[1]["a0"])
                             for v in grp.values())


symmetric, broken = [], []
for fp in sorted(_glob.glob("runs/cffollow_*.json")):
    tag = os.path.basename(fp)[len("cffollow_"):-len(".json")]
    mp = f"runs/states_{tag}_meta.json"
    if not os.path.exists(mp):
        continue
    pair_of = {m["id"]: m.get("pair") for m in json.load(open(mp))}
    recs = json.load(open(fp))
    for fam in {r["family"] for r in recs}:
        g = [r for r in recs if r["family"] == fam]
        if not reversal_closed(g, pair_of):
            continue
        symmetric.append(f"{tag}/{fam}")
        d0 = sum(r["probe0"] == r["a0"] for r in g)
        d1 = sum(r["probe1"] == r["a1"] for r in g)
        if d0 != d1:
            broken.append(f"{tag}/{fam} {d0}!={d1}")
chk("forward and reverse coincide on every reversal-closed set, as the geometry requires",
    not broken, f"{len(symmetric)} symmetric cell(s)"
    + (" -- vacuous until one is" if not symmetric else "")
    + (f", diverging: {broken}" if broken else ""))

# and nothing quotes the degenerate value. A "Rev" macro carrying a reversal-closed cell's
# reverse rate would read as evidence of direction-insensitivity while being arithmetic. To
# negative-test this one, inject the macro in run/canon.py where it is generated: editing
# out/tables/facts.tex does nothing, because canon.py takes VLM_LOCUS_TEX as its *output*
# directory and this harness reruns it before reading the file back.
degenerate = set()
for cell in symmetric:
    tag, fam = cell.split("/", 1)
    fl = canon.follow(tag).get(fam, {})
    if "probe_reverse_ci" in fl:
        degenerate |= {f"{100 * fl['probe_reverse_ci'][0]:.0f}", f"{100 * fl['probe_reverse']:.0f}"}
quoted = sorted(k for k, v in facts.items() if "Rev" in k and v in degenerate)
chk("no reverse-rate macro is sourced from a cell where it cannot differ",
    not quoted, f"{len(degenerate)} degenerate value(s) guarded"
    + (" -- vacuous until one is" if not degenerate else "")
    + (f", quoted by: {quoted}" if quoted else ""))

# 15 -- the split table and its macros are populated from the canonical rows, not from a string
# test on the key. run/splits.py merges rather than clobbers, so runs/splits.json accumulates
# every cell ever resampled; `"glyph" not in k` admitted all of them and additionally conflated
# "decision cell" with "locus". Five bidirectional cells merged in took SplitProbeMin from 76
# down to 31 and SplitLoci from 7 up to 12, and rendered their raw tags into a LaTeX table,
# where the underscore is a hard error. This check is live rather than hypothetical: those five cells are
# in runs/splits.json now and the note below says how many are being excluded.
sdj = json.load(open(canon.SPLITS)) if os.path.exists(canon.SPLITS) else {}
cells = {f"{r['tag']}/{r['family']}" for r in rows}
loci_k = {f"{r['tag']}/{r['family']}" for r in rows if r["readout"]}
extra = sorted(k for k in sdj if k not in cells)
stex = open(f"{TEX}/splits.tex").read()
drawn = [l for l in stex.split("\n") if l.rstrip().endswith("\\\\") and "&" in l
         and "Model &" not in l]
want = [k for k in sdj if k in cells]
chk("the split table renders every canonical cell and nothing else",
    len(drawn) == len(want) and not any("_" in l.split("&")[0] for l in drawn),
    f"{len(drawn)} rows, {len(want)} canonical, {len(extra)} non-canonical excluded")
# and the two populations the macros name are the sets they claim, recomputed from the verdicts
chk("the split macros count loci and decision cells separately, and correctly",
    int(facts["SplitLoci"]) == len([k for k in sdj if k in loci_k])
    and int(facts["SplitCells"]) == len([k for k in sdj if k in cells
                                         and not k.split("/")[1].startswith("glyph")]),
    f"loci {facts['SplitLoci']}, decision cells {facts['SplitCells']}")

# 16 -- the band paragraph's two structural claims, the ones no macro carries. Its numbers were
# literals until the larger chart set made every one of them wrong -- +27.8 pp against an actual
# +9.7, "18--20 items per band" against 22--33, and "the 25--49 band is 100% for both" naming a
# band that is 93.9/97.0 -- while tab:bands beside them regenerated and disagreed. The numbers
# are macros now; these two are the sentence's remaining prose assertions.
bands = C.get("bands") or []
if bands:
    peak = max(range(len(bands)), key=lambda i: bands[i]["model"])
    chk("the band paragraph's 'peaks in the middle and falls again' still holds",
        0 < peak < len(bands) - 1 and bands[-1]["model"] < bands[peak]["model"],
        f"peak at band {peak + 1} of {len(bands)}, "
        f"{100 * bands[peak]['model']:.0f}% then {100 * bands[-1]['model']:.0f}%")
    chk("the bands with no gap are the high ones, as the paragraph says",
        all(b["probe"] == b["model"] for b in bands[-int(facts["BandTiedN"]):]),
        f"{facts['BandTiedN']} of {facts['BandN']} tied, all at the top")

# 17 -- the bidirectional set is a control and must stay one. Its base items are half
# generator-painted, because the lowering direction has to start from an edited image, so
# promoting it into REAL would buy a cleaner counterfactual by making the real-image replication
# half synthetic. The claims the manuscript makes about it are that the class-support truncation
# is gone and that the four cells carrying a gap bound well above the one that does not.
bd = C.get("bidir") or []
if bd:
    chk("the bidirectional control is not in the canonical grid",
        not ({b["tag"] for b in bd} & {r["tag"] for r in rows}),
        f"{len(bd)} control cells, {len({r['tag'] for r in rows})} canonical tags")
    chk("its class-support truncation is gone, and beats the one-directional set",
        max(b["unsupported"] for b in bd) == 0 < int(facts["UnsupChartMax"]),
        f"{facts['BidirUnsupMax']} unfollowable, against {facts['UnsupChartMax']}")
    # the manuscript's strongest claim about the control is that judging it by the same four
    # conditions returns the same answer as the replication set, model for model. That is what
    # makes it a control rather than a second experiment, so it is asserted rather than eyeballed.
    cm = {r["model"] for r in C["real"] if r["family"] == "chart" and r["readout"]}
    bm = {b["model"] for b in bd if b["readout"]}
    chk("the control reproduces the replication set's verdict, model for model", cm == bm,
        f"canonical {sorted(cm)}, control {sorted(bm)}")
    # the split the macros make is on the gap, so it has to actually separate the cells
    pos = [b for b in bd if b["gap"] > 0]
    chk("the control's gap split separates the cells it claims to",
        len(pos) == int(facts["BidirCellsPos"]) < len(bd)
        and min(b["joint_ci"][0] for b in pos) > max(b["joint_ci"][0] for b in bd if b["gap"] <= 0),
        f"{len(pos)} with a gap bound >= {100 * min(b['joint_ci'][0] for b in pos):.0f}%, "
        f"{len(bd) - len(pos)} without <= "
        f"{100 * max(b['joint_ci'][0] for b in bd if b['gap'] <= 0):.0f}%")

# 18 -- run/canonpred.py and run/layers.py fit the same probe and must agree. They are separate
# fits of one definition, which is the arrangement this protocol keeps finding bugs in, so the
# agreement is asserted rather than assumed. It is not exact everywhere: the fit is thread-count
# dependent on exactly one cell of the grid, the InternVL degraded-glyph control, where the probe
# scores below chance and its predictions are arbitrary by construction. canonpred pins BLAS
# threads so a reader on another machine gets the same file; layers.py does not. The boundary is
# the point of the check -- any cell that clears its own null must agree exactly, because a cell
# with real signal has no near-ties to flip.
dis = []
for r in rows:
    fp = f"runs/canonpred_{r['tag']}.json"
    if not os.path.exists(fp):
        continue
    d = json.load(open(fp)).get(r["family"])
    if d is None or abs(d["acc"] - r["probe"]) < 1e-12:
        continue
    dis.append((f"{r['tag']}/{r['family']}", round(abs(d['acc'] - r['probe']) * r['n_test']),
                r["nullcal"]["p_null"] < canon.NULL_ALPHA))
chk("canonpred and layers.py agree on every cell that clears its own null",
    not any(clears for _, _, clears in dis),
    f"{len(dis)} cell(s) differ, all below their null: "
    + (", ".join(f"{n} by {k} item(s)" for n, k, _ in dis) if dis else "none"))
# and the producer reproduces what is committed, which is what makes the verdict rederivable
rc = subprocess.run([PY, "run/canonpred.py", "--check"], capture_output=True, text=True)
# A failure here reads as "the artefacts are wrong" and is far more often "this is the other
# interpreter": the committed files were emitted under requirements-analysis.txt, and numpy 2.5.2
# against 2.4.6 flips three tied predictions of 1125 across this grid. Say so in the note rather
# than leaving a reader to find it in the data.
chk("the committed canonpred files are reproduced by their producer",
    rc.returncode == 0 and "all reproduce" in rc.stdout,
    (rc.stdout.strip().split("\n")[-1] if rc.stdout else "no output")
    + (f"; note this is not the pinned analysis environment ({', '.join(ENVDRIFT)})"
       if ENVDRIFT else ""))

# 19 -- every artefact family the analysis reads is named, literally, in some tracked producer.
# runs/canonpred_*.json had a producer, run/canon_pred.py, that built its output path from argv
# and was called by no driver. The string "canonpred" therefore appeared nowhere in it, so a
# search for the artefact's name found only the consumer and the files looked producerless. The
# artefact was reproducible all along and nothing could show that it was. A producer that does
# not name what it writes is invisible to exactly the search anyone would run.
# the consumers are excluded: run/canon.py names every family it reads, so searching them too
# would let a consumer stand in for a producer -- which is the exact confusion that hid this.
READERS = {"run/canon.py", "run/verify_protocol.py", "run/manifest.py",
           "run/verify_provenance.py", "run/results_md.py"}
# shell drivers count: one that invokes a producer with an explicit --out names the artefact
# just as well as a default path does. runs/lora20_3b.json is findable that way. canonpred was
# findable neither way, which is what separated it.
prod = [f for f in subprocess.run(["git", "ls-files", "run/*.py", "gen/*.py", "*.sh"],
                                  capture_output=True, text=True).stdout.split()
        if f not in READERS]
src = {f: open(f).read() for f in prod}
fams, unnamed = set(), []
for pth in json.loads(subprocess.run([PY, "run/manifest.py", "--json"],
                                     capture_output=True, text=True).stdout)["reads"]:
    b = os.path.basename(pth)
    if not pth.startswith("runs/") or "_" not in b:
        continue
    # the family is the stem up to the tag, e.g. runs/canonpred_q7b...json -> "canonpred_"
    fams.add(b.split("_")[0] + "_")
for f in sorted(fams):
    if not any(f in t for t in src.values()):
        unnamed.append(f)
chk("every artefact family the analysis reads is named in a tracked producer",
    not unnamed, f"{len(fams)} families checked"
    + (f", unnamed: {unnamed}" if unnamed else ""))

# 20 -- licensing. Copyleft propagates from the ChartQA-derived index files, so the licence is
#       not a cosmetic field: a permissive grant left behind anywhere would be a claim the
#       author is not in a position to make. Checked, for the same reason the entry counts are.
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

# 21 -- the SPDX headers shifted every source file down by two lines, which is exactly the way
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

# 22 -- the label-preserving control, and the dissociation that is the whole point of it. Neither
# half means anything alone: a probe that never moves on a sham has shown nothing if it also never
# follows a real edit, and that is not hypothetical -- it is the cell without a gap, which moves on
# shams more than any other and tracks real edits least. So the two are asserted together, on the
# same items, with the sham edit the same size as the real one it is paired with.
sh = C.get("sham") or []
if sh:
    ch = [b for b in sh if b["family"] == "chart"]
    pos = [b for b in ch if b["gap"] > 0]
    neg = [b for b in ch if b["gap"] <= 0]
    # one held-out set PER FAMILY, not one across all of them. This assertion read
    # `len({b["n"] for b in sh}) == 1` while chart was the only family with a sham, so it would
    # have failed the moment a second family arrived -- correctly, but for the wrong reason: the
    # families have different answer spaces and different drop rates, so a single n was never
    # the property. What has to hold is that within a family every model scored the same items.
    byfam = {}
    for b in sh:
        byfam.setdefault(b["family"], set()).add(b["n"])
    chk("within each family the sham control is scored on one held-out set",
        all(len(v) == 1 for v in byfam.values()),
        ", ".join(f"{f} n={sorted(v)[0]}" for f, v in sorted(byfam.items())))
    # ...and those items are the cell's own held-out set, not some other split's
    bad = []
    for b in sh:
        cp = f"runs/canonpred_{b['tag']}.json"
        if os.path.exists(cp):
            held = json.load(open(cp)).get(b["family"], {}).get("ids") or []
            if held and b["n"] > len(held):
                bad.append(f"{b['tag']}/{b['family']} {b['n']}>{len(held)}")
    chk("every sham cell is a subset of that cell's canonical held-out items", not bad,
        f"{len(sh)} cells" + (f", over-sized: {bad[:3]}" if bad else ""))
    chk("cells with a gap follow the real edit and do not move on the sham",
        all(b["follow_real"] > 0.5 and b["probe_moved"] / b["n"] < 0.05 for b in pos),
        f"{len(pos)} chart cells: follow "
        f"{100*min(b['follow_real'] for b in pos):.0f}-{100*max(b['follow_real'] for b in pos):.0f}%, "
        f"moved at most {max(b['probe_moved'] for b in pos)}/{pos[0]['n']}")
    chk("the cell that moves on the sham is the cell that does not follow the real edit",
        bool(neg) and max(ch, key=lambda b: b["probe_moved"]) is
        max(neg, key=lambda b: b["probe_moved"])
        and min(ch, key=lambda b: b["follow_real"]) in neg,
        f"{neg[0]['label'] if neg else '-'}: moved {neg[0]['probe_moved'] if neg else 0}, "
        f"follows {100*neg[0]['follow_real'] if neg else 0:.0f}%")
    # and the images themselves hold: same answer, exact pixel guard, queried attribute untouched
    for g, root in (("gen/verify_sham.py", "data/real_chart_sham"),
                    ("gen/verify_real_sham.py", "data/real_3b_sham")):
        if not os.path.isdir(root):
            continue
        rs = subprocess.run([PY, g], capture_output=True, text=True)
        chk(f"the sham images pass their own guards ({os.path.basename(root)})",
            rs.returncode == 0 and "PASS" in rs.stdout,
            rs.stdout.strip().split("\n")[-1] if rs.stdout else "did not run")
    # The control's own result, and the reason it is worth running on families that are not loci:
    # sham stability separates the verdict set on its own, with no threshold chosen to make it do
    # so. This is a max against a min -- if the two groups ever overlap, the claim in the
    # manuscript is false and this fails rather than narrowing.
    # grouped on the COUNTERFACTUAL condition, not on the full verdict. Both statistics here are
    # counterfactual; the verdict also prices the model, and one chart cell reads its attribute
    # perfectly while failing the verdict only because the model reads it too. Grouping on the
    # verdict puts that cell among the failures at 0 of 114 moved and destroys a real separation.
    L = [b for b in sh if b["cf_pass"]]
    O = [b for b in sh if not b["cf_pass"]]
    if L and O:
        lo, hi = (max(b["probe_moved"] / b["n"] for b in L),
                  min(b["probe_moved"] / b["n"] for b in O))
        chk("sham stability separates the cells that pass the counterfactual gate from those that do not",
            lo < hi, f"{len(L)} passing move at most {100*lo:.1f}%, "
            f"{len(O)} failing at least {100*hi:.1f}%")
    # and on the designed absence the two readers' stabilities invert: the probe, which has
    # nothing to read, moves more than the model whose state it is reading
    gy = [b for b in sh if b["family"].startswith("glyph")]
    if gy:
        inv = [b for b in gy if b["probe_moved"] / b["n"]
               <= b["model_moved"] / max(b["model_stable_den"], 1)]
        chk("on the designed absence the probe is the less stable of the two readers",
            not inv, f"{len(gy)} cells, probe "
            f"{100*min(b['probe_moved']/b['n'] for b in gy):.0f}-"
            f"{100*max(b['probe_moved']/b['n'] for b in gy):.0f}% against model "
            f"{100*min(b['model_moved']/b['model_stable_den'] for b in gy):.0f}-"
            f"{100*max(b['model_moved']/b['model_stable_den'] for b in gy):.0f}%"
            + (f"; not inverted in {[b['label'] for b in inv]}" if inv else ""))
    # run/shamfollow.py refits the probe itself rather than loading a serialised one, and says it
    # does so "exactly as run/canonpred.py does". That is a second copy of a fitting procedure,
    # which is this cycle's dominant defect class, so the claim is measured: its pre-edit
    # prediction must equal canonpred's on every item the two share.
    S = json.load(open("runs/sham.json")) if os.path.exists("runs/sham.json") else {}
    same, diff = 0, []
    for tag, fams in S.items():
        cp = f"runs/canonpred_{tag}.json"
        if not os.path.exists(cp):
            continue
        CP = json.load(open(cp))
        for fam, r in fams.items():
            if fam not in CP:
                continue
            ref = dict(zip(CP[fam]["ids"], CP[fam]["pred"]))
            for x in r["rows"]:
                if x["id"] in ref:
                    same += 1
                    if str(ref[x["id"]]) != x["probe0"]:
                        diff.append(f"{tag}/{fam}/{x['id']}")
    chk("the sham control's probe is the canonical probe, item for item", not diff,
        f"{same} predictions compared" + (f", differing: {diff[:3]}" if diff else ""))
    # the cross-check that justified scoring only the sham half of the real-image stage has to
    # still have an input, or the justification quietly stops being tested. run/shamfollow.py
    # exits on disagreement; what is asserted here is that the comparison is not empty.
    rescored = 0
    for b in sh:
        g = f"runs/{b['tag']}_sham_gen.jsonl"
        if os.path.exists(g):
            rescored += sum(1 for l in open(g) if not json.loads(l)["id"].endswith("_sham"))
    chk("the base half is re-scored somewhere, so the agreement check has an input",
        rescored > 0, f"{rescored} re-scored base generations across {len(sh)} cells")
    # the size rule, from the measurement rather than from the builder's claim about itself
    if os.path.exists("runs/sham_guards.json"):
        G = json.load(open("runs/sham_guards.json"))["pixels"]
        gated = {f: v for f, v in G.items() if f != "spatial"}
        chk("where the sham's size is chosen, it is never the smaller edit",
            all(v["ge"] == v["n"] for v in gated.values()),
            ", ".join(f"{f} {v['ge']}/{v['n']} (median {v['median']:.2f}x)"
                      for f, v in sorted(gated.items())))

# 23 -- the prediction join's family filter is load-bearing. canon.prediction() now refuses to
# pair a probe and an adaptation measured on different held-out sets, which is enforcement enough
# on its own: it exits, and this harness runs canon.py first. What is NOT self-evident, and what
# this checks, is the filter inside that comparison. A tag's per-item file spans every family its
# dataset holds, so comparing the whole file against one family's ids reports a mismatch on the
# twelve multi-family cells and agreement on the four single-family ones -- a guard that looks
# like it is working, fires constantly, and means nothing. Asserting that the unfiltered form
# really would disagree is what keeps the filter from being "cleaned up" later.
M = json.load(open("runs/p3_lora_matched.json"))
cidx = {(r["model"], r["family"]): r for r in C["real"]}
filt, unfilt, checked = [], [], 0
for m in M:
    c = cidx.get((m["model"], m["family"]))
    cp = f"runs/canonpred_{c['tag']}.json" if c else None
    if not c or not cp or not os.path.exists(cp):
        continue
    f = f"runs/lora_items_{m['tag']}-ep{m['epoch']}.json"
    if not os.path.exists(f):
        continue
    checked += 1
    allids = set(json.load(open(f)))
    want = set(json.load(open(cp))[m["family"]]["ids"])
    if canon.lora_items(m, m["family"]) != want:
        filt.append(f"{m['model']}/{m['family']}")
    if allids != want:
        unfilt.append(f"{m['model']}/{m['family']}")
chk("every adaptation row scores the probe's own held-out items",
    not filt and checked == len(M), f"{checked}/{len(M)} cells"
    + (f", differing: {filt}" if filt else ""))
chk("...and the family filter that establishes it is doing work",
    len(unfilt) > 0, f"without it {len(unfilt)} of {checked} cells would read as a mismatch")

# 24 -- the mask convention, measured from the pixels rather than read from a comment. A
# counterfactual's mask says which pixels were allowed to differ, and gen/real.py's
# paste_instance and gen/chart_real.py's raise_bar both write 255 there, which is what
# gen/verify_real.py reads. gen/build_chart_sham.py wrote the INVERSE for two days behind the
# comment "0 = edited, as the other families" -- self-consistent with its own verifier, so
# nothing failed, and the comment asserting conformity is what made it invisible. A second
# builder then copied the split, writing one polarity for counting and the other for glyph.
# Reading the polarity off the images is the only form of this check that could have caught it.
# One record per (dataset, family) is enough and is the point: the property is a convention a
# builder either follows or inverts for everything it writes, not a per-file risk of corruption.
mp = []
for d in sorted(_glob.glob("data/*/manifest.jsonl")):
    root = os.path.dirname(d)
    recs = [json.loads(l) for l in open(d)]
    by = {x["id"]: x for x in recs}
    seen = set()
    for r in recs:
        if not r.get("cf_of") or r["family"] in seen or r["cf_of"] not in by:
            continue
        m = r.get("cf_mask") or r["image"].replace(".png", "_mask.png")
        if not os.path.exists(os.path.join(root, m)):
            continue
        A = np.asarray(Image.open(os.path.join(root, by[r["cf_of"]]["image"])).convert("RGB"))
        B = np.asarray(Image.open(os.path.join(root, r["image"])).convert("RGB"))
        M = np.asarray(Image.open(os.path.join(root, m)).convert("L"))
        if A.shape != B.shape or M.shape != A.shape[:2]:
            continue
        diff = (A != B).any(2)
        seen.add(r["family"])
        mp.append((f"{os.path.basename(root)}/{r['family']}",
                   int(diff[M == 255].sum()), int(diff[M == 0].sum())))
wrong = [n for n, at255, at0 in mp if at0 > at255]
chk("every counterfactual mask marks the edited pixels with 255", not wrong,
    f"{len(mp)} (dataset, family) masks measured" + (f", inverted: {wrong}" if wrong else ""))

# 25 -- the analysis interpreter. The repository has two: a venv carrying torch for the capture
# and scoring passes, and the interpreter README.md's reproduction path names for everything that
# only reads artefacts. They are not interchangeable. The committed canonpred files reproduce
# under the second and not under the first -- numpy 2.4.6 against 2.5.2 flips one tied prediction
# in q3b4_real_3b/counting and two in ivl_real_3b/glyph, 3 of 1125, moving two cells by 1.3 pp.
# Neither cell is near a threshold, so nothing in the manuscript turns on it; what does turn on it
# is whether a re-run reproduces the committed files, and a stage script that fits a probe under
# whichever interpreter happens to have torch is how it stops. So the separation is structural:
# analysis producers are invoked through $APY, capture and scoring through $PY.
# run/headread.py is deliberately NOT in this set although it fits probes like the rest. It
# loads the released unembedding, so it needs torch and belongs to the capture environment; a set
# that demanded $APY for it would demand an interpreter that cannot run it. Nothing invokes it
# from a driver today, so the contradiction was latent rather than firing. The cost is real and
# named in requirements-analysis.txt: runs/headread.json is the one canonical artefact produced
# under the other numpy, and it feeds tables/headread.tex rather than sitting unread.
ANALYSIS = {"canon.py", "canonpred.py", "cffollow.py", "fit_probes.py",
            "layers.py", "lora_matched.py", "nullcal.py", "p0.py", "p3.py", "probe.py",
            "results_md.py", "shamfollow.py", "splits.py", "fix_chart_scoring.py"}
wrongpy = []
for f in subprocess.run(["git", "ls-files", "*.sh"], capture_output=True, text=True).stdout.split():
    for ln in open(f):
        m = re.search(r"\$(\w+)\s+run/([a-z_0-9]+\.py)", ln)
        if m and m.group(2) in ANALYSIS and m.group(1) != "APY":
            wrongpy.append(f"{f}:{m.group(2)} under ${m.group(1)}")
chk("analysis producers run under the analysis interpreter, not the GPU venv", not wrongpy,
    f"{len(ANALYSIS)} analysis scripts"
    + (f", mis-invoked: {wrongpy[:4]}" if wrongpy else ""))

# 26 -- the check numbering itself. Two blocks were both numbered 12b for several commits, and
# 12c never existed, because the numbers were prose that nothing read -- while run/canon.py
# cross-references one of them by number. A header is "# N -- ", and they must be 1..N, once each,
# in order.
nums = [int(m) for m in re.findall(r"^# (\d+) -- ", open(__file__).read(), re.M)]
chk("the check numbers are unique, gapless and in order",
    nums == list(range(1, len(nums) + 1)),
    f"{len(nums)} headers, 1..{max(nums) if nums else 0}"
    + ("" if nums == sorted(set(nums)) else f", out of order or repeated: {nums}"))

# 27 -- the analysis environment, pinned. Check 25 keeps analysis producers off the GPU venv,
# which is the half that was going wrong silently; it does not make the committed artefacts
# re-derivable, because nothing said which numpy "the analysis interpreter" meant.
# requirements-analysis.txt says it now, and requirements.txt no longer claims to.
#
# The assertion here is the portable half: every third-party module the analysis path imports
# must be named in the pin, so a dependency added later cannot arrive unpinned -- which is the
# drift a version list actually suffers. The running versions are reported and not gated,
# deliberately: out/tables/*.tex and docs/RESULTS.md are byte-identical under both interpreters
# in this repository, so a check that failed a clean clone over a difference that moves no
# reported number would be enforcing something the repository has measured to be false. What a
# mismatch does mean is that verify_provenance.py's "tables match a fresh run" is the check now
# carrying the claim, so the detail line says so instead of passing silently.
LOCAL = {f[:-3] for f in os.listdir("run") if f.endswith(".py")}
ENTRY = ANALYSIS | {"figs.py", "manifest.py", "verify_provenance.py",
                    os.path.basename(__file__)}
seen, imports = set(), {}


def _walk(name):
    """Third-party imports reachable from an entry point, following local modules."""
    if name in seen or not os.path.exists(f"run/{name}"):
        return
    seen.add(name)
    for n in ast.walk(ast.parse(open(f"run/{name}").read())):
        if isinstance(n, ast.Import):
            ms = [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            ms = [n.module.split(".")[0]]
        else:
            continue
        for m in ms:
            if m in LOCAL:
                _walk(m + ".py")
            elif m not in sys.stdlib_module_names and m != "__future__":
                imports.setdefault(m, set()).add(name)


for _e in sorted(ENTRY):
    _walk(_e)
# the module -> distribution mapping is measured, not a hardcoded PIL/sklearn dict that would be
# one more copy of something the installation already knows
dist = packages_distributions()
unpinned = sorted(f"{m} (run/{sorted(imports[m])[0]})" for m in imports
                  if not any(d.lower() in PINS for d in dist.get(m, [m])))
chk("every third-party import of the analysis path is pinned", not unpinned,
    f"{len(imports)} packages over {len(seen)} files"
    + (f", unpinned: {unpinned}" if unpinned else "")
    + (f"; this interpreter differs ({', '.join(ENVDRIFT)}), so verify_provenance.py is what "
       f"establishes the tables still match" if ENVDRIFT else "; this interpreter matches the pin"))

print(f"\n{sum(1 for _, o in ck if o)}/{len(ck)} checks pass")
sys.exit(0 if all(o for _, o in ck) else 1)
