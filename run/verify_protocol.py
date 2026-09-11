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
# the gated statistic's name comes from canon.py, not from a copy here. Check 1 below asserts
# properties of "the bound the gate uses", and for three commits it asserted them of
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
chk("the committed canonpred files are reproduced by their producer",
    rc.returncode == 0 and "all reproduce" in rc.stdout,
    rc.stdout.strip().split("\n")[-1] if rc.stdout else "no output")

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
    pos = [b for b in sh if b["gap"] > 0]
    neg = [b for b in sh if b["gap"] <= 0]
    chk("the sham control is scored on the same held-out items as the real edit",
        len({b["n"] for b in sh}) == 1
        and all(b["n"] == sh[0]["n"] for b in sh), f"n={sh[0]['n']} in all {len(sh)} cells")
    chk("cells with a gap follow the real edit and do not move on the sham",
        all(b["follow_real"] > 0.5 and b["probe_moved"] / b["n"] < 0.05 for b in pos),
        f"{len(pos)} cells: follow "
        f"{100*min(b['follow_real'] for b in pos):.0f}-{100*max(b['follow_real'] for b in pos):.0f}%, "
        f"moved at most {max(b['probe_moved'] for b in pos)}/{sh[0]['n']}")
    chk("the cell that moves on the sham is the cell that does not follow the real edit",
        bool(neg) and max(sh, key=lambda b: b["probe_moved"]) is
        max(neg, key=lambda b: b["probe_moved"])
        and min(sh, key=lambda b: b["follow_real"]) in neg,
        f"{neg[0]['label'] if neg else '-'}: moved {neg[0]['probe_moved'] if neg else 0}, "
        f"follows {100*neg[0]['follow_real'] if neg else 0:.0f}%")
    # and the images themselves hold: same answer, exact pixel guard, queried bar untouched
    rs = subprocess.run([PY, "gen/verify_sham.py"], capture_output=True, text=True)
    chk("the sham images pass their own guards", rs.returncode == 0 and "PASS" in rs.stdout,
        rs.stdout.strip().split("\n")[-1] if rs.stdout else "did not run")

# 23 -- the check numbering itself. Two blocks were both numbered 12b for several commits, and
# 12c never existed, because the numbers were prose that nothing read -- while run/canon.py
# cross-references one of them by number. A header is "# N -- ", and they must be 1..N, once each,
# in order.
nums = [int(m) for m in re.findall(r"^# (\d+) -- ", open(__file__).read(), re.M)]
chk("the check numbers are unique, gapless and in order",
    nums == list(range(1, len(nums) + 1)),
    f"{len(nums)} headers, 1..{max(nums) if nums else 0}"
    + ("" if nums == sorted(set(nums)) else f", out of order or repeated: {nums}"))

print(f"\n{sum(1 for _, o in ck if o)}/{len(ck)} checks pass")
sys.exit(0 if all(o for _, o in ck) else 1)
