# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Which artefact feeds which paper number -- measured by running the producers, not asserted.

Anyone asking "where does this number come from?" needs an answer that cannot drift from
the code. A hand-written list drifts on the first refactor, so this records the artefacts each
producer *actually opens*, by wrapping builtins.open and numpy.load for the duration of a run.

It also enforces two things a list cannot:

  * nothing in the canonical load path lives in a quarantine directory. runs/v1_contaminated/
    holds sdpa-captured states under names identical to the live ones (states_3b.npz,
    probes_3b.npy, branches6_test.jsonl, probe_g1.json), so a stray relative path reads
    contaminated data and still produces a plausible table.

  * files that look canonical but are not stay unread. runs/probe_g1_smol.json is the clearest:
    its tracking probe hits 100.0% at layer 2, which is the numeric skew run/capture.py:35-39
    documents, and its name is one character from the canonical smolm tag.

Usage:
    .venv/bin/python run/manifest.py            # print the manifest, verify, exit nonzero on fail
    .venv/bin/python run/manifest.py --json      # machine-readable, for the appendix table
"""
from __future__ import annotations
import argparse, builtins, io, json, os, sys, tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories whose contents must never be read by a producer. stage7.sh/stage8.sh move the
# pre-eager-capture run here; the filenames deliberately match the live ones.
QUARANTINE = ("runs/v1_contaminated/", "runs/superseded/")

# Files that resemble a canonical artefact and are not. Reason is printed with the manifest so
# the entry explains itself rather than needing the commit that added it.
NONCANONICAL = {
    "runs/superseded/probe_g1_smol.json":
        "sdpa capture: tracking probe 100.0% at layer 2 (the numeric skew capture.py:35-39 "
        "documents). The canonical SmolVLM synthetic tag is smolm.",
    "runs/superseded/probe_g1_smol_v2.json":
        "superseded intermediate of the SmolVLM re-capture; canon.py's SYNTH names smolm.",
    "runs/superseded/smol_v2_gen.jsonl":
        "generations for the superseded smol_v2 sweep; the canonical file is smolm_gen.jsonl.",
    "runs/superseded/lora_items_base.json":
        "shared-path base eval, overwritten by every --eval-base run; the surviving copy holds "
        "75 real-chart items. p0.py now reads base from the canonical scored generations.",
}

# The upstream half of the chain. Tracing it would mean re-running capture and the layer sweep
# on the GPU, so it is declared -- but the declaration is checked: every file named here must
# exist, and every artefact the traced producers read must be accounted for by one of these
# stages or by PRODUCES below.
UPSTREAM = [
    ("run/capture.py",  "data/<set>/manifest.jsonl",     "runs/states_<tag>.npz"),
    ("run/score.py",    "data/<set>/manifest.jsonl",     "runs/<tag>_gen.jsonl"),
    ("run/layers.py",   "states + gen",                  "runs/layers_<tag>.json"),
    ("run/fit_probes.py", "states",                      "runs/probes_<tag>.npy"),
    ("run/nullcal.py",  "states + gen",                  "runs/null_<tag>.json"),
    ("run/cfprobe.py",  "probes + counterfactual images", "runs/cfprobe_<tag>.json"),
    ("run/cfcapture.py", "counterfactual images",        "runs/cfstates_<tag>.npz"),
    ("run/cffollow.py", "cfstates + probes",             "runs/cffollow_<tag>.json"),
    ("run/headread.py", "states + gen + unembedding",    "runs/headread.json"),
    ("run/lora.py",     "data + probes",                 "runs/lora_items_<tag>-ep<n>.json"),
    ("run/lora_matched.py", "lora_items_<tag>-ep<n>",    "runs/p3_lora_matched.json"),
]

# every canonical (tag, model, label) triple, mirrored from canon.py so a drift is visible
CANONICAL_TAGS = None      # filled from canon.REAL + canon.SYNTH at run time

# producer -> the outputs it is the sole source of
PRODUCES = {
    "run/canon.py":   ["tables/cells_synthetic.tex", "tables/cells_real.tex",
                       "tables/prediction.tex", "tables/pairs.tex", "tables/bands.tex",
                       "tables/decomp.tex", "tables/headread.tex", "tables/facts.tex"],
    # p0 produces the artefact; canon.py turns it into the table. tab:decomp used to be typed
    # into the manuscript from these numbers, which made p0 its sole source in a weaker sense.
    "run/p0.py":      ["runs/p0_3b.json -> tables/decomp.tex (sec 7)",
                       "tab:readouts LoRA rows (appendix)"],
    "run/figs.py":    ["figs/fig1..fig5"],
}


class Recorder:
    """Record every existing path a block of code opens for reading."""

    def __init__(self):
        self.read: set[str] = set()
        self._open, self._npload = builtins.open, np.load

    def _note(self, path, mode):
        try: p = os.fspath(path)
        except TypeError: return
        if isinstance(p, bytes): p = p.decode()
        if "r" in mode and "+" not in mode:
            ap = os.path.abspath(p)
            if ap.startswith(ROOT) and os.path.exists(ap):
                self.read.add(os.path.relpath(ap, ROOT))

    def __enter__(self):
        rec = self

        def op(file, mode="r", *a, **k):
            rec._note(file, mode)
            return rec._open(file, mode, *a, **k)

        def nl(file, *a, **k):
            rec._note(file, "r")
            return rec._npload(file, *a, **k)

        builtins.open, np.load = op, nl
        return self

    def __exit__(self, *e):
        builtins.open, np.load = self._open, self._npload
        return False


def run_producer(mod, call):
    """Import a producer and run it with writes diverted, returning the paths it read."""
    sys.path.insert(0, os.path.join(ROOT, "run"))
    rec = Recorder()
    tmp = tempfile.mkdtemp(prefix="manifest_")
    env = {"VLM_LOCUS_JSON": os.path.join(tmp, "canon.json"),
           "VLM_LOCUS_TEX": os.path.join(tmp, "tables")}
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    os.makedirs(env["VLM_LOCUS_TEX"], exist_ok=True)
    hush = io.StringIO()
    try:
        with rec:
            m = __import__(mod)
            so, sys.stdout = sys.stdout, hush
            try: call(m)
            finally: sys.stdout = so
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return rec.read


def main(as_json=False):
    os.chdir(ROOT)
    fail = []

    sys.path.insert(0, os.path.join(ROOT, "run"))
    import canon as _c

    read = {}
    read["run/canon.py"] = run_producer(
        "canon", lambda m: m.main(argparse.Namespace(audit=False)))
    read["run/p0.py"] = run_producer("p0", lambda m: m.main())

    # ---- report -------------------------------------------------------------------------
    by_kind = defaultdict(list)
    for prod, paths in read.items():
        for p in sorted(paths):
            if p.startswith("run/") or p.startswith("paper/"): continue
            by_kind[p].append(prod)

    if as_json:
        print(json.dumps(dict(
            produces=PRODUCES,
            reads={k: sorted(v) for k, v in sorted(by_kind.items())},
            noncanonical=NONCANONICAL, quarantine=list(QUARANTINE)), indent=1))
    else:
        print("CANONICAL ARTEFACT MANIFEST")
        print("=" * 78)
        for prod, outs in PRODUCES.items():
            print(f"\n{prod}")
            for o in outs: print(f"    -> {o}")
            if prod in read:
                for p in sorted(read[prod]):
                    if p.startswith("run/") or p.startswith("paper/"): continue
                    print(f"       {p}")
            else:
                print("       (not traced here)")
        print(f"\n{len(by_kind)} artefacts feed the traced producers.")

    # ---- checks -------------------------------------------------------------------------
    for p, prods in by_kind.items():
        if any(p.startswith(q) for q in QUARANTINE):
            fail.append(f"quarantined file in the load path: {p}  (read by {', '.join(prods)})")
        if p in NONCANONICAL:
            fail.append(f"non-canonical file is being read: {p}  (read by {', '.join(prods)})")

    for p in NONCANONICAL:
        if not os.path.exists(p):
            continue                       # already removed; nothing to shadow
    # a quarantine directory that shadows a live filename is only safe while nothing reads it
    shadowed = []
    for q in QUARANTINE:
        if not os.path.isdir(q): continue
        for f in sorted(os.listdir(q)):
            if os.path.join("runs", f) in by_kind: shadowed.append((q + f, "runs/" + f))

    # ---- inventory: nothing in runs/ should be unclassified ------------------------------
    tags = [t for t, _, _ in _c.REAL] + [t for t, _, _ in _c.SYNTH]
    live = {p for p in by_kind}
    orphan = []
    for f in sorted(os.listdir("runs")):
        rp = os.path.join("runs", f)
        if os.path.isdir(rp) or rp in live: continue
        # an upstream artefact is accounted for if a canonical tag owns it
        if any(f.endswith(x) for x in (".npz", "_meta.json")): continue      # states, never read
        if any(t in f for t in tags): continue                               # tag-owned
        orphan.append(f)

    for p in NONCANONICAL:
        if os.path.exists(p) and not any(p.startswith(q) for q in QUARANTINE):
            fail.append(f"non-canonical file sits outside the quarantine directories: {p}")

    # ---- provenance invariants a file list cannot express --------------------------------
    # (a) no producer may read a path that another producer overwrites. The base eval was the
    #     one such path: run/lora.py --eval-base wrote runs/lora_items_base.json on every run.
    import glob as _g, re as _re
    for f in sorted(_g.glob("run/*.py")):
        if os.path.basename(f) == "manifest.py": continue
        t = open(f).read()
        if _re.search(r'open\(\s*f?["\']runs/lora_items_base\.json', t):
            fail.append(f"{f} reads the shared-path base eval; use the canonical generations")
        if _re.search(r'evaluate\(.*["\']base["\']\s*\)', t):
            fail.append(f"{f} writes the untagged 'base' per-item file, which every run clobbers")

    # (b) run/p0.py's base row must be the same measurement as canon.py's model column, not an
    #     independently re-run one -- that is the whole point of reading the canonical file.
    try:
        import p0 as _p0
        Pr = np.load("runs/probes_3b.npy", allow_pickle=True).item()
        Gb = _p0.base_items()
        Lv = json.load(open("runs/layers_3b.json"))
        if _p0.BASE_GEN != _c.GEN["3b"]:
            fail.append(f"p0.BASE_GEN ({_p0.BASE_GEN}) has drifted from canon.GEN['3b'] "
                        f"({_c.GEN['3b']})")
        for fam in Pr:
            ids = [i for i in Pr[fam]["test_ids"] if i in Gb]
            a = float(np.mean([Gb[i]["ok"] for i in ids]))
            if abs(a - Lv[fam]["model"]) > 1e-9:
                fail.append(f"p0 base != canon model_acc on {fam}: "
                            f"{100*a:.1f}% vs {100*Lv[fam]['model']:.1f}%")
    except Exception as e:                          # a missing artefact is reported, not raised
        fail.append(f"could not check p0 base provenance: {e}")

    if not as_json:
        print(f"\nupstream chain (declared; tracing it needs the GPU capture)")
        for prod, src, out in UPSTREAM:
            print(f"    {prod:24} {src:34} -> {out}")
        print(f"\n{len(tags)} canonical tags: {', '.join(tags)}")
        if orphan:
            print(f"\n{len(orphan)} artefacts in runs/ owned by no canonical tag "
                  f"(historical sweeps, not read):")
            for f in orphan[:12]: print(f"    runs/{f}")
            if len(orphan) > 12: print(f"    ... and {len(orphan)-12} more")
        if shadowed:
            print("\nquarantined files shadowing a canonical name (safe: unread, but do not "
                  "run a producer from inside these directories)")
            for a, b in shadowed: print(f"    {a:44} shadows {b}")
        print("\nnon-canonical lookalikes (must stay out of the load path)")
        for p, why in sorted(NONCANONICAL.items()):
            print(f"    {p}{'' if os.path.exists(p) else '   [absent]'}\n        {why}")
        print("\n" + ("FAIL\n  " + "\n  ".join(fail) if fail else "PASS: "
              "no quarantined or non-canonical artefact is in the load path"))
    return 1 if fail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    sys.exit(main(p.parse_args().json))
