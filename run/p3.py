"""Assemble every (model, family) pair measured anywhere in the study into one table.

The §7 claim -- that probe accuracy lower-bounds and rank-orders what fine-tuning achieves --
rested on four pairs and p=1/24 under a random-ordering null. That is an anecdote. This collects
all pairs across models and datasets so the claim can be tested as a regression instead.

Every row carries the three numbers needed to interpret it: what the model emits, what the probe
reads from the frozen final layer, and whether the probe follows a counterfactual edit. A gap
without a follow rate is not evidence, which is the lesson the glyph control taught.
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np

# Apparent gap when the attribute is absent, measured on the designed negative control (glyph at
# 3 px). It matters WHICH gap: taking the best of 37 layers inflates far more than reading the
# final layer, so a final-layer gap must not be judged against a peak-derived threshold.
#   peak  : Qwen +8.6, SmolVLM +8.3          -> ~8.5
#   final : Qwen +2.0, q7b +2.3, InternVL -1.0, q3b4 -2.0, SmolVLM +7.0  -> ~2, SmolVLM an outlier
#
# That last line is the whole problem with a single constant: it was set to ~2.5 by treating
# SmolVLM as an outlier rather than as data. The threshold is now per cell, from
# run/nullcal.py's label-permutation null, and SmolVLM's glyph cell is exactly the one that
# clears it (12.0% against a 10.7% null 95th percentile, p=0.030) -- a false positive on the
# family whose attribute is absent by construction, which is what the null exists to catch.
# ARTEFACT_PEAK is kept: it prices layer selection, which is a different quantity.
ARTEFACT_PEAK = 8.5
NULL_ALPHA = 0.05
FOLLOW_MIN = 0.50


def wilson_lo(k, m, z=1.96):
    """Lower Wilson bound. The follow denominator is conditional -- items the reader answered
    correctly *before* the edit -- and falls to 3-9 of 75 on the designed-absence family, so the
    gate is applied to the bound rather than to the point estimate. run/canon.py does the same;
    the two verdict paths must not disagree, which is how the retired 2.5 pp constant went wrong."""
    if not m: return float("nan")
    ph, d = k / m, 1 + z * z / m
    return max(0.0, (ph + z * z / (2 * m)) / d
               - z * np.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m)) / d)


def null_ok(tag, fam):
    """Does this cell's final-layer probe clear its own permutation null?"""
    f = f"runs/null_{tag}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)).get(fam)
    return None if d is None else bool(d["p_null"] < NULL_ALPHA)


def rows():
    out = []
    for f in sorted(glob.glob("runs/layers_*.json")):
        tag = os.path.basename(f)[len("layers_"):-len(".json")]
        d = json.load(open(f))
        cf = {}
        cfp = f.replace("layers_", "cfprobe_")
        if os.path.exists(cfp):
            for r in json.load(open(cfp)):
                cf.setdefault(r["family"], []).append(r)
        for fam, v in d.items():
            vis = v["vis"]
            c = cf.get(fam, [])
            hit = [r for r in c if r["probe0"] == r["a0"]]
            mhit = [r for r in c if r["model0"] == r["a0"]]
            out.append(dict(
                tag=tag, family=fam, n=v.get("n_test"), chance=v.get("chance"),
                model=v.get("model"), final=vis[-1], peak=max(vis),
                peak_layer=int(np.argmax(vis)),
                follow=(np.mean([r["probe1"] == r["a1"] for r in hit]) if hit else None),
                follow_num=int(sum(r["probe1"] == r["a1"] for r in hit)), follow_den=len(hit),
                follow_lo=(wilson_lo(sum(r["probe1"] == r["a1"] for r in hit), len(hit))
                           if hit else None),
                mfollow=(np.mean([r["model1"] == r["a1"] for r in mhit]) if mhit else None),
                n_cf=len(c)))
    return out


def main():
    rs = rows()
    hdr = (f"{'model/data':<22}{'family':<10}{'n':>4}{'chance':>8}{'model':>8}{'probe':>8}"
           f"{'gap':>8}{'follow':>8}{'verdict':>10}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rs, key=lambda r: (r["tag"], r["family"])):
        if r["model"] is None: continue
        gap = 100 * (r["final"] - r["model"])
        fol = r["follow"]
        # a pair counts as a readout gap only if the probe clears its own permutation null,
        # reads above what the model emits, AND tracks the counterfactual edit -- the last judged
        # by the follow rate's lower confidence bound. Without the last condition a correlate
        # reader passes; without the first, an artefact does; on the point estimate alone, a
        # denominator of five clears the gate on one item.
        nk = null_ok(r["tag"], r["family"])
        lo = r.get("follow_lo")
        v = ("READOUT" if nk and gap > 0 and lo is not None and lo > FOLLOW_MIN
             else "-" if fol is None else "no gap")
        print(f"{r['tag']:<22}{r['family']:<10}{r['n'] or 0:4d}{100*(r['chance'] or 0):7.1f}%"
              f"{100*r['model']:7.1f}%{100*r['final']:7.1f}%{gap:+8.1f}"
              f"{(f'{100*fol:.0f}%' if fol is not None else '-'):>8}{v:>10}")
    json.dump(rs, open("runs/p3_pairs.json", "w"), indent=1, default=float)
    n = sum(1 for r in rs if r["model"] is not None)
    print(f"\n{n} (model, family) pairs   ->  runs/p3_pairs.json")


if __name__ == "__main__":
    main()
