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
ARTEFACT_PEAK = 8.5
ARTEFACT_FINAL = 2.5


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
        # a pair counts as a readout gap only if the final-layer gap clears the final-layer
        # artefact AND the probe tracks the counterfactual edit. The glyph control supplies the
        # first threshold; without the second, a correlate reader passes.
        v = ("READOUT" if gap > ARTEFACT_FINAL and fol is not None and fol > 0.5
             else "-" if fol is None else "no gap")
        print(f"{r['tag']:<22}{r['family']:<10}{r['n'] or 0:4d}{100*(r['chance'] or 0):7.1f}%"
              f"{100*r['model']:7.1f}%{100*r['final']:7.1f}%{gap:+8.1f}"
              f"{(f'{100*fol:.0f}%' if fol is not None else '-'):>8}{v:>10}")
    json.dump(rs, open("runs/p3_pairs.json", "w"), indent=1, default=float)
    n = sum(1 for r in rs if r["model"] is not None)
    print(f"\n{n} (model, family) pairs   ->  runs/p3_pairs.json")


if __name__ == "__main__":
    main()
