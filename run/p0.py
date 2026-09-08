"""P0: decompose LoRA's +7.0 pp over the frozen-state readout ceiling into language and vision.

The 83.2% reference adapts both stacks -- `target_modules` was given as bare suffixes, so PEFT
matched the language model *and* the vision tower's MLPs (192 of 696 adapter tensors). That
confounds "fine-tuning changes the representation" with "fine-tuning learns a deeper readout of
an unchanged one". `lang` and `vis` are the exact complement halves of that run, so together
they say which.

Oracle epoch per config, matching how the 83.2% reference was selected.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from scipy.stats import binomtest

FAM = ["chart", "counting", "spatial", "tracking"]


def best_epoch(tag):
    """Per-item dict at the epoch with the highest pooled accuracy -- the reference's rule."""
    best = (-1, None, None)
    for f in sorted(glob.glob(f"runs/lora_items_{tag}-ep*.json")):
        d = json.load(open(f))
        a = np.mean([v["ok"] for v in d.values()])
        if a > best[0]: best = (a, d, int(f.split("-ep")[1].split(".")[0]))
    return best[1], best[2]


def mcnemar(a, b):
    """Exact binomial on discordant pairs; a, b are aligned 0/1 arrays."""
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    if n01 + n10 == 0: return 1.0, n10, n01
    return binomtest(n10, n01 + n10, 0.5).pvalue, n10, n01


def main():
    cfg = {"base": json.load(open("runs/lora_items_base.json"))}
    ep = {}
    for tag, name in [("lora", "both"), ("loralang", "lang"), ("loravis", "vis")]:
        if not glob.glob(f"runs/lora_items_{tag}-ep*.json"):
            sys.exit(f"! no per-item files for {tag}; is the run finished?")
        cfg[name], ep[name] = best_epoch(tag)
    ids = sorted(set.intersection(*[set(d) for d in cfg.values()]))
    fam = {i: i.rsplit("_", 1)[0] for i in ids}
    ok = {k: np.array([d[i]["ok"] for i in ids]) for k, d in cfg.items()}
    print(f"{len(ids)} items shared by all four configs; oracle epochs {ep}\n")

    hdr = f"{'config':8s} " + "".join(f"{f:>10s}" for f in FAM) + f"{'ALL':>9s}"
    print(hdr); print("-" * len(hdr))
    acc = {}
    for k in ["base", "lang", "vis", "both"]:
        row = [float(ok[k][[fam[i] == f for i in ids]].mean()) for f in FAM]
        acc[k] = dict(zip(FAM, row)); acc[k]["ALL"] = float(ok[k].mean())
        print(f"{k:8s} " + "".join(f"{100*v:9.1f}%" for v in row) + f"{100*acc[k]['ALL']:8.1f}%")

    print("\npaired tests (McNemar exact, discordant pairs)")
    for x, y in [("lang", "both"), ("vis", "both"), ("lang", "vis"),
                 ("lang", "base"), ("vis", "base")]:
        p, n10, n01 = mcnemar(ok[x], ok[y])
        print(f"  {x:5s} vs {y:5s}  {100*acc[x]['ALL']:5.1f}% vs {100*acc[y]['ALL']:5.1f}%   "
              f"{x}-only {n10:3d}  {y}-only {n01:3d}   p = {p:.3g}")

    print("\nper-family: is adapting both stacks better than the better single stack?")
    for f in FAM:
        m = np.array([fam[i] == f for i in ids])
        one = "lang" if ok["lang"][m].mean() >= ok["vis"][m].mean() else "vis"
        pv, n10, n01 = mcnemar(ok["both"][m], ok[one][m])
        print(f"  {f:10s} best single = {one:4s} {100*ok[one][m].mean():5.1f}%   "
              f"both {100*ok['both'][m].mean():5.1f}%   p = {pv:.3g}")

    READOUT = 0.762      # best frozen-state readout, run/readout.py (linear @ final layer)
    b, l, v, t = acc["base"]["ALL"], acc["lang"]["ALL"], acc["vis"]["ALL"], acc["both"]["ALL"]
    # The stacks are redundant, not additive -- each alone recovers most of the gain and `both`
    # is indistinguishable from the better single one, per family and pooled. So a sequential
    # "+x pp language, +y pp vision" split would misdescribe the data and is not printed.
    tot = t - b
    print(f"\nshare of the recoverable gain ({100*b:.1f}% -> {100*t:.1f}%) reached by each route")
    print(f"  frozen linear readout, no training      {100*(READOUT-b)/tot:4.0f}%  ({100*READOUT:.1f}%)")
    print(f"  language only, vision tower FROZEN      {100*(l-b)/tot:4.0f}%  ({100*l:.1f}%)")
    print(f"  vision only, language model FROZEN      {100*(v-b)/tot:4.0f}%  ({100*v:.1f}%)")
    print("\n  Language-only holds the visual representation bit-identical to base, so its gain\n"
          "  is reachable without any change to perception. That is the presence claim, made by\n"
          "  the model itself rather than by a probe.")
    json.dump(dict(acc=acc, epochs=ep, readout=READOUT, n=len(ids)),
              open("runs/p0_3b.json", "w"), indent=1)
    print("\nwrote runs/p0_3b.json")


if __name__ == "__main__":
    main()
