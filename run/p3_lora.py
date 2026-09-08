"""Fine-tuning gain per (model, family), against what the probe read from the frozen states.

The §7 claim was that probe accuracy lower-bounds the fine-tuned result and rank-orders families
correctly, established on four pairs (p=1/24 under a random-ordering null). This assembles every
pair so it can be checked properly -- and checked on the right quantity: raw-accuracy ordering is
partly a ranking of chance levels (5.6% to 50% here), so gain and chance-normalised accuracy are
reported alongside.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from scipy.stats import spearmanr

PROBE_FOR = {("q3b", "real_3b"): "real", ("q3b", "real_chart"): "realchart"}


def best_lora(tag):
    b = (-1, None, None)
    for f in glob.glob(f"runs/lora_items_{tag}-ep*.json"):
        d = json.load(open(f))
        a = np.mean([v["ok"] for v in d.values()])
        if a > b[0]: b = (a, d, int(f.split("-ep")[1].split(".")[0]))
    return b[1], b[2]


def main():
    base = json.load(open("runs/lora_items_base.json")) if os.path.exists("runs/lora_items_base.json") else {}
    rows = []
    for f in sorted(glob.glob("runs/lora_*_real*.json")):
        tag = os.path.basename(f)[len("lora_"):-len(".json")]
        if "_items_" in f or "adapter" in f: continue
        model, data = tag.split("_", 1)
        items, ep = best_lora(f"lora_{tag}")
        if items is None: continue
        pt = PROBE_FOR.get((model, data), f"{model}_{data}")
        pf = f"runs/probes_{pt}.npy"
        if not os.path.exists(pf): print(f"  ! no probes for {tag}"); continue
        P = np.load(pf, allow_pickle=True).item()
        lj = json.load(open(f))
        bacc = lj.get("base_acc") or {}
        for fam, d in P.items():
            ids = [i for i in d["test_ids"] if i in items]
            if not ids: continue
            lo = float(np.mean([items[i]["ok"] for i in ids]))
            rows.append(dict(model=model, data=data, family=fam, n=len(ids),
                             probe=float(d["test_acc"]), lora=lo, epoch=ep))
    # base accuracy comes from the scored generations, the same source p3.py uses
    for r in rows:
        gf = f"runs/{PROBE_FOR.get((r['model'], r['data']), r['model'] + '_' + r['data'])}_gen.jsonl"
        if not os.path.exists(gf): r["base"] = None; continue
        g = [json.loads(l) for l in open(gf)]
        v = [bool(x["gen_correct"]) for x in g if x["family"] == r["family"]]
        r["base"] = float(np.mean(v)) if v else None

    hdr = (f"{'model':6}{'data':12}{'family':10}{'n':>4}{'base':>8}{'probe':>8}{'LoRA':>8}"
           f"{'gain':>8}{'probe>base':>11}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["model"], r["data"], r["family"])):
        if r["base"] is None: continue
        print(f"{r['model']:6}{r['data']:12}{r['family']:10}{r['n']:4d}"
              f"{100*r['base']:7.1f}%{100*r['probe']:7.1f}%{100*r['lora']:7.1f}%"
              f"{100*(r['lora']-r['base']):+8.1f}{100*(r['probe']-r['base']):+11.1f}")

    ok = [r for r in rows if r["base"] is not None]
    lb = [r for r in ok if r["probe"] <= r["lora"] + 1e-9]
    print(f"\nprobe <= fine-tune (the lower-bound claim): {len(lb)}/{len(ok)} pairs")
    pg = np.array([r["probe"] - r["base"] for r in ok])
    lg = np.array([r["lora"] - r["base"] for r in ok])
    rho, p = spearmanr(pg, lg)
    print(f"probe gain vs fine-tune gain:  Spearman rho = {rho:+.3f}  p = {p:.3g}  (n={len(ok)})")
    rho2, p2 = spearmanr([r["probe"] for r in ok], [r["lora"] for r in ok])
    print(f"probe acc  vs fine-tune acc:   Spearman rho = {rho2:+.3f}  p = {p2:.3g}")
    json.dump(ok, open("runs/p3_lora_pairs.json", "w"), indent=1, default=float)
    print("\nwrote runs/p3_lora_pairs.json")


if __name__ == "__main__":
    main()
