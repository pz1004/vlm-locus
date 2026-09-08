"""Scan every ChartQA image for one that admits an exact bar-value counterfactual.

Cheap table checks first (they reject ~45% without touching an image), then bar detection and
the affine calibration that is the real admission test. Parallel across cores; the result is a
JSON index that gen/build_chart.py turns into a dataset.
"""
from __future__ import annotations
import json, os, sys
from multiprocessing import Pool
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_real as C

R = "data/real/chartqa/ChartQA Dataset"


def one(arg):
    split, fn = arg
    stem = fn[:-4]
    cats, vals = C.read_table(f"{R}/{split}/tables/{stem}.csv")
    if cats is None: return ("table not single-series", None)
    if not all(0 <= v <= 100 for v in vals): return ("values outside 0-100", None)
    if len(set(cats)) != len(cats): return ("duplicate category labels", None)
    try: im = Image.open(f"{R}/{split}/png/{fn}")
    except Exception: return ("image unreadable", None)
    bars, base, col = C.find_bars(im)
    if not bars: return ("no bars detected", None)
    if len(bars) != len(vals): return ("bar/value count mismatch", None)
    cal = C.calibrate(bars, base, vals)
    if cal is None: return ("calibration rejected", None)
    a, b = cal
    cand = []
    for i, v in enumerate(vals):
        if abs(v / 5 - round(v / 5)) > 0.4: continue      # rounding must be unambiguous
        r = int(round(v / 5) * 5)
        tgt = [t for t in range(r + 5, 101, 5)
               if C.raise_bar(im, bars[i], base, a, b, t)[0] is not None]
        if tgt: cand.append(dict(bar=i, cat=cats[i], value=r, raw=v, targets=tgt))
    if not cand: return ("no raisable bar", None)
    return (None, dict(split=split, stem=stem, bars=[list(map(int, b_[:3])) + [list(b_[3])]
                                                     for b_ in bars],
                       baseline=int(base), a=a, b=b, cats=cats, vals=vals, cand=cand))


if __name__ == "__main__":
    jobs = [(s, f) for s in ("train", "val", "test")
            for f in sorted(os.listdir(f"{R}/{s}/png"))]
    print(f"scanning {len(jobs)} charts on {os.cpu_count()} cores", flush=True)
    with Pool(24) as p:
        res = p.map(one, jobs, chunksize=32)
    fails, ok = {}, []
    for why, d in res:
        if d: ok.append(d)
        else: fails[why] = fails.get(why, 0) + 1
    print(f"\nadmitted {len(ok)}/{len(jobs)}  ({100*len(ok)/len(jobs):.1f}%)")
    for k, v in sorted(fails.items(), key=lambda kv: -kv[1]): print(f"  {v:6d}  {k}")
    json.dump(ok, open("runs/chart_index.json", "w"))
    print(f"\nwrote runs/chart_index.json  ({sum(len(d['cand']) for d in ok)} usable bars)")
