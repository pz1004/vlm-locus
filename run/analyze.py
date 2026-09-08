"""First-pass analysis of a pilot run: difficulty calibration, the measured prior, and the
L category. Reports the quantities that decide whether the design is sound, not the ones
that would look good in a paper."""
import json, sys
from collections import Counter, defaultdict
import numpy as np

rs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot_3b.jsonl")]
fams = sorted({r["family"] for r in rs})
n = len(rs)
print(f"{n} items\n")

print("=" * 78)
print("1. DIFFICULTY CALIBRATION  (generator target: base error 40-60%)")
print("=" * 78)
print(f"{'family':10s} {'n':>4s} {'err all':>8s} {'err easy':>9s} {'err hard':>9s} {'chance':>7s}  verdict")
for f in fams:
    g = [r for r in rs if r["family"] == f]
    e = lambda xs: 100 * (1 - np.mean([x["gen_correct"] for x in xs])) if xs else float("nan")
    ea, eh = e([x for x in g if not x["hard"]]), e([x for x in g if x["hard"]])
    all_e = e(g)
    v = "in band" if 40 <= all_e <= 60 else ("too easy" if all_e < 40 else "too hard")
    print(f"{f:10s} {len(g):4d} {all_e:7.1f}% {ea:8.1f}% {eh:8.1f}% {100*g[0]['chance']:6.1f}%  {v}")

print()
print("=" * 78)
print("2. THE MEASURED PRIOR  (blindfold = same item, mid-grey image)")
print("=" * 78)
for f in fams:
    g = [r for r in rs if r["family"] == f]
    blind_gen = Counter(r["parse_blind"] for r in g)
    acc_b = 100 * np.mean([r["gen_correct_blind"] for r in g])
    print(f"{f:10s} blind accuracy {acc_b:5.1f}%  (chance {100*g[0]['chance']:.1f}%)   "
          f"blind answers: {dict(blind_gen.most_common(4))}")

print()
print("--- does the generator's arm label agree with the MEASURED prior? ---")
print("    (arm is only a knob; if it disagrees, the analysis must use the measurement)")
for f in fams:
    g = [r for r in rs if r["family"] == f]
    lo = {a: np.mean([r["logp_true_blind"] for r in g if r["arm"] == a]) for a in ("canonical", "anti")}
    ok = lo["canonical"] > lo["anti"]
    print(f"    {f:10s} mean log p_blind(true): canonical={lo['canonical']:+.2f}  anti={lo['anti']:+.2f}"
          f"   -> {'agrees' if ok else 'CONTRADICTS the design assumption'}")

print()
print("=" * 78)
print("3. CATEGORY L  (right with no visual evidence) and the evidence margin")
print("=" * 78)
print(f"{'family':10s} {'L share':>8s} {'margin>0':>9s} {'mean margin':>12s} {'margin | wrong':>15s}")
for f in fams:
    g = [r for r in rs if r["family"] == f]
    L = 100 * np.mean([r["gen_correct_blind"] for r in g])
    m = np.array([r["margin_true"] for r in g])
    mw = np.array([r["margin_true"] for r in g if not r["gen_correct"]])
    print(f"{f:10s} {L:7.1f}% {100*np.mean(m>0):8.1f}% {m.mean():+11.2f} {mw.mean():+14.2f}")
print("\n  'margin | wrong' is the quantity LRC routes on: how much the image still raises the")
print("  true answer's log-probability on items the model gets wrong. Above zero means the")
print("  evidence is present but not decisive -- readout territory. At or below zero on a")
print("  wrong item is perception territory.")

print()
print("=" * 78)
print("4. SCREEN (candidate scoring) vs DECIDE (free generation)")
print("=" * 78)
agree = np.mean([r["score_vis"] == r["parse_vis"] for r in rs])
sc = np.mean([r["score_correct"] for r in rs]); gc = np.mean([r["gen_correct"] for r in rs])
print(f"  agreement {100*agree:.1f}%   scoring acc {100*sc:.1f}%   generation acc {100*gc:.1f}%"
      f"   gap {100*(sc-gc):+.1f} pp")
for f in fams:
    g = [r for r in rs if r["family"] == f]
    a = 100 * np.mean([r["score_vis"] == r["parse_vis"] for r in g])
    d = 100 * (np.mean([r["score_correct"] for r in g]) - np.mean([r["gen_correct"] for r in g]))
    print(f"    {f:10s} agreement {a:5.1f}%   scoring-minus-generation {d:+5.1f} pp")
print("\n  A large positive gap means scoring flatters the model: it picks the right answer from")
print("  a menu it would not produce on its own. That is why the plan decides by generation.")

print()
print("=" * 78)
print("5. ROUTING HEADROOM, cheapest possible estimate")
print("=" * 78)
wrong = [r for r in rs if not r["gen_correct"]]
rec = [r for r in wrong if r["score_correct"]]
print(f"  items wrong under generation: {len(wrong)}/{n} ({100*len(wrong)/n:.0f}%)")
print(f"  of those, the answer is already top-1 under scoring: {len(rec)} ({100*len(rec)/max(1,len(wrong)):.0f}%)")
print(f"  of those, margin_true > 0 (image raises the true answer): "
      f"{sum(1 for r in wrong if r['margin_true'] > 0)} ({100*np.mean([r['margin_true']>0 for r in wrong]):.0f}%)")
print("\n  This is not the oracle over corrections -- it is a floor on it, using two conditions")
print("  we already have. A high number here means there is something for a router to recover.")
