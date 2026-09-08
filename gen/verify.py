# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Independent verification of a generated set.

Nothing here calls the generator's own label functions. Every answer is re-derived from the
spec by separate logic, and the rendered pixels are checked against the spec where that is
possible. A generator whose labels are wrong is worse than no generator, so this runs before
any GPU time is spent.
"""
import hashlib
import json, sys, os
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import PALETTE, RENDER, W, H          # renderers only, no label code

FAIL = []
def check(cond, msg):
    if not cond: FAIL.append(msg)
    return cond

# --- independent re-derivation of each family's answer, written from the task definition ---
def answer_counting(s): return str(len(s["targets"]))
def answer_spatial(s):
    if s["axis"] == "horizontal": return "left" if s["ba"][0] < s["bb"][0] else "right"
    return "above" if s["ba"][1] < s["bb"][1] else "below"
def answer_chart(s): return str(s["vals"][s["q"]])
def answer_tracking(s):
    p = s["start"]
    for i, j in s["swaps"]:
        p = j if p == i else (i if p == j else p)
    return str(p + 1)
ANS = dict(counting=answer_counting, spatial=answer_spatial, chart=answer_chart,
           tracking=answer_tracking)

def main(root):
    recs = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    by_id = {r["id"]: r for r in recs}
    base = [r for r in recs if r["cf_of"] is None]
    cfs  = [r for r in recs if r["cf_of"] is not None]
    print(f"{len(recs)} records: {len(base)} base + {len(cfs)} counterfactual\n")

    # 1 ---------- labels re-derived from spec
    bad = [r["id"] for r in recs if ANS[r["family"]](r["spec"]) != r["answer"]]
    check(not bad, f"label mismatch on {bad[:5]}")
    print(f"[1] labels re-derived independently .......... {len(recs)-len(bad)}/{len(recs)} agree")

    # 2 ---------- answer space, chance, and answer inside space
    off = [r["id"] for r in recs if r["answer"] not in r["answer_space"]]
    ch  = [r["id"] for r in recs if abs(r["chance"] - 1/len(r["answer_space"])) > 1e-6]
    check(not off, f"answer outside its space: {off[:5]}")
    check(not ch,  f"chance != 1/|space|: {ch[:5]}")
    print(f"[2] answer in space / chance correct ......... {len(recs)-len(off)}/{len(recs)}, "
          f"{len(recs)-len(ch)}/{len(recs)}")

    # 3 ---------- prior arm: the answer SPACE must be identical across arms. Whether the
    # arm actually shifts the prior is not checkable here -- it is a property of the model,
    # measured from the blindfold run (run/analyze.py section 2). The pilot showed the arm
    # label contradicting the measured prior for chart and tracking, which is why the
    # analysis conditions on measured log p_blind(true) and treats the arm only as a knob
    # for producing spread.
    print("[3] prior arms (answer space must be identical across arms; separation is measured, not asserted):")
    for fam in sorted({r["family"] for r in base}):
        rs = [r for r in base if r["family"] == fam]
        canon = [r for r in rs if r["prior_arm"] == "canonical"]
        anti  = [r for r in rs if r["prior_arm"] == "anti"]
        sc = {tuple(r["answer_space"]) for r in canon}; sa = {tuple(r["answer_space"]) for r in anti}
        ok = check(sc == sa, f"{fam}: answer space differs between arms")
        print(f"    {fam:9s} canonical={sorted({r['answer'] for r in canon})[:6]}  "
              f"anti={sorted({r['answer'] for r in anti})[:6]}   "
              f"spaces identical across arms: {'yes' if sc == sa else 'NO'}"
              + ("" if len(sc) == 1 else f" ({len(sc)} sizes, varies with n_cats/n_cups)"))

    # 4 ---------- counterfactual: answer changes, and by the stated amount
    print("[4] counterfactual pairs:")
    per = defaultdict(lambda: [0, 0])
    for c in cfs:
        b = by_id[c["cf_of"]]; fam = c["family"]; per[fam][0] += 1
        if fam == "counting":   ok = int(c["answer"]) == int(b["answer"]) - 1
        elif fam == "spatial":  ok = c["answer"] != b["answer"]
        elif fam == "chart":    ok = abs(int(c["answer"]) - int(b["answer"])) == 10
        else:                   ok = True                       # tracking: may or may not move
        per[fam][1] += bool(ok)
        check(ok, f"cf did not move the answer as stated: {c['id']}")
    for fam, (n, k) in sorted(per.items()):
        print(f"    {fam:9s} {k}/{n} moved the answer as the edit specifies")
    moved = sum(1 for c in cfs if c["answer"] != by_id[c["cf_of"]]["answer"])
    print(f"    tracking edits that changed the answer: "
          f"{sum(1 for c in cfs if c['family']=='tracking' and c['answer']!=by_id[c['cf_of']]['answer'])}"
          f"/{per['tracking'][0]}  (dropping a swap need not move the ball)")

    # 5 ---------- minimality: outside the edited region the pixels must be identical
    print("[5] counterfactual minimality (pixel diff outside the edited region):")
    for fam in ("counting", "spatial", "chart", "tracking"):
        ex = [c for c in cfs if c["family"] == fam][:8]
        worst = 0.0; region = None
        for c in ex:
            b = by_id[c["cf_of"]]
            A = np.asarray(Image.open(os.path.join(root, b["image"])), np.int16)
            B = np.asarray(Image.open(os.path.join(root, c["image"])), np.int16)
            diff = (np.abs(A - B).sum(2) > 12)
            if fam == "counting":
                x0, y0, x1, y1 = b["referents"][-1]; m = np.zeros_like(diff); m[y0:y1+1, x0:x1+1] = 1
            elif fam == "spatial":
                m = np.zeros_like(diff)
                for x0, y0, x1, y1 in b["referents"]: m[y0:y1+1, x0:x1+1] = 1
            else:
                m = np.ones_like(diff)
            outside = float((diff & ~m.astype(bool)).mean())
            worst = max(worst, outside); region = "removed target" if fam=="counting" else (
                "the two object boxes" if fam=="spatial" else "whole image (layout changes)")
        print(f"    {fam:9s} max {worst*100:6.3f}% of pixels differ outside {region}")
        if fam in ("counting", "spatial"):
            check(worst < 0.01, f"{fam}: counterfactual is not a minimal edit ({worst:.3%} outside)")

    # 6 ---------- renderer honesty: targets actually drawn, counted from pixels
    print("[6] counting renderer checked against pixels (connected components of target colour):")
    exact = split = merged = 0; N = 0
    for r in [x for x in base if x["family"] == "counting"][:40]:
        s = r["spec"]; rgb = np.array(PALETTE[s["color"]], np.int16); N += 1
        blank = np.asarray(RENDER["counting"](dict(s, distractors=[])), np.int16)
        mask = (np.abs(blank - rgb).sum(2) < 40)
        _, n = ndimage.label(mask)
        k = len(s["targets"])
        if n == k: exact += 1
        elif n < k: merged += 1
        else: split += 1
    # Overlap makes a pixel count an unreliable oracle in BOTH directions: overlapping targets
    # merge into one component, and a later shape's dark outline drawn across an earlier one
    # splits it into two. The number of targets drawn is guaranteed by the render loop; this
    # check is a secondary sanity test, so it reports the distribution rather than asserting
    # equality on items that permit overlap.
    print(f"    {exact}/{N} exactly N components; {merged} merged (overlapping targets); "
          f"{split} split (outline crossing a target)")
    check(exact + merged + split == N, "counting render check did not run on every item")

    # 6b --------- referential uniqueness: the question must name exactly one object
    print("[6b] referent uniqueness (a question naming a colour+shape must match one object):")
    amb = []
    for r in base:
        sp = r["spec"]
        if r["family"] == "spatial":
            tgt = {(sp["ca"], sp["sa"]), (sp["cb"], sp["sb"])}
            if any((c, sh) in tgt for _b, c, sh in sp["clutter"]): amb.append(r["id"])
        elif r["family"] == "counting":
            if any(c == sp["color"] and sh == sp["shape"] for _b, c, sh in sp["distractors"]):
                amb.append(r["id"])
    check(not amb, f"ambiguous referents: {amb[:5]}")
    print(f"    {len(base)-len(amb)}/{len(base)} unambiguous"
          + (f"   AMBIGUOUS: {amb[:5]}" if amb else ""))

    # 7 ---------- determinism: rebuild one item from its spec and compare bytes
    print("[7] determinism (re-render from spec, compare to the stored PNG):")
    same = 0
    for r in recs[:24]:
        a = np.asarray(Image.open(os.path.join(root, r["image"])))
        b = np.asarray(RENDER[r["family"]](r["spec"]))
        same += int(a.shape == b.shape and (a == b).all())
    print(f"    {same}/24 re-render bit-identical")
    check(same == 24, f"only {same}/24 re-rendered identically")

    # 8 ---------- difficulty spread
    print("[8] difficulty and label spread:")
    for fam in sorted({r["family"] for r in base}):
        rs = [r for r in base if r["family"] == fam]
        h = sum(r["difficulty"]["hard"] for r in rs)
        print(f"    {fam:9s} hard={h}/{len(rs)}  distinct answers={len(Counter(r['answer'] for r in rs))}"
              f"  chance={rs[0]['chance']:.3f}")

    # 9 ---------- duplicate items: the leak that made the tracking probe score 100% on
    # memorised twins and 30% on clean ones. Two items that render to the same pixels are the
    # same item, and any split that puts one in train and the other in test is contaminated.
    print("[9] duplicate items (identical rendered images):")
    for fam in sorted({r["family"] for r in base}):
        rs = [r for r in base if r["family"] == fam]
        h = Counter(hashlib.sha256(open(os.path.join(root, r["image"]), "rb").read()).hexdigest()
                    for r in rs)
        dup = sum(c - 1 for c in h.values() if c > 1)
        biggest = max(h.values())
        print(f"    {fam:9s} {len(h)}/{len(rs)} distinct images"
              + (f"   DUPLICATES: {dup} extra copies, largest group {biggest}" if dup else ""))
        check(dup == 0, f"{fam}: {dup} duplicate images -- the spec space is smaller than the "
                        f"sample, so any train/test split leaks")

    print()
    if FAIL:
        print(f"FAILED {len(FAIL)} check(s):")
        for m in FAIL[:10]: print("   -", m)
        return 1
    print("all checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/smoke"))
