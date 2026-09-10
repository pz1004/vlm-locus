# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Both directions of the existing chart pairs, from the images already rendered.

gen/build_chart.py raises a bar, so every edit increases the value and the highest target can
never itself be an original -- which makes it a class the probe is not fitted on and an edit into
it unfollowable however well the representation encodes the change. On the 466-item family that
is 16 counterfactuals asking for 100, a value no chart in the scan index carries.

Reversing an edit needs no inpainting, which is the point: the stored original *is* the lowered
image. So each pair is emitted twice, once in each direction, and every image serves as an
original in one row and as the counterfactual of its partner in the other. Three things follow:

  * the edit set is bidirectional, half raising and half lowering, so a probe that tracks only
    increases is distinguishable from one that tracks the value;
  * the probe is fitted on raised bars as well as unraised ones, so an edited image is no longer
    out of distribution at scoring time -- one of the competing explanations the paper cannot
    currently separate from "the probe reads a correlate";
  * pooling both roles gives 21 classes at or above the rare-class filter including 100, so no
    counterfactual target sits outside the probe's class support. The defect is gone by
    construction rather than reported.

The two members of a pair differ in one bar and are otherwise identical, so they must not
straddle the probe's train/test split. This writes a pair key into each record; run/layers.py's
split is by item, so the consumer has to group on it -- which is why the key is in the manifest
rather than left implicit.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_real as C


def main(a):
    src = [json.loads(l) for l in open(os.path.join(a.src, "manifest.jsonl"))]
    orig = {r["id"]: r for r in src if not r["cf_of"]}
    cf = [r for r in src if r["cf_of"]]
    os.makedirs(a.out, exist_ok=True)

    man, n_up, n_down = [], 0, 0
    for c in cf:
        o = orig[c["cf_of"]]
        pair = o["id"]
        for lo, hi, direction in ((o, c, "raise"), (c, o, "lower")):
            iid = f"{lo['id']}__{direction}"
            # which member carries a painted bar, declared rather than inferred: a painted
            # bar's value is not in the source table, so gen/verify_real.py's check [2] cannot
            # re-derive it and gen/verify_geom.py must. Marking it here is what lets the two
            # checks cover every item between them instead of each covering half.
            rec = dict(lo, id=iid, cf_of=None, cf_kind=f"{direction}_bar",
                       synthetic_bar=bool(lo is c),
                       difficulty=dict(lo["difficulty"], pair=pair, direction=direction))
            man.append(rec)
            # both directions share the pair's one mask; name it, because deriving the path
            # from the counterfactual's image only works in the raising direction
            man.append(dict(hi, id=f"{iid}_cf", cf_of=iid, cf_kind=f"{direction}_bar",
                            cf_mask=f"images/{pair}_cf_mask.png",
                            synthetic_bar=bool(hi is c),
                            difficulty=dict(hi["difficulty"], pair=pair, direction=direction)))
            n_up += direction == "raise"
            n_down += direction == "lower"

    with open(os.path.join(a.out, "manifest.jsonl"), "w") as fh:
        for r in man:
            fh.write(json.dumps(r) + "\n")
    items = [r for r in man if not r["cf_of"]]
    vals = {}
    for r in man:
        vals[int(r["answer"])] = vals.get(int(r["answer"]), 0) + 1
    kept = {v for v, n in vals.items() if n >= 8}
    bad = sorted({int(r["answer"]) for r in man if r["cf_of"]} - kept)
    print(f"wrote {a.out}/manifest.jsonl   {len(items)} items + {len(items)} counterfactuals")
    print(f"  {n_up} raising, {n_down} lowering, {len(set(r['difficulty']['pair'] for r in items))}"
          f" source pairs")
    print(f"  {len(kept)} classes at or above the filter, targets outside it: {bad or 'none'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/real_chart_v2")
    p.add_argument("--out", default="data/real_chart_bidir")
    main(p.parse_args())
