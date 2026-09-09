# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Build the real chart family from the scan index, in the standard manifest schema.

One item per chart -- never two bars from the same image, so a near-duplicate cannot straddle the
probe's train/test split (the defect that invalidated synthetic tracking). Answers are stratified
across the 21-value space because ChartQA's natural distribution is skewed low, and a skewed
answer set makes a majority-class guesser look like a reader.

The counterfactual raises the queried bar to a different multiple of 5, painting only the added
rectangle; gen/verify_real.py check [4] then requires every pixel outside that rectangle to be
bit-identical.

On the top of the value range, and why it cannot be fixed here. A raise-only edit needs a target
above the original, so the highest target value can never itself be an original -- and a value
that is never an original is not a class the probe is fitted on, so an edit into it is
unfollowable however well the representation encodes the change. Excluding the top value does not
help: it makes the next one down unbuildable and moves the same gap, and measuring that showed it
moves the gap the wrong way (43 affected items instead of 7, because the tail is thinner than the
head). The gap is structural to raising bars, and the fix is a bidirectional edit -- render the
pair as (raised -> stored original) so the target is always an attainable class. Until then
run/canon.py reports the reverse and joint statistics, which do not have the defect, beside the
forward rate that does.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import Counter, defaultdict
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "run"))
import chart_real as C
from canon import MIN_CLASS

R = "data/real/chartqa/ChartQA Dataset"


def rec(iid, image, question, answer, level, cf_of=None, referents=None, extra=None, coco=None):
    return dict(id=iid, family="chart", image=image, question=question, answer=str(answer),
                answer_space=C.VALS, chance=1.0 / len(C.VALS),
                attribute=dict(value=int(answer)), referents=referents or [],
                difficulty=dict(hard=False, level=level, **(extra or {})),
                prior_arm="single", cf_kind="raise_bar", cf_of=cf_of, spec={}, coco=coco or {})


def main(a):
    rng = random.Random(a.seed)
    ix = json.load(open(a.index))
    rng.shuffle(ix)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    # stratify: each chart goes to whichever of its candidate answers is least filled, and ties
    # break toward the value with the *scarcest global supply*. Breaking them toward the smaller
    # value instead starves the top of the range -- ChartQA offers 317 charts at 5 and 20 at 95,
    # so a chart offering both was going to 5 and 95 never filled.
    supply = defaultdict(int)
    for d in ix:
        for v in {cd["value"] for cd in d["cand"]}:
            supply[v] += 1
    per = defaultdict(list)
    for d in ix:
        vs = {cd["value"] for cd in d["cand"]}
        v = min(vs, key=lambda v: (len(per[v]), supply[v], v))
        if len(per[v]) < a.n // len(C.VALS) + 4:
            per[v].append((d, [cd for cd in d["cand"] if cd["value"] == v]))
    picked = [x for v in sorted(per) for x in per[v]][: a.n * 2]
    rng.shuffle(picked)

    # The probe is fitted on the values this set actually contains, at or above the rare-class
    # filter run/layers.py applies, so an edit target outside that set is unfollowable however
    # well the representation encodes it. Restrict targets to it rather than discover the gap
    # downstream: with the old builder 34 counterfactuals asked for 95 and no original had it.
    intended = Counter(v for v in per for _ in per[v])
    attainable = {v for v, c in intended.items() if c >= MIN_CLASS}

    man, made, skipped = [], 0, 0
    for d, cands in picked:
        if made >= a.n: break
        cd = cands[0]
        im = Image.open(f"{R}/{d['split']}/png/{d['stem']}.png").convert("RGB")
        bars = [(b[0], b[1], b[2], tuple(b[3])) for b in d["bars"]]
        # erase the printed values from every bar BEFORE deriving the counterfactual, so the
        # task is reading the bar against the axis (as in the synthetic family) rather than OCR,
        # and the raised bar is not the only unlabelled one
        im = C.strip_labels(im, bars)
        if C.has_data_labels(im, bars): skipped += 1; continue
        # prefer a target the probe can emit, and fall back rather than lose the item: only the
        # top of the range has no attainable target, and dropping those items costs more than
        # the handful of unfollowable counterfactuals it would save (see the module docstring)
        tgt_ok = [g for g in cd["targets"] if int(g) in attainable] or cd["targets"]
        tgt = rng.choice(tgt_ok)
        cf, mask = C.raise_bar(im, bars[cd["bar"]], d["baseline"], d["a"], d["b"], tgt)
        if cf is None: skipped += 1; continue
        iid = f"chart_{made:06d}"
        p0 = f"images/{iid}.png"; im.save(os.path.join(a.out, p0))
        q = (f"What is the value of the bar labelled \"{cd['cat']}\"? "
             f"Answer with the nearest multiple of 5, between 0 and 100.")
        man.append(rec(iid, p0, q, cd["value"], 0, referents=[cd["cat"]],
                       extra=dict(n_bars=len(bars), raw=cd["raw"], labels_stripped=True),
                       coco=dict(source=f"chartqa/{d['split']}/{d['stem']}")))
        p1 = f"images/{iid}_cf.png"; cf.save(os.path.join(a.out, p1))
        mask.save(os.path.join(a.out, f"images/{iid}_cf_mask.png"))
        man.append(rec(f"{iid}_cf", p1, q, tgt, 0, cf_of=iid, referents=[cd["cat"]],
                       extra=dict(n_bars=len(bars)),
                       coco=dict(source=f"chartqa/{d['split']}/{d['stem']}")))
        made += 1
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as fh:
        for r in man: fh.write(json.dumps(r) + "\n")
    ans = defaultdict(int)
    for r in man:
        if not r["cf_of"]: ans[r["answer"]] += 1
    print(f"wrote {a.out}/manifest.jsonl   {made} items + {made} counterfactuals "
          f"({skipped} skipped)")
    print(f"  {len(ans)} distinct answers, most common "
          f"{100*max(ans.values())/made:.0f}%  (chance {100/len(C.VALS):.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="runs/chart_index.json")
    p.add_argument("--out", default="data/real_chart")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
