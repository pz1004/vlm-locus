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
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_real as C

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

    # stratify: each chart is assigned to whichever of its candidate answers is least filled
    per = defaultdict(list)
    for d in ix:
        vs = {cd["value"] for cd in d["cand"]}
        v = min(vs, key=lambda v: (len(per[v]), v))
        if len(per[v]) < a.n // len(C.VALS) + 4:
            per[v].append((d, [cd for cd in d["cand"] if cd["value"] == v]))
    picked = [x for v in sorted(per) for x in per[v]][: a.n * 2]
    rng.shuffle(picked)

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
        tgt = rng.choice(cd["targets"])
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
