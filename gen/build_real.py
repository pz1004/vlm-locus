"""Build the real-image datasets, in the manifest schema run/capture.py already consumes.

Every counterfactual here is exact, and the exactness is checked rather than asserted:

  spatial   flip(counterfactual) must be bit-identical to the original
  counting  every pixel outside the pasted silhouette must be bit-identical
  glyph     every pixel outside the glyph box must be bit-identical

gen/verify_real.py re-derives all of that from the written PNGs.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, sys
from collections import defaultdict
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real as R

GLYPH_VALS = [str(v) for v in range(10, 100, 5)]      # 18 classes, as the synthetic chart family


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rec(iid, fam, image, question, answer, space, attribute, level, cf_kind,
        cf_of=None, referents=None, extra=None, coco=None):
    return dict(id=iid, family=fam, image=image, question=question, answer=str(answer),
                answer_space=[str(s) for s in space], chance=1.0 / len(space),
                attribute=attribute, referents=referents or [],
                difficulty=dict(hard=False, level=level, **(extra or {})),
                prior_arm="single", cf_kind=cf_kind, cf_of=cf_of, spec={},
                coco=coco or {})


def main(a):
    rng = random.Random(a.seed)
    imdir = os.path.join(a.out, "images"); os.makedirs(imdir, exist_ok=True)
    cats, imgs, per = R.load_coco(a.coco)
    src = lambda f: os.path.join(a.coco, "val2017", f)
    man, stats, used = [], {}, set()
    fams = a.families.split(",")
    lv = {f: int(v) for f, v in (x.split("=") for x in a.levels.split(","))} if a.levels else {}

    if "spatial" in fams:
        L = lv.get("spatial", 0); made = 0
        for s in R.gen_spatial(cats, imgs, per, L, a.n * 4, rng):
            if made >= a.n: break
            if s["file"] in used: continue
            used.add(s["file"])
            iid = f"spatial_{made:06d}"
            im = Image.open(src(s["file"])).convert("RGB")
            p0 = f"images/{iid}.png"; im.save(os.path.join(a.out, p0))
            q = (f"Where is the {s['ca']} relative to the {s['cb']}? "
                 f"Answer with one word: left or right.")
            man.append(rec(iid, "spatial", p0, q, s["answer"], ["left", "right"],
                           dict(relation=s["answer"]), L, "hflip",
                           referents=[s["ca"], s["cb"]], extra=dict(sep=s["sep"]),
                           coco=dict(image_id=s["image_id"], subj=s["ca"], obj=s["cb"])))
            if made < a.n_cf:
                cf = im.transpose(Image.FLIP_LEFT_RIGHT)
                p1 = f"images/{iid}_cf.png"; cf.save(os.path.join(a.out, p1))
                flip = "right" if s["answer"] == "left" else "left"
                man.append(rec(f"{iid}_cf", "spatial", p1, q, flip, ["left", "right"],
                               dict(relation=flip), L, "hflip", cf_of=iid,
                               referents=[s["ca"], s["cb"]], extra=dict(sep=s["sep"])))
            made += 1
        stats["spatial"] = made

    if "counting" in fams:
        L = lv.get("counting", 0); made = 0
        donors = defaultdict(list)
        for iid_, anns in per.items():
            for an in anns:
                if isinstance(an.get("segmentation"), list) and an["bbox"][2] >= 12 \
                        and an["bbox"][3] >= 12:
                    donors[an["category_id"]].append((iid_, an))
        for s in R.gen_counting(cats, imgs, per, L, a.n * 4, rng):
            if made >= a.n: break
            if s["file"] in used: continue
            used.add(s["file"])
            lo, hi = R.COUNT_LEVELS[L]
            space = [str(v) for v in range(lo, hi + 2)]
            iid = f"counting_{made:06d}"
            im = Image.open(src(s["file"])).convert("RGB")
            cfim = mask = None
            if made < a.n_cf:
                pool = [d for d in donors[s["cid"]] if d[0] != s["image_id"]]
                for _ in range(12):
                    if not pool: break
                    di, dan = pool[rng.randrange(len(pool))]
                    dim = Image.open(src(imgs[di]["file_name"])).convert("RGB")
                    cfim, mask = R.paste_instance(im, dim, dan, rng)
                    if cfim is not None: break
                if cfim is not None and str(s["n"] + 1) not in space: cfim = None
            p0 = f"images/{iid}.png"; im.save(os.path.join(a.out, p0))
            q = f"How many {s['cat']} are in the image? Answer with a number."
            man.append(rec(iid, "counting", p0, q, s["answer"], space,
                           dict(count=s["n"]), L, "paste_one",
                           referents=[s["cat"]], extra=dict(n=s["n"]),
                           coco=dict(image_id=s["image_id"], cat_id=s["cid"])))
            if cfim is not None:
                p1 = f"images/{iid}_cf.png"; cfim.save(os.path.join(a.out, p1))
                mask.save(os.path.join(a.out, f"images/{iid}_cf_mask.png"))
                man.append(rec(f"{iid}_cf", "counting", p1, q, s["n"] + 1, space,
                               dict(count=s["n"] + 1), L, "paste_one", cf_of=iid,
                               referents=[s["cat"]], extra=dict(n=s["n"] + 1)))
            made += 1
        stats["counting"] = made

    if "glyph" in fams:
        L = lv.get("glyph", 0); made = 0
        pool = [i["file_name"] for i in imgs.values()]
        rng.shuffle(pool)
        for f in pool:
            if made >= a.n: break
            if f in used: continue
            im = Image.open(src(f)).convert("RGB")
            W, H = im.size
            if W < 200 or H < 200: continue
            bw = bh = 40
            bx, by = rng.randrange(10, W - bw - 10), rng.randrange(10, H - bh - 10)
            box = (bx, by, bx + bw, by + bh)
            used.add(f)
            v = rng.choice(GLYPH_VALS)
            iid = f"glyph_{made:06d}"
            p0 = f"images/{iid}.png"
            R.draw_glyph(im, box, v, L).save(os.path.join(a.out, p0))
            q = ("What number is printed in the white box? The number is a multiple of 5 "
                 "between 10 and 95. Answer with the number.")
            man.append(rec(iid, "glyph", p0, q, v, GLYPH_VALS, dict(value=int(v)), L,
                           "redraw_value", extra=dict(px=R.GLYPH_LEVELS[L], box=list(box))))
            if made < a.n_cf:
                v2 = rng.choice([x for x in GLYPH_VALS if x != v])
                p1 = f"images/{iid}_cf.png"
                R.draw_glyph(im, box, v2, L).save(os.path.join(a.out, p1))
                man.append(rec(f"{iid}_cf", "glyph", p1, q, v2, GLYPH_VALS,
                               dict(value=int(v2)), L, "redraw_value", cf_of=iid,
                               extra=dict(px=R.GLYPH_LEVELS[L], box=list(box))))
            made += 1
        stats["glyph"] = made

    with open(os.path.join(a.out, "manifest.jsonl"), "w") as fh:
        for r in man: fh.write(json.dumps(r) + "\n")
    n_cf = sum(1 for r in man if r["cf_of"])
    print(f"wrote {a.out}/manifest.jsonl   {len(man)} records "
          f"({len(man)-n_cf} items + {n_cf} counterfactuals)")
    for f, n in stats.items(): print(f"  {f:10s} {n}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco", default="data/real/coco")
    p.add_argument("--out", default="data/real_v1")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--n-cf", type=int, default=100)
    p.add_argument("--families", default="spatial,counting,glyph")
    p.add_argument("--levels", default="", help="e.g. spatial=1,counting=0,glyph=2")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
