# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Real-image families with exact counterfactuals.

The synthetic study's credibility rests on a counterfactual control -- edit one field of the
spec, re-render from the same seed, and ask whether the probe follows. Real photographs have no
spec and no seed, and a diffusion inpainter is not exact, so the control has to be rebuilt out of
edits that are exact by construction:

  spatial   COCO photos, left/right read off annotated box centres. The counterfactual is a
            HORIZONTAL FLIP, which inverts the label deterministically and leaves every pixel a
            real photographic pixel -- no compositing, no generative model, no artefacts.
  counting  COCO counts from the annotations. The counterfactual pastes one more instance of the
            queried category through its own segmentation mask, so every pixel outside the pasted
            silhouette is bit-identical and the count provably changes by one.
  glyph     the designed negative control: a numeral rendered on a real photo at a size swept
            across the vision encoder's patch scale. Only the glyph box changes. At small sizes
            the information is not in the image, so a probe that reads it is reading something
            else -- the real-image analogue of synthetic `tracking`.

One deliberate departure from the synthetic protocol: **no prior arms**. In the synthetic set the
canonical/anti arms turned out to have disjoint answer sets by construction (defect (c)), which
made pooled chance wrong and forced a within-arm refit. Re-introducing that here would repeat a
known mistake for no gain, so every real family is single-arm and prior-answerability is measured
by the blindfold contrast instead, which is what capture.py already records.
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# horizontal separation between box centres as a fraction of image width; smaller is harder. The
# level sweep picks the band whose error rate sits nearest 50% for each model, as the synthetic
# families did.
SPATIAL_LEVELS = [(0.40, 1.00), (0.25, 0.40), (0.12, 0.25), (0.04, 0.12)]
COUNT_LEVELS   = [(1, 3), (2, 5), (3, 8), (4, 12)]      # inclusive count range per level
# font px. The ViT patch is 14 px, but Qwen2.5-VL uses dynamic resolution and still reads a 7 px
# numeral at 68% against 5.6% chance, so "below the patch scale" is not the same as "not in the
# image". The levels run down to 2 px to cross the point where the information really is gone;
# the sweep across them is a dose-response curve, which is a stronger control than any single
# unreadable level -- the probe must fall with the model, not stay up while the model falls.
GLYPH_LEVELS   = [28, 18, 11, 7, 5, 4, 3, 2]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_coco(root):
    d = json.load(open(os.path.join(root, "annotations/instances_val2017.json")))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd") and a["area"] >= 2000:
            per[a["image_id"]].append(a)
    return cats, imgs, per


def uniq_cats(anns):
    """Categories appearing exactly once -- a question naming one of them is unambiguous, which
    is the real-image version of the synthetic set's referent-uniqueness check [6b]."""
    n = defaultdict(int)
    for a in anns: n[a["category_id"]] += 1
    return {c for c, k in n.items() if k == 1}


def gen_spatial(cats, imgs, per, level, want, rng):
    lo, hi = SPATIAL_LEVELS[level]
    out = []
    for iid in sorted(per):
        if len(out) >= want: break
        anns = per[iid]; u = uniq_cats(anns)
        cand = [a for a in anns if a["category_id"] in u]
        if len(cand) < 2: continue
        W = imgs[iid]["width"]
        for i in range(len(cand)):
            hit = False
            for j in range(i + 1, len(cand)):
                A, Bn = cand[i], cand[j]
                xa = A["bbox"][0] + A["bbox"][2] / 2
                xb = Bn["bbox"][0] + Bn["bbox"][2] / 2
                if not (lo <= abs(xa - xb) / W < hi): continue
                if rng.random() < 0.5:                       # randomise which object is subject
                    A, Bn, xa, xb = Bn, A, xb, xa
                out.append(dict(file=imgs[iid]["file_name"], image_id=iid,
                                ca=cats[A["category_id"]], cb=cats[Bn["category_id"]],
                                answer="left" if xa < xb else "right",
                                sep=round(abs(xa - xb) / W, 4)))
                hit = True; break
            if hit: break
    return out[:want]


def gen_counting(cats, imgs, per, level, want, rng):
    """Balanced across the count range. COCO's natural distribution is steeply skewed -- at
    level 1 a bare 'first match' scan gives 52% of items the count 2, which is above the answer
    space's 20% chance and makes 'always say 2' look like a result. Stratifying removes that."""
    lo, hi = COUNT_LEVELS[level]
    pool = defaultdict(list)
    for iid in sorted(per):
        grp = defaultdict(list)
        for a in per[iid]: grp[a["category_id"]].append(a)
        for cid in sorted(grp):
            k = len(grp[cid])
            if lo <= k <= hi:
                pool[k].append(dict(file=imgs[iid]["file_name"], image_id=iid,
                                    cat=cats[cid], cid=cid, answer=str(k), n=k))
                break
    for k in pool: rng.shuffle(pool[k])
    out, ks = [], sorted(pool)
    while len(out) < want and any(pool[k] for k in ks):
        for k in ks:
            if pool[k] and len(out) < want: out.append(pool[k].pop())
    return out[:want]


def paste_instance(im, donor, ann, rng):
    """Paste one instance through its COCO polygon mask. Returns (image, mask) where the mask
    marks exactly the pixels allowed to differ; everything else stays bit-identical."""
    seg = ann.get("segmentation")
    if not isinstance(seg, list) or not seg: return None, None
    x, y, w, h = [int(v) for v in ann["bbox"]]
    if w < 12 or h < 12: return None, None
    m = Image.new("L", donor.size, 0); dr = ImageDraw.Draw(m)
    for poly in seg:
        if len(poly) >= 6:
            dr.polygon([tuple(poly[k:k + 2]) for k in range(0, len(poly) - 1, 2)], fill=255)
    crop, cmask = donor.crop((x, y, x + w, y + h)), m.crop((x, y, x + w, y + h))
    if np.asarray(cmask).mean() < 20: return None, None      # mask too sparse to read as an object
    W, H = im.size
    if w >= W or h >= H: return None, None
    px, py = rng.randrange(0, W - w), rng.randrange(0, H - h)
    out = im.copy(); out.paste(crop, (px, py), cmask)
    full = Image.new("L", im.size, 0); full.paste(cmask, (px, py))
    return out, full


def draw_glyph(im, box, value, level):
    """Stamp a numeral into a fixed white box. Only the box may differ between an item and its
    counterfactual, which is what makes the edit exact."""
    im = im.copy(); d = ImageDraw.Draw(im)
    d.rectangle(box, fill=(255, 255, 255))
    try: font = ImageFont.truetype(FONT, GLYPH_LEVELS[level])
    except OSError: font = ImageFont.load_default()
    d.text((box[0] + 2, box[1] + 1), str(value), fill=(0, 0, 0), font=font)
    return im
