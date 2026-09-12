# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Label-preserving (sham) edits for the three real-image families.

gen/build_chart_sham.py built the chart family's sham: raise a bar the question does not ask
about. This is the same control for spatial, counting and glyph, and the design rule is the one
that family established -- apply the SAME primitive as the real counterfactual, to something the
question does not ask about, at a size that is not smaller than the real edit:

  spatial   real: a horizontal flip, which inverts left/right.
            sham: a VERTICAL flip. The same operation about the axis the question does not ask
            about. x-coordinates are untouched, so the relation is preserved exactly, and the
            round trip is checkable: flipping the sham vertically must return the original bit
            for bit. Magnitude needs no filter here -- both reflect the whole frame, and the
            sham changes 0.95 to 1.11 times as many pixels as the real edit (median 1.00),
            measured per item and recorded.
            The confound this one carries is named rather than hidden: an upside-down photograph
            is out of distribution, so a model that moves on it may have failed to parse the
            scene rather than read a correlate. That is why the model's own stability is
            reported beside the probe's -- it is the measurement that separates the two.

  counting  real: paste one more instance of the QUERIED category, so the count rises by one.
            sham: paste one instance of a category that is neither the queried one nor in its
            COCO supercategory, so the count cannot rise -- a truck pasted into "how many car"
            is not a label-preserving edit, and supercategory is the annotation that says so.
            The silhouette may not touch any annotated instance of the queried category, since
            occluding one would lower the count. Chosen to change at least as many pixels as
            that item's real paste, and as few more as the donor pool allows.

  glyph     real: redraw the box with a different numeral, which at 3 px is a 3 to 8 pixel edit.
            sham: MOVE the same numeral inside the same white box, clear of where it was. The
            ink is copied, not re-rendered, so the sham's ink is a pure translation of the
            original's -- exactly checkable, and immune to any font-rendering difference. Landing
            clear of the original makes the edit about twice the ink, so it is the larger change.

Every sham record carries `sham_px` and `real_px`, the measured pixel counts of the two edits, so
"the sham is not the smaller change" is a property of the data rather than a claim in a docstring.
gen/verify_real_sham.py re-derives all of it from the written PNGs.

Masks follow the one convention the canonical datasets use: **255 marks the pixels allowed to
differ**, as gen/real.py's paste_instance and gen/chart_real.py's raise_bar both write and
gen/verify_real.py reads. gen/build_chart_sham.py wrote the inverse for two days under a comment
claiming it matched them, which is why run/verify_protocol.py now measures the polarity from the
pixels in every dataset that has a mask rather than trusting any file's comment.
"""
from __future__ import annotations
import argparse, json, os, random, shutil, sys, zlib
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real as R

WHITE = 255


def changed(A, B):
    """Pixels that differ between two same-shape RGB arrays."""
    return int((A != B).any(2).sum())


_DONOR = {}


def donor_image(root, fn):
    """A small cache: COCO val2017 is 1.8 GB and a donor is reused across items."""
    if fn not in _DONOR:
        if len(_DONOR) > 256:
            _DONOR.clear()
        _DONOR[fn] = Image.open(os.path.join(root, "val2017", fn)).convert("RGB")
    return _DONOR[fn]


def poly_mask(size, seg):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for poly in seg:
        if len(poly) >= 6:
            d.polygon([tuple(poly[k:k + 2]) for k in range(0, len(poly) - 1, 2)], fill=255)
    return m


# ---------------------------------------------------------------- spatial


def sham_spatial(im):
    """The reflection about the other axis. No choice to make, so nothing to tune."""
    out = im.transpose(Image.FLIP_TOP_BOTTOM)
    return out, None, changed(np.asarray(im), np.asarray(out)), dict(sham_kind="vflip")


# ---------------------------------------------------------------- counting


def sham_counting(im, r, want_px, coco, rng, drop):
    """Paste one instance of a category the question cannot be asking about.

    Two exclusions, both from the annotations rather than from judgement: the queried category
    itself, and everything sharing its supercategory. Plus one geometric guard -- the silhouette
    may not touch an annotated instance of the queried category, because occluding one would
    change the count the sham is supposed to preserve.
    """
    cats, imgs, per, donors, supers, root = coco
    cid = r["coco"]["cat_id"]
    bad_super = supers.get(cid)
    forbidden = np.zeros(im.size[::-1], bool)
    for an in per.get(r["coco"]["image_id"], []):
        if an["category_id"] != cid:
            continue
        x, y, w, h = [int(v) for v in an["bbox"]]
        forbidden[max(0, y):y + h + 1, max(0, x):x + w + 1] = True

    W, H = im.size
    A = np.asarray(im)
    cands = []
    # the silhouette is a subset of the bounding box, so a donor whose box is smaller than the
    # real edit cannot possibly match it. Filtering on that before opening any file turns a
    # 1.8 GB scan per item into a few dozen reads.
    pool = []
    for dc in donors:
        if dc == cid or supers.get(dc) == bad_super:
            continue
        for di, dan in donors[dc]:
            x, y, w, h = [int(v) for v in dan["bbox"]]
            if w < 12 or h < 12 or w >= W or h >= H or w * h < want_px:
                continue
            if not isinstance(dan.get("segmentation"), list) or not dan["segmentation"]:
                continue
            pool.append((dc, di, dan))
    rng.shuffle(pool)
    for dc, di, dan in pool[:40]:
        x, y, w, h = [int(v) for v in dan["bbox"]]
        seg = dan["segmentation"]
        dim = donor_image(root, imgs[di]["file_name"])
        cmask = poly_mask(dim.size, seg).crop((x, y, x + w, y + h))
        if np.asarray(cmask).mean() < 20:
            continue
        crop = dim.crop((x, y, x + w, y + h))
        placed = False
        for _ in range(24):                      # deterministic: rng is seeded from the item id
            px, py = rng.randrange(0, W - w), rng.randrange(0, H - h)
            sil = np.asarray(cmask) > 127
            if forbidden[py:py + h, px:px + w][sil].any():
                continue
            placed = True
            break
        if not placed:
            continue
        out = im.copy()
        out.paste(crop, (px, py), cmask)
        full = Image.new("L", im.size, 0)
        full.paste(cmask, (px, py))
        n = changed(A, np.asarray(out))
        if n < want_px:
            continue
        cands.append((n, dc, di, px, py, out, full))
        if len(cands) >= 6:
            break
    if not cands:
        drop["counting_no_donor"] += 1
        return None, None, 0, None
    n, dc, di, px, py, out, full = min(cands, key=lambda c: c[0])
    return out, full, n, dict(sham_kind="paste_other", sham_cat=cats[dc], sham_cat_id=dc,
                              sham_donor_image=di, sham_at=[px, py],
                              queried_cat=cats[cid], queried_cat_id=cid)


# ---------------------------------------------------------------- glyph


def sham_glyph(im, r, want_px, drop):
    """Move the numeral inside its box, clear of where it was, by copying ink rather than
    re-rendering it. The sham's ink is then a translation of the original's, exactly."""
    x0, y0, x1, y1 = r["difficulty"]["box"]
    A = np.asarray(im)
    box = A[y0:y1, x0:x1]
    ink = (box != WHITE).any(2)
    if not ink.any():
        drop["glyph_no_ink"] += 1
        return None, None, 0, None
    ys, xs = np.nonzero(ink)
    h, w = y1 - y0, x1 - x0
    best = None
    for dy in range(-h, h + 1):
        for dx in range(-w, w + 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = ys + dy, xs + dx
            if ny.min() < 0 or ny.max() >= h or nx.min() < 0 or nx.max() >= w:
                continue
            moved = np.zeros_like(ink)
            moved[ny, nx] = True
            if (moved & ink).any():              # must land clear, not smear
                continue
            new = np.full_like(box, WHITE)
            new[ny, nx] = box[ys, xs]
            n = changed(box, new)
            if n < want_px:
                continue
            key = (n, abs(dx) + abs(dy), dy, dx)
            if best is None or key < best[0]:
                best = (key, int(dx), int(dy), new, moved | ink)
    if best is None:
        drop["glyph_no_room"] += 1
        return None, None, 0, None
    n, sdx, sdy, new, touched = best[0][0], best[1], best[2], best[3], best[4]
    out = A.copy()
    out[y0:y1, x0:x1] = new
    mask = np.zeros(A.shape[:2], bool)
    mask[y0:y1, x0:x1] = touched
    return (Image.fromarray(out), Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)),
            n, dict(sham_kind="move_glyph", sham_shift=[sdx, sdy],
                    sham_box=[x0, y0, x1, y1]))


# ---------------------------------------------------------------- driver


def main(a):
    src = [json.loads(l) for l in open(os.path.join(a.src, "manifest.jsonl"))]
    by = {r["id"]: r for r in src}
    base = [r for r in src if r["cf_of"] is None]
    cf_of = {r["cf_of"]: r for r in src if r["cf_of"]}
    fams = a.families.split(",")
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    coco = None
    if "counting" in fams:
        cats, imgs, per = R.load_coco(a.coco)
        raw = json.load(open(os.path.join(a.coco, "annotations/instances_val2017.json")))
        supers = {c["id"]: c["supercategory"] for c in raw["categories"]}
        donors = {}
        for iid_, anns in per.items():
            for an in anns:
                if isinstance(an.get("segmentation"), list) and an["bbox"][2] >= 12 \
                        and an["bbox"][3] >= 12:
                    donors.setdefault(an["category_id"], []).append((iid_, an))
        coco = (cats, imgs, per, donors, supers, a.coco)

    man, drop, px, per = [], {}, {}, {}
    for k in ("no_cf", "counting_no_donor", "glyph_no_ink", "glyph_no_room"):
        drop[k] = 0
    for r in base:
        if r["family"] not in fams:
            continue
        per[r["family"]] = per.get(r["family"], 0) + 1
        if a.limit and per[r["family"]] > a.limit:
            continue
        c = cf_of.get(r["id"])
        if c is None:
            drop["no_cf"] += 1
            continue
        im = Image.open(os.path.join(a.src, r["image"])).convert("RGB")
        cfim = Image.open(os.path.join(a.src, c["image"])).convert("RGB")
        real_px = changed(np.asarray(im), np.asarray(cfim))
        rng = random.Random(a.seed ^ zlib.crc32(r["id"].encode()))

        if r["family"] == "spatial":
            out, mask, n, extra = sham_spatial(im)
        elif r["family"] == "counting":
            out, mask, n, extra = sham_counting(im, r, real_px, coco, rng, drop)
        elif r["family"] == "glyph":
            out, mask, n, extra = sham_glyph(im, r, real_px, drop)
        else:
            continue
        if out is None:
            continue

        sid = f"{r['id']}_sham"
        out.save(os.path.join(a.out, "images", f"{sid}.png"))
        if mask is not None:
            mask.save(os.path.join(a.out, "images", f"{sid}_mask.png"))
        shutil.copyfile(os.path.join(a.src, r["image"]),
                        os.path.join(a.out, "images", os.path.basename(r["image"])))
        man.append(dict(r))
        man.append(dict(r, id=sid, cf_of=r["id"], cf_kind=f"sham_{extra['sham_kind']}",
                        image=f"images/{sid}.png",
                        cf_mask=(f"images/{sid}_mask.png" if mask is not None else None),
                        # answer, question and attribute are the original's: that is the sham
                        difficulty=dict(r["difficulty"], sham_px=n, real_px=real_px, **extra)))
        px.setdefault(r["family"], []).append((n, real_px))

    with open(os.path.join(a.out, "manifest.jsonl"), "w") as fh:
        for r in man:
            fh.write(json.dumps(r) + "\n")
    n_sham = sum(1 for r in man if r["cf_of"])
    print(f"wrote {a.out}: {n_sham} sham pairs from {len(base)} base items")
    if any(drop.values()):
        print("  dropped: " + ", ".join(f"{k} {v}" for k, v in drop.items() if v))
    for f in sorted(px):
        s = np.array([p[0] for p in px[f]], float)
        t = np.array([p[1] for p in px[f]], float)
        print(f"  {f:9s} n={len(s):3d}  sham {s.min():8.0f} to {s.max():8.0f} px, "
              f"real {t.min():8.0f} to {t.max():8.0f};  sham/real "
              f"min {(s/t).min():.3f} median {np.median(s/t):.3f} max {(s/t).max():.3f}; "
              f"sham >= real on {(s >= t).sum()}/{len(s)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/real_3b")
    p.add_argument("--coco", default="data/real/coco")
    p.add_argument("--out", default="data/real_3b_sham")
    p.add_argument("--families", default="spatial,counting,glyph")
    p.add_argument("--limit", type=int, default=0, help="first N base items per family; for smoke tests")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
