# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Verify the label-preserving edits for spatial, counting and glyph.

The guards are inverted, as gen/verify_sham.py's are for the chart family: elsewhere a
counterfactual must change the answer and the check is that it does; here it must NOT, and an
edit that did nothing at all would pass that trivially. So "the edit happened" is checked as
carefully as "the answer held", and each family's exactness is checked in its own terms:

  spatial   flipping the sham vertically must return the original bit for bit. That is the whole
            claim -- the sham IS the reflection, so nothing about it is approximate. A pixel
            count is reported beside it but not gated: both edits reflect the frame and displace
            every off-axis pixel, and how many of them land on a different colour is a fact about
            the photograph's symmetry rather than about the size of the edit.
  counting  pixels outside the pasted silhouette are bit-identical; the pasted category is
            neither the queried one nor in its supercategory; and the silhouette touches no
            annotated instance of the queried category, since occluding one would change the
            count. The pixel count must be at least the real paste's.
  glyph     pixels outside the white box are bit-identical, and the ink inside it is an EXACT
            translation of the original's by the recorded shift -- the numeral moved, it was not
            re-rendered. The pixel count must be at least the real redraw's.

One measurement is reported rather than enforced: the real counting counterfactual pastes its
new instance at a random position with no such occlusion guard, so it can cover an existing one.
That is a property of the published dataset, not of this control, and it is printed here because
this is the first place the annotation needed to answer it was loaded.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real as R

FAIL = 0


def chk(cond, msg):
    global FAIL
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAIL += 1


def px(root, p):
    return np.asarray(Image.open(os.path.join(root, p)).convert("RGB"))


def poly(size, seg):
    """A COCO polygon rasterised to a boolean mask, as gen/real.py's paste_instance does."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for q in seg:
        if len(q) >= 6:
            d.polygon([tuple(q[k:k + 2]) for k in range(0, len(q) - 1, 2)], fill=255)
    return np.asarray(m) > 127


def silhouette(root, r):
    """The pixels allowed to differ. 255 marks them, as every canonical dataset writes."""
    return np.asarray(Image.open(os.path.join(root, r["cf_mask"])).convert("L")) == 255


def main(a):
    recs = [json.loads(l) for l in open(os.path.join(a.root, "manifest.jsonl"))]
    by = {r["id"]: r for r in recs}
    sham = [r for r in recs if r["cf_of"]]
    fams = sorted({r["family"] for r in sham})
    stats = {}
    srcbase = [x for x in (json.loads(l) for l in open(os.path.join(a.src, "manifest.jsonl")))
               if x["cf_of"] is None]
    per = {f: sum(1 for r in sham if r["family"] == f) for f in fams}
    counts = ", ".join("%s %d" % (f, per[f]) for f in fams)
    print(f"{len(recs)} records: {len(recs)-len(sham)} originals, {len(sham)} sham edits "
          f"({counts})\n")

    print("[1] the sham preserves the label: answer, question and attribute all unchanged")
    bad = [r["id"] for r in sham
           if (r["answer"], r["question"], r["attribute"])
           != (by[r["cf_of"]]["answer"], by[r["cf_of"]]["question"], by[r["cf_of"]]["attribute"])]
    chk(not bad, f"{len(sham)-len(bad)}/{len(sham)} identical" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[2] the edit is real, and the recorded pixel count is the measured one")
    bad, zero = [], []
    for r in sham:
        A, B = px(a.root, by[r["cf_of"]]["image"]), px(a.root, r["image"])
        n = int((A != B).any(2).sum())
        if n == 0:
            zero.append(r["id"])
        if n != r["difficulty"]["sham_px"]:
            bad.append((r["id"], n, r["difficulty"]["sham_px"]))
    chk(not zero and not bad,
        f"{len(sham)-len(bad)-len(zero)}/{len(sham)} changed and recorded exactly"
        + (f"  unchanged {zero[:3]}" if zero else "") + (f"  mismatch {bad[:3]}" if bad else ""))

    print("\n[3] the sham is not the smaller edit")
    for f in fams:
        s = np.array([r["difficulty"]["sham_px"] for r in sham if r["family"] == f], float)
        t = np.array([r["difficulty"]["real_px"] for r in sham if r["family"] == f], float)
        stats.setdefault("pixels", {})[f] = dict(
            n=len(s), lo=float((s / t).min()), hi=float((s / t).max()),
            median=float(np.median(s / t)), ge=int((s >= t).sum()),
            # absolute counts too: the ratio says the sham is the larger edit, the count says
            # how large, and the manuscript quotes the count for the glyph family
            px_lo=int(s.min()), px_hi=int(s.max()), px_median=int(np.median(s)),
            real_px_lo=int(t.min()), real_px_hi=int(t.max()),
            # how many base items of this family there were to build from, so the drop rate is
            # an artefact rather than a line of the builder's terminal output
            base=sum(1 for x in srcbase if x["family"] == f))
        if f == "spatial":
            # not gated: see the module docstring. Reported so the claim is a measurement.
            print(f"  --    spatial reflects the frame either way; sham/real pixels "
                  f"{(s/t).min():.3f} to {(s/t).max():.3f}, median {np.median(s/t):.3f}")
        else:
            chk(bool((s >= t).all()),
                f"{f}: {int((s >= t).sum())}/{len(s)} at least the real edit, "
                f"ratio {(s/t).min():.3f} to {(s/t).max():.3f}")

    if "spatial" in fams:
        print("\n[4] spatial: flipping the sham vertically returns the original, bit for bit")
        bad = []
        for r in (x for x in sham if x["family"] == "spatial"):
            A, B = px(a.root, by[r["cf_of"]]["image"]), px(a.root, r["image"])
            if A.shape != B.shape or not np.array_equal(A, B[::-1, :]):
                bad.append(r["id"])
        n = sum(1 for x in sham if x["family"] == "spatial")
        chk(not bad, f"{n-len(bad)}/{n} exact" + (f"  e.g. {bad[:3]}" if bad else ""))

    if "counting" in fams:
        ct = [x for x in sham if x["family"] == "counting"]
        print("\n[5] counting: pixels outside the pasted silhouette are bit-identical, and the "
              "silhouette is a proper subset")
        bad, whole = [], []
        for r in ct:
            A, B = px(a.root, by[r["cf_of"]]["image"]), px(a.root, r["image"])
            m = silhouette(a.root, r)
            if m.all():
                whole.append(r["id"])
            if A.shape != B.shape or not np.array_equal(A[~m], B[~m]):
                bad.append(r["id"])
        chk(not bad and not whole, f"{len(ct)-len(bad)-len(whole)}/{len(ct)} exact"
            + (f"  leaked {bad[:3]}" if bad else "") + (f"  whole-frame mask {whole[:3]}" if whole else ""))

        print("\n[6] counting: the pasted category cannot be the answer -- not the queried "
              "category, and not its supercategory")
        raw = json.load(open(os.path.join(a.coco, "annotations/instances_val2017.json")))
        sup = {c["id"]: c["supercategory"] for c in raw["categories"]}
        bad = [(r["id"], r["difficulty"]["sham_cat"], r["difficulty"]["queried_cat"])
               for r in ct
               if r["difficulty"]["sham_cat_id"] == r["difficulty"]["queried_cat_id"]
               or sup.get(r["difficulty"]["sham_cat_id"]) == sup.get(r["difficulty"]["queried_cat_id"])]
        chk(not bad, f"{len(ct)-len(bad)}/{len(ct)} from an unrelated supercategory"
            + (f"  e.g. {bad[:3]}" if bad else ""))

        print("\n[7] counting: the paste occludes no annotated instance of the queried category")
        _, imgs, per = R.load_coco(a.coco)
        bad = []
        for r in ct:
            m = silhouette(a.root, r)
            cid = r["difficulty"]["queried_cat_id"]
            for an in per.get(by[r["cf_of"]]["coco"]["image_id"], []):
                if an["category_id"] != cid:
                    continue
                x, y, w, h = [int(v) for v in an["bbox"]]
                if m[max(0, y):y + h + 1, max(0, x):x + w + 1].any():
                    bad.append(r["id"])
                    break
        chk(not bad, f"{len(ct)-len(bad)}/{len(ct)} clear of every queried instance"
            + (f"  e.g. {bad[:3]}" if bad else ""))

        # Reported, not gated, and measured against the instance rather than its bounding box.
        # The guard above is deliberately conservative for the sham -- a bbox is a superset of
        # the object -- but the property that would actually change a count is an instance being
        # covered enough to stop being countable, so the published counterfactual is measured on
        # the covered FRACTION of each queried instance's own segmentation. It has no such guard:
        # gen/real.py's paste_instance picks a random position and nothing looks at what is
        # already there.
        src = [json.loads(l) for l in open(os.path.join(a.src, "manifest.jsonl"))]
        sby = {x["id"]: x for x in src}
        cov = []
        for x in src:
            if x["family"] != "counting" or not x["cf_of"]:
                continue
            b = sby[x["cf_of"]]
            # gen/build_real.py saves the mask but does not name it in the record, so the path
            # comes from the image name, exactly as gen/verify_real.py derives it
            mp = os.path.join(a.src, x.get("cf_mask") or x["image"].replace(".png", "_mask.png"))
            if not os.path.exists(mp):
                continue
            m = np.asarray(Image.open(mp).convert("L")) == 255
            cid = b["coco"]["cat_id"]
            size = (m.shape[1], m.shape[0])
            worst = 0.0
            for an in per.get(b["coco"]["image_id"], []):
                if an["category_id"] != cid or not isinstance(an.get("segmentation"), list):
                    continue
                sg = poly(size, an["segmentation"])
                if sg.sum():
                    worst = max(worst, float((sg & m).sum()) / float(sg.sum()))
            cov.append((worst, b["id"]))
        cv = np.array([c for c, _ in cov])
        occl = dict(n=len(cov), full=int((cv >= 0.999).sum()), ge90=int((cv >= 0.90).sum()),
                    ge50=int((cv >= 0.50).sum()), median=float(np.median(cv)) if len(cv) else 0.0,
                    # the ids, so run/canon.py can say what they cost without loading COCO
                    full_ids=sorted(i for c, i in cov if c >= 0.999))
        stats["counting_cf_occlusion"] = occl
        print(f"  --    the REAL counting edit has no such guard. Of {occl['n']} published "
              f"counterfactuals, the paste covers an existing instance of the queried category "
              f"completely in {occl['full']}, at least 90% in {occl['ge90']}, at least 50% in "
              f"{occl['ge50']}; median coverage {100*occl['median']:.1f}%. The {occl['full']} "
              f"fully covered ones do not raise the count, so their recorded answer is wrong.")

    if "glyph" in fams:
        gl = [x for x in sham if x["family"] == "glyph"]
        print("\n[8] glyph: pixels outside the white box are bit-identical")
        bad = []
        for r in gl:
            A, B = px(a.root, by[r["cf_of"]]["image"]), px(a.root, r["image"])
            x0, y0, x1, y1 = r["difficulty"]["sham_box"]
            C, D = A.copy(), B.copy()
            C[y0:y1, x0:x1] = D[y0:y1, x0:x1] = 0
            if not np.array_equal(C, D):
                bad.append(r["id"])
        chk(not bad, f"{len(gl)-len(bad)}/{len(gl)} exact" + (f"  e.g. {bad[:3]}" if bad else ""))

        print("\n[9] glyph: the ink is an exact translation of the original's, by the recorded "
              "shift -- the numeral moved, it was not redrawn")
        bad = []
        for r in gl:
            A, B = px(a.root, by[r["cf_of"]]["image"]), px(a.root, r["image"])
            x0, y0, x1, y1 = r["difficulty"]["sham_box"]
            dx, dy = r["difficulty"]["sham_shift"]
            oa, ob = A[y0:y1, x0:x1], B[y0:y1, x0:x1]
            ia, ib = (oa != 255).any(2), (ob != 255).any(2)
            ys, xs = np.nonzero(ia)
            want = np.zeros_like(ia)
            want[ys + dy, xs + dx] = True
            if not np.array_equal(ib, want) or not np.array_equal(ob[ys + dy, xs + dx], oa[ys, xs]):
                bad.append(r["id"])
        chk(not bad, f"{len(gl)-len(bad)}/{len(gl)} exact translations"
            + (f"  e.g. {bad[:3]}" if bad else ""))

        print("\n[10] glyph: the numeral lands clear of where it was, so the edit is a move "
              "rather than a nudge")
        bad = []
        for r in gl:
            A = px(a.root, by[r["cf_of"]]["image"])
            x0, y0, x1, y1 = r["difficulty"]["sham_box"]
            dx, dy = r["difficulty"]["sham_shift"]
            ia = (A[y0:y1, x0:x1] != 255).any(2)
            ys, xs = np.nonzero(ia)
            moved = np.zeros_like(ia)
            moved[ys + dy, xs + dx] = True
            if (moved & ia).any():
                bad.append(r["id"])
        chk(not bad, f"{len(gl)-len(bad)}/{len(gl)} disjoint from the original ink"
            + (f"  e.g. {bad[:3]}" if bad else ""))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(stats, open(a.out, "w"), indent=1)
        print(f"\n  wrote {a.out}")
    print(f"\n{'PASS' if not FAIL else str(FAIL) + ' CHECK(S) FAILED'}")
    return FAIL


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/real_3b_sham")
    p.add_argument("--src", default="data/real_3b")
    p.add_argument("--coco", default="data/real/coco")
    # the measurements this makes are quoted in the paper, so they leave as an artefact rather
    # than as terminal output. run/canon.py reads it; COCO itself stays off canon's load path.
    p.add_argument("--out", default="runs/sham_guards.json")
    sys.exit(main(p.parse_args()))
