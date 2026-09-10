# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Verify the real-image datasets. The counterfactual exactness checks are the point: on
synthetic data the guard was "re-render from the same seed and compare"; here each family needs
its own guard, and every one of them is a pixel identity that either holds or does not.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import defaultdict
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real as R

FAIL = 0


def check(cond, msg):
    global FAIL
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond: FAIL += 1


def px(root, r):
    return np.asarray(Image.open(os.path.join(root, r["image"])).convert("RGB"))


def main(a):
    recs = [json.loads(l) for l in open(os.path.join(a.root, "manifest.jsonl"))]
    by = {r["id"]: r for r in recs}
    base = [r for r in recs if r["cf_of"] is None]
    cfs = [r for r in recs if r["cf_of"]]
    print(f"{len(recs)} records: {len(base)} items, {len(cfs)} counterfactuals\n")

    print("[1] answer lies in the answer space and chance matches its size")
    bad = [r["id"] for r in recs if r["answer"] not in r["answer_space"]
           or abs(r["chance"] - 1 / len(r["answer_space"])) > 1e-9]
    check(not bad, f"{len(recs)-len(bad)}/{len(recs)} consistent" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[2] labels re-derived from the COCO annotations, independently of the generator")
    cats, imgs, per = R.load_coco(a.coco)
    name2id = {v: k for k, v in cats.items()}
    fn2id = {i["file_name"]: i["id"] for i in imgs.values()}
    import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import chart_real as _C
    bad, n, geom = [], 0, 0
    for r in base:
        c = r.get("coco") or {}
        if r["family"] == "chart" and c.get("source"):
            n += 1
            # A painted bar's value is not in the source table -- the table describes the chart
            # as published, and the generator changed it. Those items are declared in the
            # manifest and verified against the rendered geometry instead, which is the check
            # that can see them; between the two every item is re-derived from something other
            # than the label itself. Neither branch is allowed to fall back to the other.
            if r.get("synthetic_bar"):
                geom += 1
                continue
            sp, stem = c["source"].split("/")[1:]
            cats, vals = _C.read_table(f"data/real/chartqa/ChartQA Dataset/{sp}/tables/{stem}.csv")
            if cats is None or r["referents"][0] not in cats:
                bad.append((r["id"], "category not in table")); continue
            v = vals[cats.index(r["referents"][0])]
            if str(int(round(v / 5) * 5)) != r["answer"]:
                bad.append((r["id"], f"table {v} -> {int(round(v/5)*5)} != {r['answer']}"))
            continue
        if not c.get("image_id"): continue
        anns = per[c["image_id"]]; n += 1
        if r["family"] == "spatial":
            got = {}
            for cid in (name2id[c["subj"]], name2id[c["obj"]]):
                m = [an for an in anns if an["category_id"] == cid]
                if len(m) != 1: bad.append((r["id"], "referent not unique")); break
                got[cid] = m[0]["bbox"][0] + m[0]["bbox"][2] / 2
            else:
                want = "left" if got[name2id[c["subj"]]] < got[name2id[c["obj"]]] else "right"
                if want != r["answer"]: bad.append((r["id"], f"{want} != {r['answer']}"))
        elif r["family"] == "counting":
            k = sum(an["category_id"] == c["cat_id"] for an in anns)
            if str(k) != r["answer"]: bad.append((r["id"], f"{k} != {r['answer']}"))
    check(not bad, f"{n-len(bad)-geom}/{n-geom} labels re-derived from the source "
                   f"annotations/tables"
                   + (f"; {geom} painted bars deferred to gen/verify_geom.py" if geom else "")
                   + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[3] every counterfactual changes the answer")
    bad = [r["id"] for r in cfs if r["answer"] == by[r["cf_of"]]["answer"]]
    check(not bad, f"{len(cfs)-len(bad)}/{len(cfs)} differ" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[4] counterfactual exactness -- the guard that replaces 're-render from the seed'")
    per_fam = defaultdict(lambda: [0, 0, []])
    for r in cfs:
        o = by[r["cf_of"]]
        A, B = px(a.root, o), px(a.root, r)
        f = r["family"]; per_fam[f][1] += 1
        if f == "spatial":
            ok = A.shape == B.shape and np.array_equal(B[:, ::-1], A)
        elif f == "glyph":
            x0, y0, x1, y1 = r["difficulty"]["box"]
            m = np.ones(A.shape[:2], bool); m[y0:y1 + 1, x0:x1 + 1] = False
            ok = A.shape == B.shape and np.array_equal(A[m], B[m])
        elif f in ("counting", "chart"):
            # A pair's mask is one rectangle and serves both directions, but deriving its path
            # from the counterfactual's image name only works when the counterfactual is the
            # edited image. Reverse the pair and the counterfactual is the unsuffixed original,
            # whose "_mask.png" does not exist. A record may name the mask instead.
            mp = os.path.join(a.root, r.get("cf_mask")
                              or r["image"].replace(".png", "_mask.png"))
            if not os.path.exists(mp): ok = False
            else:
                m = np.asarray(Image.open(mp).convert("L")) == 0
                ok = A.shape == B.shape and np.array_equal(A[m], B[m])
        else: ok = False
        if ok: per_fam[f][0] += 1
        elif len(per_fam[f][2]) < 3: per_fam[f][2].append(r["id"])
    for f, (k, n, ex) in sorted(per_fam.items()):
        guard = dict(spatial="flip(cf) == original, bit-identical",
                     counting="pixels outside the pasted silhouette identical",
                     chart="pixels outside the edited-bar rectangle identical",
                     glyph="pixels outside the glyph box identical")[f]
        check(k == n, f"{f:9s} {k}/{n}  [{guard}]" + (f"  e.g. {ex}" if ex else ""))

    print("\n[5] no duplicate images (the defect that invalidated synthetic tracking)")
    h = defaultdict(list)
    for r in base:
        h[hashlib.sha256(open(os.path.join(a.root, r["image"]), "rb").read()).hexdigest()].append(r["id"])
    dup = {k: v for k, v in h.items() if len(v) > 1}
    check(not dup, f"{len(h)} distinct images among {len(base)} items"
                   + (f"  dups: {list(dup.values())[:3]}" if dup else ""))

    print("\n[6] referent uniqueness: a spatial question names two categories, each appearing once")
    bad = []
    for r in base:
        if r["family"] != "spatial": continue
        ca, cb = r["referents"]
        if ca == cb: bad.append(r["id"])
    check(not bad, f"{sum(r['family']=='spatial' for r in base)-len(bad)} spatial questions name "
                   f"two distinct categories" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[7] label spread per family")
    for f in sorted({r["family"] for r in base}):
        c = defaultdict(int)
        for r in base:
            if r["family"] == f: c[r["answer"]] += 1
        n = sum(c.values()); top = max(c.values()) / n
        print(f"  {f:9s} n={n:4d}  classes={len(c):2d}  most common {100*top:.0f}%  "
              f"chance {100/len(next(r for r in base if r['family']==f)['answer_space']):.1f}%")

    print(f"\n{'PASS' if not FAIL else str(FAIL) + ' CHECK(S) FAILED'}")
    return FAIL


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/real_test")
    p.add_argument("--coco", default="data/real/coco")
    sys.exit(main(p.parse_args()))
