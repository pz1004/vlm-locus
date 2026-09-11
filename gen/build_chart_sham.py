# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Label-preserving (sham) chart edits: the appearance changes, the answer does not.

Every counterfactual elsewhere in this study changes the answer, so a probe that follows one has
shown it responds to *something* the edit did -- not that it responds to the attribute rather
than to a cue that moves with it. The sham is the other half of that test. It applies the same
primitive to a bar the question does not ask about: a rectangle painted from the bar's old top to
a new one, in the bar's own colour. The queried bar is untouched, so the answer is unchanged by
construction, and a reader of the attribute must keep its answer.

The edit is the same size as that item's real counterfactual, so "the probe follows the real edit
and not the sham" cannot be explained by the sham being smaller. Where the distractor cannot take
that delta without exceeding the axis, the item is dropped rather than edited by a different
amount.

Guards, all exact: pixels outside the painted rectangle are bit-identical; the answer equals the
original's; and gen/verify_geom.py re-derives the QUERIED bar's value from the rendered result and
must get the original answer back.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_geom import bar_top, GRID


def main(a):
    idx = {(e["split"], e["stem"]): e for e in json.load(open(a.index))}
    src = [json.loads(l) for l in open(os.path.join(a.src, "manifest.jsonl"))]
    by = {r["id"]: r for r in src}
    base = [r for r in src if r["cf_of"] is None]
    cf_of = {r["cf_of"]: r for r in src if r["cf_of"]}
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    man, drop = [], {"no_index": 0, "no_cat": 0, "no_room": 0, "no_bar": 0}

    for r in base:
        c = cf_of.get(r["id"])
        srcpath = (r.get("coco") or {}).get("source")
        if c is None or not srcpath:
            drop["no_index"] += 1
            continue
        _, split, stem = srcpath.split("/")
        e = idx.get((split, stem))
        if e is None:
            drop["no_index"] += 1
            continue
        if not r["referents"] or r["referents"][0] not in e["cats"]:
            drop["no_cat"] += 1
            continue
        qi = e["cats"].index(r["referents"][0])
        delta = int(c["answer"]) - int(r["answer"])          # the real edit's magnitude
        im = Image.open(os.path.join(a.src, r["image"])).convert("RGB")

        # the distractor with the most headroom that can take the same delta, deterministic
        cands = []
        for j, (x0, x1, ytop, colour) in enumerate(e["bars"]):
            if j == qi or j >= len(e["vals"]):
                continue
            v_old = int(round(e["vals"][j] / GRID) * GRID)
            v_new = v_old + delta
            if not (0 <= v_new <= 100) or v_new == v_old:
                continue
            cands.append((100 - v_old, -j, j, v_old, v_new))
        if not cands:
            drop["no_room"] += 1
            continue
        _, _, j, v_old, v_new = max(cands)
        x0, x1, _, colour = e["bars"][j]

        # the bar's top is measured in the rendered image, not taken from the index: the source
        # images had their printed labels painted out, and a top that disagreed with the
        # calibration would mean the edit is being placed by an assumption rather than a reading
        top = bar_top(im, x0, x1, colour)
        if top is None:
            drop["no_bar"] += 1
            continue
        new_top = int(round(e["baseline"] - (e["a"] * v_new + e["b"])))
        if not (0 <= new_top < top):
            drop["no_room"] += 1
            continue

        A = np.asarray(im).copy()
        A[new_top:top, x0:x1 + 1] = np.array(colour, np.uint8)
        sid = f"{r['id']}_sham"
        Image.fromarray(A).save(os.path.join(a.out, "images", f"{sid}.png"))
        m = np.full(A.shape[:2], 255, np.uint8)
        m[new_top:top, x0:x1 + 1] = 0                        # 0 = edited, as the other families
        Image.fromarray(m).save(os.path.join(a.out, "images", f"{sid}_mask.png"))
        shutil.copyfile(os.path.join(a.src, r["image"]),
                        os.path.join(a.out, "images", os.path.basename(r["image"])))

        man.append(dict(r, spec=dict(r.get("spec") or {})))
        man.append(dict(r, id=sid, cf_of=r["id"], cf_kind="sham_bar",
                        image=f"images/{sid}.png",
                        cf_mask=f"images/{sid}_mask.png",
                        # the answer, the attribute and the question are the original's: that is
                        # what makes it a sham. Only the recorded edit differs.
                        difficulty=dict(r["difficulty"], sham_bar=j, sham_from=v_old,
                                        sham_to=v_new, queried_bar=qi, delta=delta)))

    with open(os.path.join(a.out, "manifest.jsonl"), "w") as fh:
        for r in man:
            fh.write(json.dumps(r) + "\n")
    n = sum(1 for r in man if r["cf_of"])
    print(f"wrote {a.out}: {n} sham pairs from {len(base)} base items")
    print(f"  dropped: " + ", ".join(f"{k} {v}" for k, v in drop.items() if v))
    d = [r["difficulty"]["delta"] for r in man if r["cf_of"]]
    print(f"  edit size matches the real counterfactual, {min(d)} to {max(d)} points")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/real_chart_v2")
    p.add_argument("--index", default="runs/chart_index.json")
    p.add_argument("--out", default="data/real_chart_sham")
    main(p.parse_args())
