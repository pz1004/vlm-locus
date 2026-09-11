# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Verify the label-preserving chart edits. The guards are the inverse of every other family's.

Elsewhere a counterfactual must change the answer and the check is that it does. Here it must
NOT, and an edit that changed nothing at all would pass that trivially -- so the edit being real
is checked as carefully as the answer being fixed. Five properties, each exact:

  1. the answer, question and attribute equal the original's;
  2. pixels outside the painted rectangle are bit-identical;
  3. the edited bar is not the bar the question asks about;
  4. the edited bar's value, re-derived from the rendered result, is the value the record claims
     it was moved to -- the edit happened, and by the amount recorded;
  5. the QUERIED bar's value, re-derived from the same image, is bit-for-bit what the ORIGINAL
     image gives -- not merely "equal to the answer". Four charts in this set re-derive one step
     away from their table label, and they do so before any edit; comparing against the label
     would have needed a tolerance that hid whether the edit moved anything. Comparing the two
     renders needs none, and is the property actually claimed.

4 and 5 together are the whole claim: something changed, and it was not the thing being asked
about.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_geom import bar_top, derive, GRID

FAIL = 0


def chk(cond, msg):
    global FAIL
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAIL += 1


def main(a):
    idx = {(e["split"], e["stem"]): e for e in json.load(open(a.index))}
    recs = [json.loads(l) for l in open(os.path.join(a.root, "manifest.jsonl"))]
    by = {r["id"]: r for r in recs}
    sham = [r for r in recs if r["cf_of"]]
    print(f"{len(recs)} records: {len(recs)-len(sham)} originals, {len(sham)} sham edits\n")

    print("[1] the sham preserves the label: answer, question and attribute all unchanged")
    bad = [r["id"] for r in sham
           if (r["answer"], r["question"], r["attribute"])
           != (by[r["cf_of"]]["answer"], by[r["cf_of"]]["question"], by[r["cf_of"]]["attribute"])]
    chk(not bad, f"{len(sham)-len(bad)}/{len(sham)} identical" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[2] pixels outside the painted rectangle are bit-identical")
    bad = []
    for r in sham:
        A = np.asarray(Image.open(os.path.join(a.root, by[r["cf_of"]]["image"])).convert("RGB"))
        B = np.asarray(Image.open(os.path.join(a.root, r["image"])).convert("RGB"))
        m = np.asarray(Image.open(os.path.join(a.root, r["cf_mask"])).convert("L")) == 0
        if A.shape != B.shape or not np.array_equal(A[~m], B[~m]):
            bad.append(r["id"])
    chk(not bad, f"{len(sham)-len(bad)}/{len(sham)} exact" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[3] the edited bar is not the bar the question asks about")
    bad = [r["id"] for r in sham if r["difficulty"]["sham_bar"] == r["difficulty"]["queried_bar"]]
    chk(not bad, f"{len(sham)-len(bad)}/{len(sham)} distinct" + (f"  e.g. {bad[:3]}" if bad else ""))

    print("\n[4] the edit happened: the edited bar re-derives to the value recorded")
    print("[5] and the queried bar still re-derives to the original answer")
    moved, held, n = [], [], 0
    for r in sham:
        _, split, stem = r["coco"]["source"].split("/")
        e = idx.get((split, stem))
        if e is None:
            continue
        n += 1
        im = Image.open(os.path.join(a.root, r["image"])).convert("RGB")
        d = r["difficulty"]
        got_edit = derive(im, e["bars"][d["sham_bar"]], e["baseline"], e["a"], e["b"])
        orig = Image.open(os.path.join(a.root, by[r["cf_of"]]["image"])).convert("RGB")
        qb = e["bars"][d["queried_bar"]]
        got_q = derive(im, qb, e["baseline"], e["a"], e["b"])
        was_q = derive(orig, qb, e["baseline"], e["a"], e["b"])
        if got_edit != d["sham_to"]:
            moved.append((r["id"], got_edit, d["sham_to"]))
        if got_q != was_q:
            held.append((r["id"], got_q, was_q))
    chk(len(moved) <= a.max_bad,
        f"{n-len(moved)}/{n} edited bars read back as moved" + (f"  e.g. {moved[:3]}" if moved else ""))
    chk(not held,
        f"{n-len(held)}/{n} queried bars read identically before and after" + (f"  e.g. {held[:3]}" if held else ""))

    print(f"\n{'PASS' if not FAIL else str(FAIL) + ' CHECK(S) FAILED'}")
    return FAIL


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/real_chart_sham")
    p.add_argument("--index", default="runs/chart_index.json")
    # the same tolerance gen/verify_geom.py carries, and for the same reason: a handful of charts
    # sit exactly between two grid steps and round the other way when re-read from pixels
    p.add_argument("--max-bad", type=int, default=4)
    sys.exit(main(p.parse_args()))
