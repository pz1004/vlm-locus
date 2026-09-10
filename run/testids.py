# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Write runs/testids_<tag>.jsonl: the held-out ids run/lora.py evaluates on.

These files existed in runs/ with no producer in the repository, which meant an adaptation run
on a new dataset could not be reproduced at all -- lora.py requires the file and nothing makes
it. The ids are not arbitrary: they are exactly the test quarter of run/layers.py's 55/20/25
stratified split, so the probe and the adaptation are scored on the same items. This regenerates
them from that rule, and verifies against any file already present rather than overwriting it.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from sklearn.model_selection import train_test_split

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]), spatial=lambda m: m["attribute"]["relation"], chart=lambda m: int(m["attribute"]["value"]), tracking=lambda m: int(m["attribute"]["end"]), glyph=lambda m: int(m["attribute"]["value"]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon import MIN_CLASS, pairs_of, split


def ids_for(tag):
    meta = json.load(open(f"runs/states_{tag}_meta.json"))
    out = []
    for fam in sorted({m["family"] for m in meta} & set(TARGET)):
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == fam])
        y = np.array([TARGET[fam](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS], dtype=int)
        idx, y = idx[keep], y[keep]
        tr, _, te = split(y, groups=pairs_of(meta, idx), seed=0)
        out += [dict(id=meta[i]["id"], family=fam) for i in idx[te]]
    return out


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        rows = ids_for(tag)
        p = f"runs/testids_{tag}.jsonl"
        if os.path.exists(p):
            old = [json.loads(l) for l in open(p)]
            same = {r["id"] for r in old} == {r["id"] for r in rows}
            print(f"  {p}  exists, {len(old)} ids, reproduces: {same}")
            if not same:
                raise SystemExit(f"{p} disagrees with the split rule -- refusing to overwrite")
            continue
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote {p}  ({len(rows)} ids)")
