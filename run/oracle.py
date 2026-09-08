# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Gate G3: does the oracle over corrections clear the best fixed branch?

Reports, per family:
  * accuracy of each always-on branch (these are the S1-S4 baselines of the plan)
  * the oracle over the library -- the ceiling on any router
  * headroom = oracle minus the best single fixed branch. Below ~3 pp there is nothing for a
    router to win, however good its signal.
  * the puppet-string rate: how often steering toward a deliberately WRONG class produces
    that class. A branch that follows wherever it is pushed is discarded for that cell, and
    its contribution to the oracle is recomputed without it.
"""
import json, sys
import numpy as np
from collections import defaultdict

rs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/branches_3b.jsonl")]
BR = [b for b in ("none", "steer", "prior", "look", "attn", "cot") if b in rs[0]["ok"]]
fams = sorted({r["family"] for r in rs})
print(f"{len(rs)} items\n")
print(f"{'family':10s} {'n':>4s} " + "".join(f"{b:>8s}" for b in BR) +
      f"{'oracle':>8s} {'best fix':>9s} {'headroom':>9s} {'puppet':>8s}")
tot = defaultdict(list)
out = {}
for f in fams:
    g = [r for r in rs if r["family"] == f]
    acc = {b: np.mean([r["ok"][b] for r in g]) for b in BR}
    puppet = np.mean([r["spec_steer"] == r["spec_class"] for r in g])
    usable = [b for b in BR if not (b == "steer" and puppet > 0.5)]
    orc = np.mean([any(r["ok"][b] for b in usable) for r in g])
    best = max(acc[b] for b in usable)
    head = orc - best
    out[f] = dict(n=len(g), acc={b: float(acc[b]) for b in BR}, oracle=float(orc),
                  best_fixed=float(best), headroom=float(head), puppet=float(puppet),
                  steer_discarded=bool(puppet > 0.5))
    print(f"{f:10s} {len(g):4d} " + "".join(f"{100*acc[b]:7.0f}%" for b in BR) +
          f"{100*orc:7.0f}% {100*best:8.0f}% {100*head:+8.1f}pp {100*puppet:7.0f}%"
          + ("  <- steer discarded" if puppet > 0.5 else ""))
print()
a = {b: np.mean([r["ok"][b] for r in rs]) for b in BR}
allorc = np.mean([any(r["ok"][b] for b in BR
                      if not (b == "steer" and out[r["family"]]["steer_discarded"]))
                  for r in rs])
bestg = max(a.values())
print(f"{'ALL':10s} {len(rs):4d} " + "".join(f"{100*a[b]:7.0f}%" for b in BR) +
      f"{100*allorc:7.0f}% {100*bestg:8.0f}% {100*(allorc-bestg):+8.1f}pp")
print()
print("G3 asks whether the oracle beats the best fixed branch by >=3 pp in at least half the")
print("families. That is the precondition for a router: it is the accuracy a perfect")
print("per-instance selector would reach, so no signal can do better.\n")
passing = sum(1 for f in fams if out[f]["headroom"] >= 0.03)
print(f"  families with >=3 pp headroom: {passing}/{len(fams)}"
      f"   -> G3 {'PASS' if passing >= len(fams)/2 else 'FAIL'}")
for f in fams:
    d = out[f]
    win = [b for b in BR if d['acc'][b] == d['best_fixed']][0]
    print(f"    {f:10s} headroom {100*d['headroom']:+5.1f} pp   best fixed branch: {win}"
          + ("   (steer unusable: puppet string)" if d["steer_discarded"] else ""))
json.dump(out, open("runs/oracle_%d.json" % len(BR), "w"), indent=1)
