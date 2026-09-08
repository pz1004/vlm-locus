# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""How much of the oracle is real, and how much is four noisy branches getting lucky?

`oracle = any branch correct` is inflated whenever branches make partly independent errors:
four conditions at 35% accuracy produce a high `any-correct` rate even with zero item-specific
skill. Some of the 45 points of headroom the router fails to reach may therefore be
unreachable by ANY router, because there is nothing systematic to route on.

The null: permute each branch's outcome vector independently across items within a family.
That preserves every branch's marginal accuracy and destroys only the item-branch coupling --
exactly the thing a router needs. The gap between the true oracle and the permuted oracle is
the recoverable headroom; the router should be scored against that, not against the raw
oracle.
"""
import json
import numpy as np

BR = ["none", "steer", "prior", "look"]
R = 2000
rng = np.random.default_rng(0)

tst = [json.loads(l) for l in open("runs/branches_3b.jsonl")]
fams = sorted({r["family"] for r in tst})
prev = json.load(open("runs/policy_3b.json"))

print(f"{'family':10s}{'n':>5s}{'best fix':>10s}{'oracle':>8s}{'null oracle':>13s}"
      f"{'real headroom':>15s}{'chance part':>13s}")
tot = dict(bf=[], orc=[], nul=[])
rows = {}
for f in fams:
    g = [r for r in tst if r["family"] == f]
    M = np.array([[r["ok"][b] for b in BR] for r in g], bool)
    bf = M.mean(0).max()
    orc = M.any(1).mean()
    nulls = np.empty(R)
    for t in range(R):
        Q = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
        nulls[t] = Q.any(1).mean()
    nul = nulls.mean()
    real = orc - nul                      # headroom attributable to item-specific coupling
    chance = nul - bf                     # headroom that any-of-four gets for free
    rows[f] = dict(n=len(g), bf=float(bf), orc=float(orc), nul=float(nul),
                   real=float(real), chance=float(chance),
                   p=float((nulls >= orc).mean()))
    print(f"{f:10s}{len(g):5d}{100*bf:9.0f}%{100*orc:7.0f}%{100*nul:12.0f}%"
          f"{100*real:14.1f}pp{100*chance:12.1f}pp")

M = np.array([[r["ok"][b] for b in BR] for r in tst], bool)
bf = M.mean(0).max(); orc = M.any(1).mean()
nulls = np.empty(R)
for t in range(R):
    idx = {f: [i for i, r in enumerate(tst) if r["family"] == f] for f in fams}
    Q = M.copy()
    for f in fams:
        ii = idx[f]
        for j in range(M.shape[1]): Q[ii, j] = rng.permutation(M[ii, j])
    nulls[t] = Q.any(1).mean()
nul = nulls.mean()
print(f"{'ALL':10s}{len(tst):5d}{100*bf:9.0f}%{100*orc:7.0f}%{100*nul:12.0f}%"
      f"{100*(orc-nul):14.1f}pp{100*(nul-bf):12.1f}pp")

lrc = prev["B"]
print(f"\n  Scored against the RAW oracle:       LRC captures "
      f"{100*(lrc-bf)/(orc-bf):.0f}% of {100*(orc-bf):.1f} pp")
print(f"  Scored against the CHANCE-CORRECTED   LRC captures "
      f"{100*(lrc-nul)/(orc-nul):.0f}% of {100*(orc-nul):.1f} pp"
      if orc > nul else "  chance-corrected headroom is zero")
print(f"\n  permutation p(null oracle >= observed): "
      f"{ {f: round(rows[f]['p'], 3) for f in fams} }")
print("\n  A family whose 'real headroom' is small has branches that succeed and fail together;")
print("  no per-instance router can help there however good its signal.")
json.dump(rows, open("runs/oracle_null_3b.json", "w"), indent=1)
