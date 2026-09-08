# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Probe-reranked sampling: select among samples by visual evidence, not by frequency.

Best-of-N exposes a gap the branch grid could not: over 8 samples the correct answer appears
for 76% of test items, but majority vote recovers 51% -- below greedy. The mode of the sample
distribution is wrong even when the right answer is in the distribution. Frequency is the
wrong selector.

The probe is a selector that does not use frequency. It reads p(class | visual state) from the
model's own residual stream, so it scores a candidate by how well the image supports it,
independent of how often the decoder emitted it. If readout failure is real -- the visual state
carries the answer the decoder does not emit -- then reranking samples by the probe should
recover a large part of the sampling oracle, and it costs nothing beyond the probe read the
router already pays.

Three selectors on identical samples:
  vote     majority (standard self-consistency)
  probe    argmax over samples of p_probe(sample | h_vis)
  hybrid   probe, restricted to samples that appear at least twice (frequency as a prior,
           evidence as the tie-break)
"""
from __future__ import annotations
import json, sys
from collections import Counter
import numpy as np

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def main(bon_path="runs/bon_3b.jsonl"):
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz")
    meta = json.load(open("runs/states_3b_meta.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V = npz["vis"].astype(np.float32)
    rows = [json.loads(l) for l in open(bon_path)]
    fams = sorted({r["family"] for r in rows})
    K = max(len(r["samples"]) for r in rows)

    out = []
    for r in rows:
        pf = P[r["family"]]; l = pf["layer"]
        mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
        pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
        coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
        cls = pf["classes"]; i = pos[r["id"]]
        pv = softmax(coef @ (((V[i, l] - mean) / scale - pm) @ comp.T) + inter)
        score = {c: float(pv[cls.index(c)]) if c in cls else 0.0
                 for c in set(s for s in r["samples"] if s is not None)}
        out.append(dict(r, score=score))

    def vote(ss):
        ss = [s for s in ss if s is not None]
        return Counter(ss).most_common(1)[0][0] if ss else None

    def probe_pick(ss, score):
        ss = [s for s in ss if s is not None]
        return max(ss, key=lambda s: score.get(s, 0.0)) if ss else None

    def hybrid(ss, score):
        ss = [s for s in ss if s is not None]
        if not ss: return None
        c = Counter(ss); rep = [s for s in set(ss) if c[s] >= 2] or list(set(ss))
        return max(rep, key=lambda s: score.get(s, 0.0))

    print(f"probe-reranked sampling, {len(out)} items, {K} samples each\n")
    print(f"{'family':10s}{'greedy':>8s}{'vote':>8s}{'probe':>8s}{'hybrid':>8s}"
          f"{'any/K':>8s}{'probe cap':>11s}")
    for f in fams + ["ALL"]:
        g = [r for r in out if f == "ALL" or r["family"] == f]
        gv = np.mean([vote(r["samples"]) == r["answer"] for r in g])
        gp = np.mean([probe_pick(r["samples"], r["score"]) == r["answer"] for r in g])
        gh = np.mean([hybrid(r["samples"], r["score"]) == r["answer"] for r in g])
        ga = np.mean([any(s == r["answer"] for s in r["samples"]) for r in g])
        g1 = np.mean([r["samples"][0] == r["answer"] for r in g])
        cap = (gp - gv) / (ga - gv) if ga > gv else float("nan")
        print(f"{f:10s}{100*g1:7.0f}%{100*gv:7.0f}%{100*gp:7.0f}%{100*gh:7.0f}%"
              f"{100*ga:7.0f}%{100*cap:10.0f}%")
    print("\n'probe cap' = share of the gap between majority vote and the sampling oracle that")
    print("the probe closes. It is the readout-failure share made operational: the answer was")
    print("in the samples, frequency could not find it, visual evidence could.")

    # does it beat the router's single-shot result, and at what cost?
    print("\nas a branch: cost = K sampled passes + one probe read")
    json.dump([{k: v for k, v in r.items() if k != "score"} for r in out],
              open("runs/rerank_3b.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/bon_3b.jsonl")
