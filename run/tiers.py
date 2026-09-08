"""What does each branch actually require, and what survives if you take the labels away?

Running the probe as an ANSWER rather than a router signal scores 98.6/69.3/97.3/90.7 on
held-out items where the model itself scores 34/61/76/35. That is the readout claim in its
strongest form -- but it is not a method, and reporting it as one would be the single easiest
thing for a reviewer to dismantle: a supervised linear head over a closed 3-17 way answer
space, fitted on 165 in-distribution labelled items, beating zero-shot open-ended generation
is not a surprise.

It matters anyway, because it reveals that the branch library mixes supervision levels and the
current comparison does not say so:

  none, prior, cot   need nothing
  look, attn         need referent boxes -- generator metadata here, a detector in the wild
  steer, read        need a labelled calibration set to fit the probe

A router built on probe features and compared against unsupervised published baselines is not
an apples-to-apples comparison. This tabulates the library by tier, and reports what the
oracle and the router are worth within each tier.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np

TIER = {"none": "free", "prior": "free", "cot": "free",
        "look": "boxes", "attn": "boxes",
        "steer": "labels", "read": "labels"}
ORDER = ["free", "boxes", "labels"]


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def load(split):
    P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
    npz = np.load("runs/states_3b.npz")
    meta = json.load(open("runs/states_3b_meta.json"))
    pos = {m["id"]: i for i, m in enumerate(meta)}
    V = npz["vis"].astype(np.float32)
    out = []
    for line in open(split):
        r = json.loads(line); pf = P[r["family"]]; l = pf["layer"]
        mean, scale = np.array(pf["mean"]), np.array(pf["scale"])
        pm, comp = np.array(pf["pca_mean"]), np.array(pf["components"])
        coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
        pv = softmax(coef @ (((V[pos[r["id"]], l] - mean) / scale - pm) @ comp.T) + inter)
        read = pf["classes"][int(np.argmax(pv))]
        ok = dict(r["ok"]); ok["read"] = (read == r["answer"])
        out.append(dict(id=r["id"], family=r["family"], ok=ok, answer=r["answer"],
                        puppet=(r["spec_steer"] == r["spec_class"])))
    return out


def main():
    T = load("runs/branches6_test.jsonl")
    C = load("runs/branches6_calib.jsonl")
    fams = sorted({r["family"] for r in T})
    ban = {f: np.mean([r["puppet"] for r in C if r["family"] == f]) > 0.5 for f in fams}
    BR = list(TIER)

    print(f"branch accuracy by supervision tier, held-out test ({len(T)} items)\n")
    print(f"{'branch':8s}{'tier':9s}" + "".join(f"{f:>10s}" for f in fams) + f"{'ALL':>8s}")
    for b in BR:
        acc = [np.mean([r["ok"][b] for r in T if r["family"] == f]) for f in fams]
        mark = ["*" if (b == "steer" and ban[f]) else " " for f in fams]
        print(f"{b:8s}{TIER[b]:9s}" +
              "".join(f"{100*a:9.0f}%{m}" for a, m in zip(acc, mark)) +
              f"{100*np.mean([r['ok'][b] for r in T]):7.0f}%")
    print("  * steer is a puppet string on this family and is inadmissible there\n")

    print("oracle over the library, by what you are allowed to use")
    print(f"{'allowed':22s}{'branches':10s}" + "".join(f"{f:>10s}" for f in fams) + f"{'ALL':>8s}")
    cum = []
    for t in ORDER:
        cum.append(t)
        lib = [b for b in BR if TIER[b] in cum]
        row = []
        for f in fams:
            g = [r for r in T if r["family"] == f]
            use = [b for b in lib if not (b == "steer" and ban[f])]
            row.append(np.mean([any(r["ok"][b] for b in use) for r in g]))
        allo = np.mean([any(r["ok"][b] for b in lib if not (b == "steer" and ban[r["family"]]))
                        for r in T])
        print(f"{'+'.join(cum):22s}{len(lib):<10d}" + "".join(f"{100*a:9.0f}%" for a in row) +
              f"{100*allo:7.0f}%")

    base = np.mean([r["ok"]["none"] for r in T])
    read = np.mean([r["ok"]["read"] for r in T])
    print(f"\nthe supervised readout ceiling: answering with the probe scores {100*read:.1f}% "
          f"where the model scores {100*base:.1f}%.")
    print("It is a reference point, not a baseline: it needs per-task labels and a closed")
    print("answer space, so it measures how much of the failure is readout -- it does not")
    print("compete with a zero-shot correction method. The paper must report it as a ceiling.")
    json.dump(dict(tier=TIER, read_acc=float(read), base_acc=float(base)),
              open("runs/tiers_3b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
