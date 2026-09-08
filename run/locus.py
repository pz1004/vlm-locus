# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""The locus split: on items the model gets WRONG, is the visual attribute still decodable?

This is the question the whole project is about, and it is answerable from two things already
computed -- the probe fitted in run/probe.py and the generation correctness in the pilot run.
Per item, on the held-out test split only:

    decodable   = the frozen linear probe recovers the true attribute from the visual state
    answered    = free generation gives the true answer

    wrong & decodable      -> R      readout: the evidence is there and unused
    wrong & not decodable  -> P      perception: the evidence is not recoverable
    right & not decodable  -> L?     answered without recoverable evidence (prior or luck)
"""
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ST = sys.argv[1] if len(sys.argv) > 1 else "runs/states_3b.npz"
GEN = sys.argv[2] if len(sys.argv) > 2 else "runs/cal_3b_gen.jsonl"
G1 = sys.argv[3] if len(sys.argv) > 3 else "runs/probe_g1.json"
npz = np.load(ST)
meta = json.load(open(ST.replace(".npz", "_meta.json")))
V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
gen = {r["id"]: r for r in (json.loads(l) for l in open(GEN))}
g1 = json.load(open(G1))
TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]))

def probe(k=64, C=0.5):
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                         LogisticRegression(max_iter=1000, C=C))

print(f"{'family':10s} {'n test':>7s} {'model err':>10s} {'probe acc':>10s} "
      f"{'R (wrong,decodable)':>21s} {'P (wrong,not)':>15s} {'L (right,not dec.)':>19s}")
rows = {}
for f in sorted(g1):
    idx = np.array([i for i, m in enumerate(meta) if m["family"] == f and m["id"] in gen])
    y = np.array([TARGET[f](meta[i]) for i in idx])
    keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8])
    idx, y = idx[keep], y[keep]
    tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=0, stratify=y)
    _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
    l = g1[f]["layer"]
    clf = probe(min(64, len(tr) - 1)); clf.fit(V[idx[tr], l], y[tr])
    dec = clf.predict(V[idx[te], l]) == y[te]
    ok = np.array([gen[meta[i]["id"]]["gen_correct"] for i in idx[te]])
    n = len(te)
    R = float(((~ok) & dec).mean()); P = float(((~ok) & ~dec).mean()); L = float((ok & ~dec).mean())
    rows[f] = dict(n=n, err=float((~ok).mean()), probe=float(dec.mean()), R=R, P=P, L=L,
                   R_of_wrong=float((dec[~ok]).mean()) if (~ok).any() else float("nan"))
    print(f"{f:10s} {n:7d} {100*(~ok).mean():9.0f}% {100*dec.mean():9.0f}% "
          f"{100*R:20.0f}% {100*P:14.0f}% {100*L:18.0f}%")
print()
print("Share of the model's ERRORS in which the attribute is still decodable "
      "(the readout share):")
for f, d in rows.items():
    print(f"   {f:10s} {100*d['R_of_wrong']:5.0f}%   of {int(round(d['err']*d['n']))} errors")
json.dump(rows, open(G1.replace("probe_g1", "locus"), "w"), indent=1)
