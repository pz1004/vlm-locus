"""Can the probe's own confidence say when to trust it?

The oracle over {model answer, probe answer} reaches 82.6%, within a point of a full fine-tune.
That headroom is only reachable with a selector, and section 8 showed that label-free selectors
have no power. This asks the narrower question the oracle actually poses: does the *probe's own*
confidence -- which costs nothing extra, it is already computed -- separate the items where it
should override the model from the ones where it should defer?

The threshold is fitted per family on the calibration split and applied once to test. A gate
that cannot beat always-read is a gate that is not measuring anything.
"""
import json
import numpy as np
from scipy.stats import binomtest

def softmax(x):
    x = np.asarray(x, float); e = np.exp(x - x.max()); return e / e.sum()

P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
npz = np.load("runs/states_3b.npz"); meta = json.load(open("runs/states_3b_meta.json"))
pos = {m["id"]: i for i, m in enumerate(meta)}
V = npz["vis"].astype(np.float32)
calib = json.load(open("runs/calib_ids.json"))
gold = {m["id"]: str(m["answer"]) for m in meta}

model_ans, model_ok = {}, {}
for l in open("runs/branches6_test.jsonl"):
    r = json.loads(l); model_ans[r["id"]] = str(r["out"]["none"]); model_ok[r["id"]] = bool(r["ok"]["none"])
for l in open("runs/branches6_calib.jsonl"):
    r = json.loads(l); model_ans[r["id"]] = str(r["out"]["none"]); model_ok[r["id"]] = bool(r["ok"]["none"])

def probe(iid, fam):
    pf = P[fam]; i = pos[iid]
    z = ((V[i, pf["layer"]] - np.array(pf["mean"])) / np.array(pf["scale"])
         - np.array(pf["pca_mean"])) @ np.array(pf["components"]).T
    p = softmax(np.array(pf["coef"]) @ z + np.array(pf["intercept"]))
    k = int(np.argmax(p))
    return pf["classes"][k], float(p[k])

test = {f: P[f]["test_ids"] for f in P}
fams = sorted(P)
print(f"{'':22s}" + "".join(f"{f:>10s}" for f in fams) + f"{'ALL':>8s}")
rows = {}
for name in ["model", "read", "gate"]: rows[name] = {}
thr = {}
for f in fams:
    ca = [(i, *probe(i, f)) for i in calib[f]]
    grid = np.linspace(0, 1, 21)
    def acc(items, t):
        return np.mean([(a == gold[i]) if c >= t else model_ok[i] for i, a, c in items])
    thr[f] = float(grid[int(np.argmax([acc(ca, t) for t in grid]))])
    te = [(i, *probe(i, f)) for i in test[f]]
    for i, a, c in te:
        rows["model"][i] = model_ok[i]
        rows["read"][i] = (a == gold[i])
        rows["gate"][i] = (a == gold[i]) if c >= thr[f] else model_ok[i]
ids = list(rows["read"])
for name in ["model", "read", "gate"]:
    per = [100 * np.mean([rows[name][i] for i in test[f]]) for f in fams]
    print(f"{name:22s}" + "".join(f"{a:9.0f}%" for a in per)
          + f"{100*np.mean([rows[name][i] for i in ids]):7.1f}%")
n01 = sum(1 for i in ids if rows["gate"][i] and not rows["read"][i])
n10 = sum(1 for i in ids if rows["read"][i] and not rows["gate"][i])
p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
print(f"\nthresholds fitted on calibration: " + ", ".join(f"{f} {thr[f]:.2f}" for f in fams))
print(f"gate vs always-read: {n01} won / {n10} lost, McNemar p={p:.2g}")
