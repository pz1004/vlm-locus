"""Where in the stack does the answer live, and where does it go?

Fits the same capacity-controlled probe at *every* layer on the *same* train split, and
evaluates on the *same* held-out test split that run/branches.py scores the model on. The
model's own accuracy on those items is a horizontal line on the same axes.

The signature we are testing for is a readout failure: the target attribute is linearly
decodable from the residual stream at some layer L at accuracy far above what the model
emits, and that decodability is *still present at the final layer* (the information was never
lost -- it was never read) or *decays* (the information is destroyed downstream). The two have
different implications, and no branch in the correction library can tell them apart.

Blind states are decoded with the same probe to price the language prior at every depth.
"""
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]),
              glyph=lambda m: int(m["attribute"]["value"]))


def main(states="runs/states_3b.npz", branches="runs/branches6_test.jsonl",
         out="runs/layers_3b.json"):
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
    # accepts either a branch grid (`ok.none`) or a plain scoring pass (`gen_correct`), so a
    # second model can be replicated without running the correction library it does not need
    # per-item, so model accuracy can be restricted to the SAME held-out split the probe is
    # scored on. Averaging the model over all items while the probe is scored on a quarter of
    # them compares two different samples: on real charts that made the base look 88.0% against
    # 81.3% on the test items, understating the gap by 6.7 pp.
    model_ok, keep_ids = {}, set()
    for line in open(branches):
        r = json.loads(line)
        ok = r["ok"]["none"] if "ok" in r else r["gen_correct"]
        model_ok[r["id"]] = bool(ok); keep_ids.add(r["id"])
    res = {}
    L = V.shape[1]
    for f in sorted(TARGET):
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        if len(idx) == 0: continue          # TARGET spans every family ever built, not this set
        y = np.array([TARGET[f](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8], dtype=int)
        idx, y = idx[keep], y[keep]
        tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=0, stratify=y)
        _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
        chance = meta[idx[0]]["chance"]
        te_ids = [meta[i]["id"] for i in idx[te]]
        m_te = [model_ok[i] for i in te_ids if i in model_ok]
        vis, blind = [], []
        for l in range(L):
            pipe = make_pipeline(StandardScaler(),
                                 PCA(n_components=min(64, len(tr) - 1), random_state=0),
                                 LogisticRegression(max_iter=1000, C=0.5))
            pipe.fit(V[idx[tr], l], y[tr])
            vis.append(float(pipe.score(V[idx[te], l], y[te])))
            blind.append(float(pipe.score(B[idx[te], l], y[te])))
        res[f] = dict(vis=vis, blind=blind, chance=float(chance),
                      model=float(np.mean(m_te)) if m_te else float('nan'),
                      model_all=float(np.mean([v for i, v in model_ok.items()
                                               if i.rsplit('_', 1)[0] == f])),
                      peak_layer=int(np.argmax(vis)), peak=float(max(vis)),
                      final=float(vis[-1]), n_test=int(len(te)))
        print(f"{f:10s} n_te={len(te):3d}  chance {100*chance:4.1f}%  model {100*res[f]['model']:4.1f}%"
              f"   peak L{res[f]['peak_layer']:2d} {100*res[f]['peak']:5.1f}%"
              f"   final L{L-1} {100*res[f]['final']:5.1f}%")
    print()
    # the curve itself, decimated so it reads in a terminal
    step = max(1, L // 12)
    cols = list(range(0, L, step)) + [L - 1]
    print(f"{'family':10s}" + "".join(f"{'L'+str(l):>7s}" for l in cols))
    for f in sorted(res):
        print(f"{f:10s}" + "".join(f"{100*res[f]['vis'][l]:6.0f}%" for l in cols))
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
