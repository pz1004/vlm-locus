"""Emit the calibration (selection) split: the items the probe was fitted with but that are in
neither its training nor its test split. Router thresholds are fitted here, so it must be
disjoint from both -- fitting them on probe-training items would report a probe that has seen
its own answers, and fitting them on test items is straightforward test-set fitting."""
import json, sys
import numpy as np

probes, meta_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
P = np.load(probes, allow_pickle=True).item()
meta = json.load(open(meta_path))
cal = {}
for f, pf in P.items():
    tr, te = set(pf["train_ids"]), set(pf["test_ids"])
    # the probe drops classes with <8 examples; those items are in no split at all
    keep = tr | te | {m["id"] for m in meta if m["family"] == f}
    cal[f] = sorted({m["id"] for m in meta if m["family"] == f} - tr - te)
json.dump(cal, open(out, "w"), indent=1)
print("calibration split: " + "  ".join(f"{f}={len(v)}" for f, v in cal.items()))
