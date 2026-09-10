# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Assemble the (model, family) adaptation table that run/canon.py's prediction section reads.

This file exists because runs/p3_lora_matched.json had no producer. It was assembled by hand,
recorded neither the epoch it took nor the stack it adapted, and could not be regenerated -- so
the single input to the entire prediction section was unauditable.

Two things are fixed here beyond writing the file down.

Epoch rule.  run/p3_lora.py's `best_lora` takes the argmax epoch over the *held-out* split, which
is the split every reported number uses, i.e. checkpoint selection on the evaluation set. The
default rule here is `last`: the final epoch, fixed in advance, chosen before any accuracy is
looked at. In fact 14 of the 16 published values already were the final epoch; only the two Qwen
chart cells were the maximum (q3b chart 94.7 at ep1 against 92.0 at ep2, q7b chart 98.7 at ep1
against 97.3 at ep2), so the correction moves two numbers. `oracle` reproduces the old rule for
the robustness appendix.

Adapted stack.  Every one of the 16 cells adapts the language stack only, and not because it was
asked to. The Qwen rows come from the `--modules lang` runs; the SmolVLM and InternVL rows were
launched as `--modules both` but their logs report `vision=0` with "no vision-tower modules
matched", so no vision-tower parameter was ever trained for them either. The intervention is
therefore consistent across all 16 cells, and every cell is a frozen-vision-tower adaptation --
which is what the frozen-features argument needs, but it has to be read off the logs rather than
off the flag. Module counts below are the `adapted modules:` lines from runs/p3_lora.log,
runs/p3_lora_lang.log and runs/p3_lora_lang7b.log.
"""
from __future__ import annotations
import json, os, re, sys
import numpy as np

# model -> (tag for the 3-family set, tag for the chart set). Language-only throughout.
# The chart arm moved to the 466-item family; the three-family arm did not. Keeping both names
# here rather than deriving them means a rename shows up as a KeyError instead of silently
# pairing a v2 probe gain against a v1 adaptation measured on a different held-out set.
TAGS = {"q3b":  ("lora_q3b_real_3b",       "lora_q3b_real_chart_v2"),
        "q7b":  ("lora_q7b_real_3b_lang",  "lora_q7b_real_chart_v2_lang"),
        "ivl":  ("lora_ivl_real_3b",       "lora_ivl_real_chart_v2"),
        "smol": ("lora_smol_real_3b",      "lora_smol_real_chart_v2")}

# language / vision LoRA modules actually trained, from the run logs
MODULES = {"q3b": (252, 0), "q7b": (196, 0), "ivl": (268, 0), "smol": (249, 0)}

EPOCHS = 3          # every real run is --epochs 3; ep2 is the final one


def per_family(tag, ep):
    f = f"runs/lora_items_{tag}-ep{ep}.json"
    if not os.path.exists(f): return {}
    out = {}
    for k, v in json.load(open(f)).items():
        out.setdefault(k.rsplit("_", 1)[0], []).append(bool(v["ok"]))
    return out


def pick(tag, rule):
    """Return {family: (acc, n, epoch)} under the chosen epoch rule."""
    per = {ep: per_family(tag, ep) for ep in range(EPOCHS)}
    fams = sorted({f for p in per.values() for f in p})
    out = {}
    for fam in fams:
        if rule == "last":
            ep = max(e for e in per if fam in per[e])
        else:                                    # oracle: argmax on the held-out split
            ep = max((e for e in per if fam in per[e]),
                     key=lambda e: float(np.mean(per[e][fam])))
        v = per[ep][fam]
        out[fam] = (float(np.mean(v)), len(v), ep)
    return out


def main(rule="last", out="runs/p3_lora_matched.json"):
    rows = []
    for model, (t3, tc) in TAGS.items():
        for tag in (t3, tc):
            got = pick(tag, rule)
            if not got: print(f"  ! no per-item files for {tag}"); continue
            lang, vis = MODULES[model]
            for fam, (acc, n, ep) in sorted(got.items()):
                rows.append(dict(model=model, family=fam, n=n, lora=acc,
                                 epoch=ep, rule=rule, tag=tag,
                                 modules="lang" if vis == 0 else "both",
                                 n_lang_modules=lang, n_vision_modules=vis))
    rows.sort(key=lambda r: (r["model"], r["family"]))
    hdr = f"{'model':6}{'family':10}{'n':>4}{'LoRA':>8}{'ep':>4}  {'stack':6}{'tag':28}"
    print(f"epoch rule: {rule}\n"); print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:6}{r['family']:10}{r['n']:4d}{100*r['lora']:7.1f}%"
              f"{r['epoch']:4d}  {r['modules']:6}{r['tag']:28}")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}  ({len(rows)} rows)")
    if len({r["modules"] for r in rows}) == 1:
        print(f"intervention is consistent: every cell adapts the "
              f"{rows[0]['modules']} stack only")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["last"]))
