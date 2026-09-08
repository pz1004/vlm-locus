"""Artefact-provenance checks for the two Stage 3 reproducibility items.

Item 14: run/p0.py's base row must come from the canonical scored generations -- the same file
run/canon.py reads for each cell's model_acc -- and no producer may read a path another producer
overwrites. Item 15: run/manifest.py's invariants, plus the check that a re-run of canon.py
reproduces what paper/tables/ currently holds.

    .venv/bin/python run/verify_stage3.py
"""
import json, os, re, subprocess, sys, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = ".venv/bin/python"
ck = []
def chk(name, ok, note=""):
    ck.append((name, ok, note)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {note}" if note else ""))

# --- item 14 ---
src = {f: open(f).read() for f in glob.glob("run/*.py")}
readers = [f for f, t in src.items()
           if re.search(r'open\(\s*f?["\']runs/lora_items_base\.json', t) and "manifest" not in f]
chk("no producer reads the shared-path lora_items_base.json", not readers, f"readers={readers}")

writers = [f for f, t in src.items() if re.search(r'evaluate\(.*["\']base["\']\s*\)', t)]
chk("lora.py no longer writes the bare 'base' tag", not writers, f"writers={writers}")
chk("lora.py base eval is tagged per run", 'f"base_{a.tag}"' in src["run/lora.py"])

chk("lora_items_base.json is out of runs/", not os.path.exists("runs/lora_items_base.json"))
chk("...and preserved in the quarantine", os.path.exists("runs/superseded/lora_items_base.json"))

# p0 reproduces byte-identically from the canonical generations
subprocess.run([PY, "run/p0.py"], capture_output=True)
a = json.load(open("runs/p0_3b.json.published")); b = json.load(open("runs/p0_3b.json"))
chk("p0_3b.json regenerates byte-identically", json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))

# the base row equals canon's model column, by construction
import numpy as np
sys.path.insert(0, "run"); import canon
P = np.load("runs/probes_3b.npy", allow_pickle=True).item()
G = {}
for line in open(canon.GEN["3b"]):
    r = json.loads(line); G[r["id"]] = bool(r["ok"]["none"] if "ok" in r else r["gen_correct"])
L = json.load(open("runs/layers_3b.json"))
same = all(abs(np.mean([G[i] for i in P[f]["test_ids"] if i in G]) - L[f]["model"]) < 1e-9 for f in P)
chk("p0's base source == canon's model_acc source, all four families", same)
chk("p0 base row matches canon model column", all(
    abs(b["acc"]["base"][f] - L[f]["model"]) < 1e-9 for f in P))

# the guard fires on the failure that shipped
import p0
old = {k: dict(ok=v["ok"]) for k, v in json.load(open("runs/superseded/lora_items_base.json")).items()}
cfg = {"base": old}
for t, n in [("lora","both"),("loralang","lang"),("loravis","vis")]: cfg[n], _ = p0.best_epoch(t)
ids = set.intersection(*[set(d) for d in cfg.values()])
chk("split-correspondence guard fires on the shipped failure",
    len(ids) < min(len(d) for d in cfg.values()), f"{len(ids)} vs {min(len(d) for d in cfg.values())}")
cfg["base"] = p0.base_items()
ids2 = set.intersection(*[set(d) for d in cfg.values()])
chk("...and is silent on the canonical path",
    len(ids2) == min(len(d) for d in cfg.values()), f"{len(ids2)} items")

# --- item 15 ---
r = subprocess.run([PY, "run/manifest.py"], capture_output=True, text=True)
chk("manifest verifies clean", r.returncode == 0 and "PASS" in r.stdout)

m = json.loads(subprocess.run([PY, "run/manifest.py", "--json"], capture_output=True, text=True).stdout)
reads = m["reads"]
chk("no quarantined path in the load path",
    not any(k.startswith(("runs/v1_contaminated/", "runs/superseded/")) for k in reads))
chk("no smol_v2 artefact in the load path", not any("smol_v2" in k for k in reads))
chk("no probe_g1_* artefact in the load path", not any("probe_g1" in k for k in reads),
    "G1 now comes from null_<tag>.json at the final layer")
chk("every read path exists", all(os.path.exists(k) for k in reads), f"{len(reads)} artefacts")
chk("every non-canonical lookalike sits inside a quarantine dir",
    all(k.startswith(("runs/v1_contaminated/", "runs/superseded/")) for k in m["noncanonical"]))
chk("quarantine README present", os.path.exists("runs/superseded/README.md"))

# --- canon.py is idempotent, so the tables cannot drift on a re-run ------------------------
import shutil, tempfile
d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
for d in (d1, d2):
    subprocess.run([PY, "run/canon.py"], capture_output=True,
                   env=dict(os.environ, VLM_LOCUS_TEX=d, VLM_LOCUS_JSON=os.path.join(d, "c.json")))
names = ("bands", "cells_real", "cells_synthetic", "facts", "pairs", "prediction")
drift = [f for f in names if open(f"{d1}/{f}.tex").read() != open(f"{d2}/{f}.tex").read()]
chk("canon.py is idempotent across two full regenerations", not drift, f"drift={drift}")
live = [f for f in names if os.path.exists(f"../paper/tables/{f}.tex")
        and open(f"{d1}/{f}.tex").read() != open(f"../paper/tables/{f}.tex").read()]
chk("paper/tables matches a fresh canon.py run", not live, f"stale={live}")
shutil.rmtree(d1, ignore_errors=True); shutil.rmtree(d2, ignore_errors=True)

print(f"\n{sum(1 for _,o,_ in ck if o)}/{len(ck)} checks pass")
sys.exit(0 if all(o for _,o,_ in ck) else 1)
