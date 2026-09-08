# Superseded artefacts

Nothing here feeds a paper number. `run/manifest.py` traces what the producers actually open and
fails if any path below appears in that trace, so this directory cannot silently come back into
the load path. Files are kept rather than deleted because the stage scripts that wrote them are
part of the run record.

| file | why it is not canonical |
|---|---|
| `probe_g1_smol.json` | sdpa capture. Its tracking probe reaches 100.0% at layer 2, the numeric skew `run/capture.py:33-37` documents (the skew costs the robust probes ≤1.3 pp and the memorising tracking probe 62 pp). The canonical SmolVLM synthetic tag is `smolm`. |
| `probe_g1_smol_v2.json` | superseded intermediate of the SmolVLM re-capture. |
| `layers_smol_v2.json`, `null_smol_v2.json`, `smol_v2_gen.jsonl` | the `smol_v2` sweep, superseded by `smolm`. `canon.py`'s `SYNTH` names `smolm`; the `--audit` glob over `runs/layers_*.json` skipped `smol_v2` only because no `probes_smol_v2.npy` exists, which is a guard by accident rather than by design. |
| `lora_items_base.json` | the shared-path base eval. `run/lora.py --eval-base` overwrote this on every run, so only the last survived: 75 real-chart items. Real-chart and synthetic-chart ids share a naming scheme, so 21 collided by name and `run/p0.py`'s four-way intersection fell from 298 items to 21 without raising. `p0.py` now reads base from the canonical scored generations (`runs/branches6_test.jsonl`, i.e. `canon.py`'s `GEN["3b"]`) and asserts the four configs share one split. |

`runs/v1_contaminated/` is the other quarantine directory, written by `run/stage7.sh` and
`run/stage8.sh`. It holds the pre-eager-capture run under filenames *identical* to the live ones
(`states_3b.npz`, `probes_3b.npy`, `branches6_test.jsonl`, `probe_g1.json`), so a producer invoked
with a stray relative path would read contaminated data and still emit a plausible table.
`run/manifest.py` checks for exactly that.

Two artefacts here have no live counterpart in the load path at all: no `probe_g1_*.json` file is
read by any producer any more. `canon.py`'s G1 verdict comes from `runs/null_<tag>.json`, which
recomputes all four gate conditions at the final layer where the paper reads the probe; the
`probe_g1_*` files were computed at the layer selected on the selection split. `states_smol.npz`
(sdpa) is likewise unread — `states_smolm.npz` is the canonical SmolVLM synthetic capture — and is
left in place because no producer reads any `states_*.npz` directly; the nulls are precomputed
into `runs/null_*.json` by `run/nullcal.py`.
