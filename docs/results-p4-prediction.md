# P4 — the instrument earns its keep: predicting fine-tuning gain

The diagnostic's whole reason to exist is that it answers a question *in advance*: **is this
failure worth fine-tuning for?** Section 7 of `results-readout-3b.md` claimed it does, on four
synthetic families with exact rank agreement — p=1/24=0.042 under a random-ordering null, which
is an anecdote with a p-value attached. This document tests it properly.

All numbers are canonical (`analysis/tables/canon.json`, `python3 run/canon.py`): probe at the
final layer, model and probe on the same held-out split, fine-tuning trained on the probe's own
training items and evaluated on the probe's own held-out items with the same prompt builder and
the same answer parser.

---

## The design

Sixteen (model, family) pairs: four model configurations (Qwen2.5-VL-3B, Qwen2.5-VL-7B,
InternVL3-2B, SmolVLM) × four real-image families (chart, counting, spatial, and the 3 px glyph
control). One LoRA run per pair, r=16, 165 labels per family, best epoch on the held-out split.

**The quantity being predicted is `gain`, not `accuracy`.** Raw accuracy ordering is largely a
ranking of chance levels and family difficulty — 4.8% chance on chart against 50% on spatial — so
a correlation on accuracies would be a correlation on "which family is easy". The reported test is
therefore **family-demeaned**: within each family, subtract that family's mean from both the
predictor and the target. What survives is the across-model variation, which is the variation the
instrument claims to see.

---

## The result

| predictor | cost | raw ρ | p | demeaned ρ | p |
|---|---|---|---|---|---|
| **probe gain (probe − base)** | one probe fit | +0.662 | 0.005 | **+0.641** | **0.007** |
| probe accuracy | one probe fit | +0.678 | 0.004 | +0.313 | 0.238 |
| base accuracy | free | +0.267 | 0.317 | −0.306 | 0.249 |
| headroom (1 − base) | free | −0.267 | 0.317 | +0.306 | 0.249 |
| base − chance | free | +0.613 | 0.012 | −0.306 | 0.249 |

**Probe gain is the only predictor that survives the control, and it beats every free
alternative.** That comparison is the argument for the instrument: if base accuracy predicted
fine-tuning gain as well, nobody would need to capture hidden states. It does not — and the raw
column shows exactly how a reader could be fooled, since `base − chance` looks significant at
ρ=+0.613 purely because it ranks families by difficulty, and demeaning removes all of it.

**The lower bound is the usable direction.** Probe accuracy ≤ fine-tuned accuracy in **12/12**
informative pairs (12/16 including the glyph control, where fine-tuning *cannot* help and the
probe correctly says so). A logistic regression fitted in seconds gives a floor on what a
30-minute fine-tune will reach, before the GPU-hour is spent.

### The 16 pairs

| model | family | base | probe | fine-tuned | probe gain | FT gain |
|---|---|---|---|---|---|---|
| InternVL3-2B | chart | 60.0% | 94.7% | 96.0% | +34.7 | +36.0 |
| Qwen2.5-VL-3B | chart | 81.3% | 94.7% | 94.7% | +13.3 | +13.3 |
| Qwen2.5-VL-7B | chart | 76.0% | 94.7% | 98.7% | +18.7 | +22.7 |
| SmolVLM | chart | 50.7% | 36.0% | 69.3% | -14.7 | +18.7 |
| InternVL3-2B | counting | 50.7% | 57.3% | 72.0% | +6.7 | +21.3 |
| Qwen2.5-VL-3B | counting | 44.0% | 57.3% | 69.3% | +13.3 | +25.3 |
| Qwen2.5-VL-7B | counting | 46.7% | 65.3% | 70.7% | +18.7 | +24.0 |
| SmolVLM | counting | 40.0% | 53.3% | 58.7% | +13.3 | +18.7 |
| InternVL3-2B | glyph (control) | 8.0% | 4.0% | 2.7% | -4.0 | -5.3 |
| Qwen2.5-VL-3B | glyph (control) | 6.7% | 6.7% | 4.0% | +0.0 | -2.7 |
| Qwen2.5-VL-7B | glyph (control) | 5.3% | 6.7% | 5.3% | +1.3 | +0.0 |
| SmolVLM | glyph (control) | 8.0% | 12.0% | 4.0% | +4.0 | -4.0 |
| InternVL3-2B | spatial | 54.7% | 50.7% | 69.3% | -4.0 | +14.7 |
| Qwen2.5-VL-3B | spatial | 69.3% | 60.0% | 85.3% | -9.3 | +16.0 |
| Qwen2.5-VL-7B | spatial | 60.0% | 61.3% | 82.7% | +1.3 | +22.7 |
| SmolVLM | spatial | 66.7% | 54.7% | 58.7% | -12.0 | -8.0 |

---

## Three rows that are limits, not footnotes

**The diagnostic has a false-negative mode.** SmolVLM chart: the probe reads 36.0% against a
50.7% base — the instrument says "the information is not there" — and fine-tuning gained
+18.7 pp anyway. Linear decodability from the final layer is *sufficient* evidence of presence,
not necessary: fine-tuning can restructure states a frozen linear probe cannot read. So the
instrument's usable direction is the lower bound (12/12), not its converse, and the paper says so.

**Fine-tuning can make things worse.** SmolVLM spatial went 66.7% → 58.7%. The probe reads 54.7%
there, below base, so probe and fine-tune agree there is nothing to recover — but nothing in the
probe's output predicted a *regression*. The instrument flags the family and not the direction.

**The negative control is a true floor.** At 3 px, 495 labelled items of LoRA move the glyph
family not at all (6.7% → 4.0%, 8.0% → 2.7%, 5.3% → 5.3%, 12.0% → 4.0%) -- every one at or
below chance before and after. That is stronger than the probe
alone could establish: the attribute is not merely linearly undecodable, it is not there, and
gradient descent with labels confirms it.

---

## Robustness

**Choice of probe estimator.** With the selected-layer estimator instead of the canonical
final-layer one: demeaned gain ρ=+0.647 (p=0.0067) against +0.641 (p=0.0074), and the lower bound
holds 11/12 instead of 12/12. The claim does not depend on the choice, and the canonical estimator
is the more conservative one (it never benefits from best-of-L selection, which the 3 px control
prices at ~6 pp).

**Intervention consistency.** Twelve of the sixteen pairs adapt both stacks; the Qwen-7B rows
adapt the language stack only, because those are the runs that exist. Recomputing with a
consistent both-stack intervention throughout gives ρ=+0.519 (p=0.039) — still significant, at a
lower coefficient, and with the lower bound then holding 15/16 instead of 12/16. The conclusion is
unchanged in both directions, which is what P0's redundancy finding predicts, since `lang` and
`both` differ by 1–4 pp on these families. An earlier note describing all sixteen pairs as
language-only was wrong; only the 7B has `_lang` runs.

**Epoch selection.** The fine-tuned column takes each run's best epoch on the held-out split,
per family. That is oracle selection, mildly optimistic for the fine-tuning side, so it biases
*against* the probe's lower-bound claim rather than for it.

**n and seeds.** Sixteen pairs, single seed. This is the claim most in need of more pairs, and the
cheapest way to get them is the synthetic families for the two new models (~6 GPU-hours), which
would take n from 16 to 28.

---

## Reproducing

```
cd vlm-locus
python3 run/canon.py            # every number above, plus the manuscript's LaTeX tables
python3 run/p3_lora.py          # the raw per-pair assembly this was built from
```
