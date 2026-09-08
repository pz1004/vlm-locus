# P3 — model breadth: 32 (model, family) pairs

Four model configurations across two axes, on the identical real-image datasets. Every row is
the same protocol: capture at every layer, probe fitted on the train split with the layer chosen
on a separate split, accuracy on the held-out third, plus the exact counterfactual control.

| config | axis | vision encoder | layers x dim |
|---|---|---|---|
| Qwen2.5-VL-3B (bf16) | reference | Qwen ViT | 37 x 2048 |
| Qwen2.5-VL-3B (nf4) | precision control | Qwen ViT | 37 x 2048 |
| Qwen2.5-VL-7B (nf4) | **scale** | Qwen ViT | 29 x 3584 |
| InternVL3-2B (bf16) | **architecture** | InternViT + pixel shuffle | 29 x 1536 |
| SmolVLM-Instruct (bf16) | reference | SigLIP | 25 x 2048 |

7B needs 16.6 GB in bf16 on a 16 GB card, so it runs in nf4 -- and the 3B is re-run in nf4 too,
so the scale comparison changes scale alone instead of scale and precision together.

## Real charts: the information is model-invariant, the readout is not

All numbers on the probe's own held-out split, for model and probe alike.

| config | model | probe | gap | probe follow | verdict |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B (bf16) | **81.3%** | 94.7% | +13.3 | 85% | READOUT |
| Qwen2.5-VL-3B (nf4) | 77.3% | 96.0% | +18.7 | 82% | READOUT |
| Qwen2.5-VL-7B (nf4) | 76.0% | 94.7% | +18.7 | 81% | READOUT |
| InternVL3-2B | 60.0% | 94.7% | +34.7 | 86% | READOUT |
| SmolVLM-Instruct | 50.7% | 36.0% | **-14.7** | 23% | no gap |

**The probe reads real chart values at 94.7-97.3% in four models spanning three architectures
and three hidden sizes, and the four probes agree with each other on 95-99% of individual test
items.** What varies across those four is not the information -- it is how much of it the model
emits, from 61.3% to 88.0%. The gap is largest for the weakest reader and smallest for the best.

SmolVLM is the exception and a useful one: its probe (36.0%) is *below* its own output (50.7%)
with a 23% follow rate. The value is not linearly present in its final layer at all. So the
claim is not "chart values are always decodable"; it is that they were decodable in four of the
five models tested, and the probe plus counterfactual is what tells you which case you are in.

**Scale does not close the gap.** 3B and 7B, precision held constant, give an identical +18.7 pp
gap with the same follow rate (82% vs 81%). Whatever the readout failure is, it does not go away
by making the model 2.3x larger.

**Quantisation widens it, and asymmetrically.** bf16 -> nf4 on the same 3B costs 4.0 pp of model
accuracy (81.3 -> 77.3) while the probe is unchanged (94.7 -> 96.0). The value stays as readable
in the quantised latent as in the unquantised one; what nf4 damages is the model's ability to
get it out. That is a readout cost, not a representation cost, and it is measurable only because
the probe separates the two.

## The other three families: no readout gap in any model

| family | models with gap > artefact | probe follow across models |
|---|---|---|
| counting (real) | 5/5 show +11 to +20 pp | **10-26%** -- all fail |
| spatial (real) | 0/5 | 14-63% |
| glyph 3 px (control) | 0/5 | 0-25% |

Counting is the case the counterfactual control exists for. Every model shows a double-digit
apparent gap, and every one has a follow rate under 30%: the probes read a correlate of the
count, not the count. Judged on probe accuracy alone all five would have been reported as
readout gaps. Judged with the control, none is.

## Threshold discipline

A pair counts as a readout gap only if the final-layer gap clears the artefact floor **and** the
probe follows the counterfactual above 50%. Both halves are measured, not chosen:

* The glyph control at 3 px has the attribute provably absent (model 4.3-6.0%, chance 5.6%), so
  any apparent gap there is pure artefact. Best-of-37-layers yields ~8.5 pp; the final layer
  yields ~2.5 pp. Judging a final-layer gap against the peak-derived number -- which an earlier
  version of `run/p3.py` did -- sets the bar three times too high.
* The follow threshold is what separates counting (10-26%) from chart (81-86%).

## Two measurement errors found and fixed here

**Truncation.** `score.py` generated 12 tokens. Qwen2.5-VL-7B prefixes its answers, so on real
charts the number never appeared and 277/300 items parsed to None -- scored as wrong, producing a
fake 0.0% model accuracy and a fake +94.7 pp gap. Budget raised, `--max-new` now explicit.

**The rounding instruction.** The real-chart question asks for the nearest multiple of 5;
`parse_in_space` accepts only exact members of that space. The model answers 73 for a bar worth
75, 29 for 30, 19 for 20 -- reading the chart and declining to round -- and all of it scored
wrong. The probe emits a class from the answer space *by construction* and cannot make this
mistake, so the comparison was rigged: a probe restricted to legal answers against a model
penalised for illegal ones. Scoring the chart family on the value read (`run/fix_chart_scoring.py`)
raises Qwen-3B from 65.0% to 88.0% over all items.

**Mismatched item sets.** `layers.py` then averaged model accuracy over every item while the probe
was scored on the held-out quarter -- two different samples reported as one comparison. On real
charts the base reads 88.0% over 300 items and 81.3% over the 75 test items, so the gap was
*understated* by 6.7 pp. It now scores the model on the probe's own split and reports `model_all`
separately, so the two can never be silently interchanged. Net of all three fixes the Qwen-3B
real-chart gap is **+13.3 pp**, not the +29.0 first reported nor the +6.7 of the intermediate
correction.

Both errors ran in the same direction -- understating the model, inflating the gap -- and neither
touched the probe, which reads hidden states rather than text. The synthetic families lose under
3% to truncation and have no rounding instruction, so they are unaffected.

## The prediction claim: done, and it is the strongest reason to build the instrument

See `analysis/results-p4-prediction.md`. Sixteen (model, family) pairs, one LoRA each on the
probe's own training items and evaluated on the probe's own held-out items. Probe gain forecasts
fine-tuning gain at rho=+0.641 (p=0.0074, family-demeaned) and lower-bounds the fine-tuned
accuracy in 12/12 informative pairs -- while base accuracy, headroom and the above-chance margin,
all free to compute, carry no signal once family difficulty is removed.

## Canonical numbers

Every cell above is regenerated by `python3 run/canon.py`, which writes
`analysis/tables/canon.json` and the manuscript's LaTeX tables from one probe protocol
(`layers.py`, final layer). `--audit` prints the disagreement against the selected-layer
estimator: 9 of 28 cells differ by >=5 pp, none of them a headline chart cell.
