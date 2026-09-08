# P2 — the real-image replication

COCO val2017 photographs, three families, 900 items + 900 counterfactuals, the identical
protocol to the synthetic study: same capture, same probe, same three-way split, same G1
criterion, same counterfactual control. Levels chosen by the same rule (`runs/levels_real_3b.json`).

## The dataset

Exactness is the whole point, and each family gets its own pixel-identity guard because a
photograph has no spec to re-render from. `gen/verify_real.py`, all at 300/300:

| family | attribute | counterfactual | guard |
|---|---|---|---|
| spatial | left/right from COCO box centres | horizontal flip | `flip(cf)` bit-identical to the original |
| counting | COCO instance count | paste one instance through its segmentation polygon | pixels outside the silhouette identical |
| glyph | numeral at a swept font size | redraw the numeral | pixels outside the glyph box identical |

Also checked: 900 distinct images, 600/600 labels re-derived from `instances_val2017.json`
independently of the generator, every counterfactual changes the answer, no counterfactual
changes the question, and both spatial referents name categories appearing exactly once.

Two design corrections made before the run, both caught by the checks rather than by inspection:

* **Counting was 52% majority class** against a 20% chance level, so "always answer 2" would have
  outscored chance and the model's 51.7% would have been uninterpretable. COCO's count
  distribution is steeply skewed; the generator now stratifies across the range (majority 26%).
* **The negative control was not negative.** A 7 px numeral is below the 14 px ViT patch, but
  Qwen2.5-VL's dynamic resolution still reads it at 68.3%. Extending downward gives a
  dose-response curve instead of an asserted-unreadable level:

  | font px | 28 | 18 | 11 | 7 | 5 | 4 | 3 | 2 |
  |---|---|---|---|---|---|---|---|---|
  | model | 100% | 98.3% | 93.3% | 68.3% | 15.0% | 3.3% | 1.7% | 1.7% |

  Chance is 5.6%, so below 5 px the information is genuinely gone. The main set uses 3 px.

## The result: the readout gap does not replicate

| family | chance | model | probe @ peak | probe @ final | gap @ final | G1 |
|---|---|---|---|---|---|---|
| counting | 20.0% | 44.0% | 61.3% (L29) | 57.3% | **+13.3 pp** | pass |
| spatial | 50.0% | 69.3% | 66.7% (L35) | 60.0% | **-9.3 pp** | pass |
| glyph (3 px) | 5.6% | 6.7% | 13.3% (L7) | 6.7% | +0.0 pp | fail |

*(Canonical protocol: model and probe both on the probe's own held-out split, probe at the final
layer. `analysis/tables/canon.json` is authoritative for every cell; regenerate with
`python3 run/canon.py`. Neither the counting nor the spatial conclusion changes -- counting's
follow rate is 25% and spatial's probe is below the model -- but note that both now PASS the
presence test G1, which is exactly why G1 cannot be the criterion on its own.)*

Counterfactual control, 225 paired items:

| family | probe acc0 | probe follow | model acc0 | model follow |
|---|---|---|---|---|
| counting | 53% | **25%** | 44% | 21% |
| spatial | 65% | **63%** | 69% | 60% |
| glyph | 9% | **0%** | 7% | 0% |

Read together these are unambiguous:

* **spatial** — the probe is *below* the model (65% vs 69%) and follows the edit at the same rate
  (63% vs 60%). There is nothing in the state the model is failing to use. Synthetic spatial had
  a +19 pp gap and a 97% probe follow rate.
* **counting** — +10.6 pp at the final layer, but a 25% follow rate. A probe that reads the count
  should follow a +1 edit; at 25% it is largely reading a correlate. Synthetic counting was
  +15 pp at 57% follow, so even the direction that survives is weaker.
* **glyph** — the designed negative control behaves exactly as designed on real photographs:
  0% follow for both probe and model, at 3 px where the information is provably absent.

**The negative control also prices the layer-selection artefact.** At 3 px, with the attribute
demonstrably not in the image, taking the best of 37 layers still yields a 13.3% probe against
5.6% chance -- roughly **+8 pp of apparent gap from selection alone**. Counting's peak-layer
+14.6 pp has to be read against that; its final-layer +10.6 pp is the honest figure. No previous
part of this study could quantify that, because no synthetic family had an attribute that was
provably absent while the images stayed otherwise realistic.

## A serialisation bug found by the contradiction

The first counterfactual run reported the spatial probe at 49% accuracy (chance 50%) with a 0%
follow rate, while `fit_probes.py` had scored the same probe at 65.3% held-out on the same items.
Cause: sklearn gives a *binary* LogisticRegression one coefficient row and decides on the sign,
but all ten consumer sites reconstruct the probe as `argmax(coef @ z + intercept)`, and argmax
over a length-1 vector is always 0 -- the stored probe predicted a constant.

Real spatial is the first two-class family in the study; every synthetic family has at least four
classes, and all saved synthetic probes were checked to have one coefficient row per class, so
**no previously reported result is affected**. Fixed at save time (symmetric two-row form, which
reproduces sklearn's `predict_proba` exactly) so all ten consumers are correct unchanged, plus a
permanent guard in `fit_probes.py` that re-derives predictions from the serialised probe and
refuses to save when they disagree with sklearn's by more than two points.

## The chart family: the gap replicates, and the diagnostic picks it out

Real charts came later than the COCO families and they overturn the conclusion above.

Source: ChartQA (Statista / Pew / OWID) with the underlying data tables. 300 items, one per
chart, answers stratified across 21 values (most common 6% against 4.8% chance; 18% naturally).
Admission required three things, each of which rejects a way the task could have been the wrong
task: the detected bar heights must be an affine function of the table values (rejects 3-D,
stacked, log and broken axes without naming them), the table must be genuinely two-column
(rejects multi-series charts, where detection finds one series and the other keeps its labels),
and the printed data labels must be strippable.

**That last one nearly invalidated the experiment.** ChartQA charts print their values above the
bars, so "read the bar" is answerable by OCR -- and raising a bar destroys its label, so the
original would have been answerable by reading text while the counterfactual was not. The follow
rate would have measured a task switch. Only 27 of 1,053 charts lack printed labels, so instead
the labels are erased from *every* bar before the counterfactual is derived: the task becomes the
synthetic family's task, no bar is distinguishable from its neighbours, and the pair stays exact.
No automated check caught this -- rendering the pairs and looking at them did.

| | model | probe @ peak | probe @ final | gap | follow | G1 |
|---|---|---|---|---|---|---|
| chart (real) | 81.3% | 96.0% (L33) | **94.7%** | **+13.3 pp** | **85%** | **PASS** |

*(Corrected. The first pass read +29.0 pp from a model accuracy of 65.7%, which was wrong twice
over: the strict parser scored "73" for a bar worth 75 as an error, and model accuracy was
averaged over all 300 items while the probe was scored on the 75 held-out ones. Fixing the parser
raises the model to 88.0% on all items; restricting to the test split -- the only comparison that
is like-for-like -- gives 81.3%. Both fixes are in `run/fix_chart_scoring.py` and `run/layers.py`.)*

G1 passes decisively: selectivity 0.907, blindfold 6.7%, shuffled 4.0%, 99% of items with a
positive margin. The layer curve has the same late-computation signature as the synthetic
family -- 7% at L0, 28% at L27, then **93% at L30** and 95-96% thereafter.

**Where the gap lives.** Splitting the held-out items by value shows the probe is uniformly
strong while the model fails on short bars:

| value band | n | model | probe | gap |
|---|---|---|---|---|
| 0-24 | 18 | 61.1% | 88.9% | **+27.8 pp** |
| 25-49 | 20 | 100.0% | 100.0% | +0.0 pp |
| 50-74 | 19 | 84.2% | 94.7% | +10.5 pp |
| 75-100 | 18 | 77.8% | 94.4% | +16.7 pp |

**Corrected, and the claim is weaker than first reported.** The original version of this table
read 28/70/63/78 for the model, from the strict parser that scored "73" for a bar worth 75 as an
error -- a penalty that falls hardest on small values, where a two-unit miss is proportionally
large. Rescored on the value read (`run/fix_chart_scoring.py`) the model does *not* collapse on
short bars: 61.1%, not 28%. What survives is that the largest gap is still in the lowest band
(+27.8 pp) and the probe is near-flat across the range (88.9-100%). What does not survive is
monotonicity -- the 25-49 band is 100% for both -- so with 18-20 items per band this is a
one-band observation, not a trend. The earlier claim that the 0-24 gap was "within noise of the
synthetic family's +64 pp" was an artefact of the parser and is withdrawn.

**Limitation of the additive edit.** Bars are only ever raised, so counterfactual values are
always higher than originals, and the model is much better at high values (85% at 80-99 vs 39%
at 0-19). The model's acc1 (87%) therefore exceeds its acc0 (59%) and its 86% follow rate is
flattered by the edit moving items into an easier regime. The probe's acc1 (83%) is *lower* than
its acc0 (95%), so the probe's 85% follow is not inflated the same way -- if anything the edit
makes the probe's job harder. The comparison is conservative in the probe's favour, but the two
follow rates are not measured under matched difficulty and should not be compared directly.

## What this means, and what it does not

The full picture across all four real families:

| family | source | model | probe @ final | gap | probe follow | G1 |
|---|---|---|---|---|---|---|
| **chart** | ChartQA | 81.3% | **94.7%** | **+13.3 pp** | **85%** | **PASS** |
| counting | COCO | 44.0% | 57.3% | +13.3 pp | 25% | fail |
| spatial | COCO | 69.3% | 60.0% | -9.3 pp | 63% | fail |
| glyph (3 px) | designed control | 6.7% | 6.7% | +0.0 pp | 0% | fail |

**The readout gap is real on real images, and it is family-specific rather than
synthetic-specific.** Chart is the family where the synthetic effect was largest (+64 pp, 89%
follow) and it is the one that replicates (+13.3 pp, 85% follow, G1 PASS). Spatial and counting
were the weaker synthetic families and they do not replicate at all.

This is the same shape as the cross-model result in §10 of `results-readout-3b.md`: the gap is a
property of a (model, task) pair, and neither the benchmark, the error rate, nor the size of the
probe gain predicts which pair is in which regime. Only the probe plus its counterfactual does.

Two things this does **not** show:

* It is not a claim that the model cannot read charts. It reads real charts at 81.3% on the
  held-out split (88.0% over all 300 items), far better than the 34.2% it manages on the synthetic
  family, where the failure was a language prior (answering multiples of ten when the values were
  not). The real gap is narrower and is largest on short bars, where the model scores 61.1% and
  the probe 88.9%.
* Two of four real families are negative, and one of those (spatial) was strongly positive in
  synthetic (+19 pp, 97% follow). Whatever drives the synthetic spatial gap does not survive the
  move to photographs. That is a limit on the generality of the synthetic result and it should be
  reported as one.
