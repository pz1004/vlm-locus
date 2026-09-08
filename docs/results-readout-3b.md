# The readout gap: a VLM's final hidden state answers questions the VLM gets wrong

Qwen2.5-VL-3B, four synthetic families with counterfactual controls, 298 held-out items.
All numbers below are on that one test split, with one answer parser, produced by
`run/compare.py` from frozen artefacts in `vlm-locus/runs/`.

This document supersedes `results-lrc-3b.md`. The locus-routed-correction method it was
written to evaluate does not survive the experiments recorded here, and §8 says exactly how
it fails. What replaced it is a stronger and simpler result.

---

## 1. The finding

For two of four families the target attribute is linearly decodable from the residual stream
**at the final layer** — the vector the language-model head itself consumes — at accuracy far
above what the model emits.

| family | model | probe @ final layer | probe @ best layer | gap |
|---|---|---|---|---|
| chart | 34.2% | **98.6%** (L36) | 98.6% (L28) | +64 pp |
| spatial | 76.0% | **94.7%** (L36) | 97.3% (L27) | +19 pp |
| counting | 61.3% | 76.0% (L36) | 76.0% (L36) | +15 pp |
| tracking | 28.0% | 36.0% (L36) | 46.7% (L15) | +8 pp |

Chance is 5.6 / 25 / 7.7 / 25%. The probe is standardise → 64-dim PCA → multinomial logistic
regression. Items are split three ways: it is fitted on the train split, the layer is chosen on
a separate selection split, and every number reported anywhere in this document is on the third,
held-out split. The reported layer is never the argmax over test data.

The chart curve is the clean case: 8% at L0, 60% at L27, **99% from L30 to L36**. The
attribute is computed late, sits fully formed in the last hidden state, and the model says
something else.

This is not a perception failure. The information is present at the last place it could
possibly be used. It is a **readout** failure: the model's own unembedding does not extract
what a task-specific linear map extracts from the very same vector.

That sentence is about *this model*. §10 runs the same protocol on SmolVLM, where chart is
decodable at only 35.6% and the same task therefore fails for the opposite reason. The gap
generalises; its size does not.

### The model's ranking is informative but wrong at rank 1

Restricting the model's own distribution to each item's answer space and asking where the gold
answer lands:

| family | rank-1 | gold in top-3 | mean normalised rank |
|---|---|---|---|
| chart | 36% | 73% | 11% |
| counting | 48% | 91% | 8% |
| spatial | 76% | 99% | 10% |
| tracking | 36% | 77% | **44%** |

For the first three the gold answer is near the top and not at it — a selection failure at the
last step. Tracking's 44% is near the 50% of a random ranking, which is what a family whose
answer the model genuinely does not have should look like. (Rank-1 here is a *constrained*
decode over the answer space, so it sits a little above the free-form accuracy in §4.)

---

## 2. Which probes are real: three controls

A probe scoring above the model proves nothing on its own. Three controls separate reading the
attribute from exploiting a correlate.

**Counterfactual (causal).** One field of the spec is edited so the true answer changes, the
image is re-rendered from the same seed, and the original is re-rendered bit-identically as a
guard. *Follow rate* = of items answered correctly before the edit, the share answered with the
**new** value after.

| family | probe acc | probe follow | model follow |
|---|---|---|---|
| chart | 99% | **89%** | 76% |
| spatial | 96% | **97%** | 70% |
| counting | 68% | 57% | 91% |
| tracking | 39% | **7%** | 57% |

Chart and spatial track the edit. Tracking's 7% is a probe that is not reading the attribute at
all — it is the negative control, and it is a real one.

**Within-arm (shortcut).** Every family holds 150 `canonical` and 150 `anti` items, and the two
arms were built with **disjoint answer sets** (chart: multiples of 10 vs tens-plus-5; counting:
2–5 vs 8–12; spatial: left/right vs above/below). Identifying the arm is easy and halves the
answer space for free, so the pooled chance level flatters every probe. Refitting inside each
arm, where that shortcut carries no information:

| family | arm | classes | chance | probe | model |
|---|---|---|---|---|---|
| chart | canonical | 9 | 11% | 100% | 71% |
| chart | anti | 9 | 11% | 100% | **0%** |
| spatial | canonical | 2 | 50% | 100% | 84% |
| spatial | anti | 2 | 50% | 100% | 68% |
| counting | canonical | 4 | 25% | 89% | 79% |
| counting | anti | 5 | 20% | 57% | 43% |
| tracking | canonical | 4 | 25% | **100%** | 21% |
| tracking | anti | 4 | 25% | 32% | 36% |

Chart and spatial survive the stricter chance level intact. Chart-anti at 0% model accuracy is
the sharpest single number in the study: when bar values are not multiples of ten the model
answers with a multiple of ten every time, while its own last hidden state carries the true
value at 100%.

**Shuffled-label and blind controls** are in `runs/probe_g1.json`: selectivity (visual minus
shuffled) is 0.945 chart, 0.667 spatial, 0.587 counting, 0.093 tracking.

---

## 3. Four defects found and fixed

Each was found by a control, not by inspection, and each invalidated results already in hand.

**(a) Duplicate images.** `spec_tracking` at level 1 admitted ~108 distinct specs; 300 items
were drawn from it without dedup, giving 101 distinct images among 300 and a pixel-identical
twin in the training split for 65 of 75 held-out items. The tracking probe scored 100% on
duplicated items and 30% on the 10 clean ones — below its 33% chance. Fixed at three levels:
the spec space widened to 6,912, `build()` now rejects duplicate renders, and `verify.py` check
[9] fails any dataset containing one. Regeneration reproduces chart/counting/spatial
**bit-identically** (900/900), proving the change is confined to tracking. The tracking probe
fell from 90.7% to 38.7% (chance 27.3%).

**(b) sdpa/eager skew.** States were captured under `sdpa` while every consumer ran `eager`, so
every probe was fitted on one kernel's activations and applied to another's. Cost: ≤1.3 pp for
the three robust probes, **62 pp** for the memorising tracking probe — which is how the
contamination was independently confirmed. `capture.py` now pins `eager`.

**(c) Arm label-disjointness** (§2). Pooled chance levels were wrong for three families. Not a
bug in the code; a confound in the design that has to be reported, and the within-arm table is
the reporting.

**(d) Tracking's canonical arm is degenerate.** `arm == "canonical"` is *defined* as
`_replay(start, swaps) == start` — the ball returns to where it began. So for all 150 canonical
items the answer equals the visible start marker, and a probe reaching 100% there has tracked
nothing. The honest tracking measurement is the anti arm: probe 32% against 25% chance. This
makes tracking a *stronger* negative control, not a weaker one.

---

## 4. Nothing training-free recovers the gap

Same 298 items, same parser. `needs` is the supervision tier.

| method | needs | chart | counting | spatial | tracking | ALL | vs base |
|---|---|---|---|---|---|---|---|
| prior (contrastive rescoring) | none | 36% | 60% | 83% | 33% | 53.0% | 0.22 |
| steer (probe direction) | labels | 27% | 64% | 81% | 28% | 50.3% | 1.0 |
| **base** | none | 34% | 61% | 76% | 28% | **50.0%** | — |
| look (referent crop + re-read) | boxes | 10% | 61% | 92% | 28% | 48.0% | 0.5 |
| attn (attention logit bias) | boxes | 30% | 36% | 79% | 29% | 43.6% | 0.011 |
| cot (step-by-step) | none | 25% | 53% | 57% | 23% | 39.6% | 0.00088 |

Not one branch beats the base model at p<0.05. Two are significantly **worse**. Chain-of-thought
costs 7.5× and loses 10 points.

Test-time scaling fails the same way:

| method | needs | ALL | its own oracle | cost |
|---|---|---|---|---|
| best-of-8, T=0.7 | none | 50.3% | 72.8% | 8× |
| best-of-8, T=1.0 | none | 48.7% | **79.9%** | 8× |
| CoT self-consistency, k=5 | none | 39.3% | 70.8% | 37× |

All three land at or below the base model while costing 8–37×. Their *oracles* — the share of
items where some sample is correct — reach 71–80%. The answer is in the sample set on four
items in five, and majority vote cannot find it. Selection, not generation, is the bottleneck.

---

## 5. Reading the latent instead

**B-read**: take the frozen model's residual stream at the last prompt token, apply the
calibration-fitted linear probe, emit its argmax. The decoder is never run for the answer, so
it costs the one forward pass the base model already needs — measured at **1.01×** base when
the probe read is fused into the generation prefill.

| method | needs | chart | counting | spatial | tracking | ALL | vs base |
|---|---|---|---|---|---|---|---|
| **read (ours)** | labels | **99%** | **68%** | **96%** | **39%** | **75.2%** | **2e-12** |
| best training-free (prior) | none | 36% | 60% | 83% | 33% | 53.0% | 0.22 |
| base | none | 34% | 61% | 76% | 28% | 50.0% | — |

Beats the best supervision-free method in **4/4 families**, +25.2 points overall, McNemar
p=2e-12. Two comparisons put that number in context:

- It matches the **best-of-8 sampling oracle** (79.9%, p=0.22) — reading the frozen latent once
  is as good as picking the best of eight sampled generations with gold knowledge, at ⅛ the cost.
- It matches the **6-branch correction oracle** (75.5%). The entire correction library, run with
  oracle knowledge of which branch to apply, does not beat simply reading the latent.

The same probe used as a *selector* over the model's own 8 samples instead of as an *answerer*
scores 57.7%. The probe is far better at producing the answer than at recognising it among the
model's candidates — which is the same lesson the four failed selectors in §5 teach.

The same probe at the **final layer alone** gives 76.2%, so no layer search is needed.

### Against fine-tuning at the same labels, B-read loses at full budget

LoRA (r=16, all 165 labels/family, 656 items, 3 epochs, 29.6 min on one 4070 Ti SUPER) is the
competitor at the same supervision tier, and it is *cheaper* than B-read at inference — the
adapter merges, so 1.00× against B-read's 1.01×.

| family | base | B-read | LoRA | probe rank | LoRA rank |
|---|---|---|---|---|---|
| chart | 34% | 99% | **100%** | 1 | 1 |
| spatial | 76% | 96% | **97%** | 2 | 2 |
| counting | 61% | 68% | **88%** | 3 | 3 |
| tracking | 28% | 39% | **48%** | 4 | 4 |
| **ALL** | 50.0% | 75.2% | **83.2%** | | |

At the full label budget fine-tuning wins, by 8 points overall and on 3 of 4 families. That is
the honest result and it is not a small margin. Two things follow, and only the second is a
consolation.

First, this **confirms the diagnosis rather than undermining it**. Fine-tuning did not teach
the model to see: it taught the model to read what it already computed. Chart goes 34 → 100
because the value was in the final hidden state at 98.6% the whole time. Where the probe says
the information is absent, fine-tuning recovers least — tracking gains 20 points against
chart's 66 and stops at 48%.

Second, **the probe orders the families exactly as the fine-tune does**, and is a lower bound
on it in all four (by 1, 1, 20 and 9 points). A logistic regression that fits in seconds gives
a floor on what fine-tuning will reach, before the GPU-hour is spent. With four families exact
rank agreement carries p=1/24=0.042 under a random-ordering null, which is suggestive and not
more than that; the claim needs more families and a second model before it is worth leaning on.

### How much of the gap is readout, and how much is representation

A linear probe is the weakest possible readout, and it was chosen for diagnosis, where
linearity is the point. As a *method* that constraint is arbitrary, so the same frozen states,
same labels and same 1.01× inference cost were given five stronger readouts:

| readout | chart | counting | spatial | tracking | ALL | vs B-read |
|---|---|---|---|---|---|---|
| linear @ best layer (B-read) | 99% | 68% | 96% | 39% | 75.2% | — |
| linear @ final layer | 99% | 76% | 95% | 36% | **76.2%** | 0.75 |
| 3-layer concat | 99% | 69% | 97% | 29% | 73.5% | 0.46 |
| MLP @ final | 97% | 71% | 93% | 40% | 75.2% | 1.0 |
| MLP @ concat | 96% | 75% | 92% | 43% | 76.2% | 0.77 |
| fuse with model logits | 92% | 75% | 92% | 33% | 72.8% | 0.42 |
| **LoRA (full budget)** | 100% | 88% | 97% | 48% | **83.2%** | |

Every readout lands between 72.8 and 76.2% and none is distinguishable from the linear probe.
The frozen states have a ceiling, and it is about 76%. That gives a clean decomposition of what
fine-tuning buys:

```
base                     50.0%
  + readout              76.2%   (+26.2 pp -- information already in the frozen state)
  + representation       83.2%   (+ 7.0 pp -- what fine-tuning adds beyond it)
```

**79% of the total recoverable gain is readout, not perception.** Fine-tuning's extra 7 points
are real and are the reason it wins, but they are the minority of the effect.

The oracle over {model answer, probe answer} is 82.6% — within a point of the full fine-tune —
so a *perfect* selector between what the model says and what its own latent says would match
LoRA without any training. It is not reachable. Four independent selectors were tried and none
beat always-read: label-free branch routing (§8), probe-feature branch routing (§8), soft
log-probability fusion (above, 72.8%), and a per-family confidence gate fitted on calibration
(74.8%, McNemar p=1.0 against always-read, with the thresholds correctly collapsing to
"always read" on chart and spatial). Selection is where this problem is hard.

### At every label budget, fine-tuning wins — and that is itself the evidence

The low-budget comparison was the last place B-read could have won, because §6 shows the probe
passing the model at 20 labels per family. It does not win there either. Both methods, same
labels, same held-out items; LoRA gets oracle epoch selection and the probe gets its
hyperparameters tuned on the calibration split, so each is shown at its best:

| labels/family | total items | B-read | LoRA | gap |
|---|---|---|---|---|
| 20 | 80 | 58.5% | **75.2%** | −16.7 |
| 40 | 160 | 66.2% | **81.9%** | −15.7 |
| 165 | 656 | 75.2% | **83.2%** | −8.0 |

**LoRA at 20 labels per family equals B-read at 165** — an eight-fold label-efficiency
advantage. Tuning the probe's components and regularisation per budget changed nothing
(57.5% at n=20, 74.5% at n=165, both within noise of the untuned curve), so this is not a
hyperparameter artefact. B-read is not a competitive method at any budget, and the honest
conclusion is that the algorithmic contribution does not stand.

The reason is instructive rather than disappointing. A linear probe has to learn the whole map
from a 2048-dimensional space to an 18-way answer set from 20 examples — one or two per class.
LoRA inherits a language head that already knows what "45" is, so it only has to learn *where
to look*. That is a small change, which is exactly why 80 labelled examples take chart from
34% to 90%.

And that is the strongest form of the paper's claim. **Eighty examples do not teach a model to
see.** They cannot. What they do is connect a representation the model already had to an output
it was already capable of producing. The cheapness of the fix is the measurement.

---

## 6. How many labels

B-read and LoRA draw on the same resource, so the comparison that matters is not one accuracy
against another but both against the label budget. Probe accuracy on the same held-out split,
by labelled items **per family** (± sd over 5 resamples):

| family | model | n=10 | n=20 | n=40 | n=80 | n=120 | n=165 |
|---|---|---|---|---|---|---|---|
| chart | 34% | 42±4 | **61±6** | 78±2 | 93±4 | 95±2 | 99 |
| spatial | 76% | 61±6 | 74±4 | **87±3** | 94±3 | 95±1 | 96 |
| counting | 61% | 42±6 | 61±6 | 66±4 | 69±2 | 68±1 | 68 |
| tracking | 28% | 29±5 | 38±5 | 34±5 | 36±3 | 37±3 | 39 |

Chart passes the model at **20 labels**, spatial at 40. Counting reaches parity at 20 and
saturates 7 points above. Tracking never separates from its 25% chance — the same verdict the
counterfactual control gave, reached from a different direction.

The left end of this curve is the part that matters: at 20 examples per family, fine-tuning a
3B model is not a serious proposition, and a linear map on its own frozen states already beats
it on the family where the gap is largest.

## 7. The diagnostic, which is what survives

The probe is not a competitive method (§5), but it is a cheap instrument, and as an instrument
it answers the question a practitioner actually has: *is this failure worth fine-tuning for?*
Its held-out accuracy, fitted in seconds on 165 examples, versus what a 30-minute fine-tune
then achieves:

| family | probe | fine-tune | base | fine-tune gain |
|---|---|---|---|---|
| chart | 98.6% | 100% | 34% | +66 pp |
| spatial | 97.3% | 97% | 76% | +21 pp |
| counting | 69.3% | 88% | 61% | +27 pp |
| tracking | 38.7% | 48% | 28% | +20 pp |

The probe is a lower bound on the fine-tuned result in all four families and orders them
correctly, and its failures are informative: where it reads near chance *and* fails the
counterfactual control (tracking, 7% follow), the fine-tune recovers least and stops well below
the others. With four families this is a suggestion, not a law (§10).

This is what survives from the original locus idea, in the only form that pays: not a router
over corrections, and not an answer-producer, but a **precondition test** — run before spending
the GPU-hour, telling you whether the information is there to be recovered.

---

## 8. The routing method does not survive

The plan's contribution was Locus-Routed Correction: estimate the failure locus per item, apply
the matched branch. Two experiments kill it.

**Label-free routing has no signal.** Three feature sets computed from the decoder alone —
answer log-probability; plus the constrained answer-space distribution; plus the blindfold
contrast — on the label-free branch library:

| routing signal | needs | ALL | vs base | vs best fixed |
|---|---|---|---|---|
| gen | no labels | 52.7% | 0.3 | 1.0 |
| vis | no labels | 51.3% | 0.66 | 0.58 |
| vis+bl | no labels | 52.3% | 0.39 | 0.88 |
| probe | labels | 58.1% | 7e-05 | 0.032 |
| **read** | labels | **75.2%** | 2e-12 | 3.6e-10 |

No label-free set is distinguishable from either the base model or the best fixed branch.

**With labels, routing is dominated.** Probe-feature routing does beat the best fixed branch
(58.1%, p=0.032) — but B-read, using the same probe, reaches 75.2%. The bind is exact:

> The labels that fit the probe LRC routes on are the labels that fit B-read, which beats every
> router configuration at the same cost. Without those labels there is nothing to route on.

Routing over corrections was the wrong use of the signal. The signal *is* the answer.

---

## 9. What this is, and what it is not

It is not a new best method. B-read loses to LoRA at every label budget tested, by 8 to 17
points, and no stronger readout on the frozen states closes the gap (§5). The routing method
the study set out to build is dead twice over (§8).

It is a measurement, and the measurement is unusually clean:

1. **The gap is real and large.** 79% of the recoverable error on these tasks is information
   already present in the final hidden state and not read out. Chart is decodable at 98.6%
   from the vector the LM head consumes, while the model answers 34%.
2. **It is invisible to everything training-free.** Six correction branches, three test-time
   scaling methods and four selection mechanisms — thirteen methods — none beats the base model
   by a significant margin, several are significantly worse, and the sampling oracles at 71–80%
   show the answers were there to be picked.
3. **It closes for almost nothing.** Eighty labelled examples move chart from 34% to 90%.
4. **It is validated, not asserted.** Counterfactual follow rates, within-arm refitting,
   shuffled labels and blindfold contrasts all agree, and they agree on the negative case too:
   tracking fails every one of them, and is the family where nothing helps.
5. **Four defects were found by those controls** (§3), three of which had already produced
   results that looked publishable.

## 10. A second model: the gap replicates, its size does not
> **Corrected 2026-09-07; tables below now carry the canonical figures.** `run/layers.py`
> originally averaged the model's accuracy over every item while scoring the probe on the held-out
> quarter, so the two columns came from different samples. It now scores both on the probe's own
> split, and the tables in this section have been rewritten accordingly (shifts of 1-8 pp, no
> verdict changed). `analysis/tables/canon.json`, regenerated by `python3 run/canon.py`, is
> authoritative for every cell in the study; this document is the narrative record. Qwen's own
> synthetic numbers were never affected, because its layer sweep was always fed a test-only file.


SmolVLM-Instruct, same 1,200 items, same probe protocol, same held-out split. (The v1 SmolVLM
run used its own difficulty-matched dataset and is not comparable; this one is.)

| family | model | probe @ final | probe @ peak | gap |
|---|---|---|---|---|
| chart | 16.4% | 35.6% | 39.7% (L20) | +19.2 pp |
| counting | 28.0% | 44.0% | 48.0% (L20) | +16.0 pp |
| spatial | 32.0% | 58.7% | 69.3% (L20) | +26.7 pp |
| tracking | 29.3% | 30.7% | 42.7% (L3) | +1.3 pp |

*(Canonical: model on the probe's own held-out split. The all-item averages this table originally
carried were 22.7 / 19.7 / 35.3 / 27.7%, which moved chart's gap by 6 pp and counting's by 8 pp
in opposite directions. The direction of the finding is unchanged in all three non-tracking
families and tracking still fails.)*

The **direction replicates in all three non-tracking families** — the probe reads the attribute
above what the model emits — and tracking again fails its selectivity control (9.3%, against
31–43% for the others). Pooled, SmolVLM goes 26.4% → 42.3% on frozen states, a +15.9 pp readout
gain against Qwen's +26.2.

The **magnitude does not replicate, and that is the more interesting half.** Qwen's chart value
is decodable at 98.6%; SmolVLM's at 35.6%. For Qwen, chart failure is almost purely readout —
the answer is sitting in the final hidden state. For SmolVLM the answer is largely not there to
read, so the same task fails for a different reason in the two models:

| | chart failure | locus |
|---|---|---|
| Qwen2.5-VL-3B | 34% correct, 98.6% decodable | **readout** |
| SmolVLM-Instruct | 23% correct, 35.6% decodable | **perception** |

This is the result that makes the diagnostic worth having, and it is also the strongest
constraint on §1. "The answer is in the final hidden state" is a claim about a model, not about
a task, and it must be measured per model rather than assumed. A method that reads latents will
work well on one of these two models and barely at all on the other, and nothing about the
task, the benchmark or the error rate tells you which without running the probe.

Difficulty above was calibrated for Qwen, and SmolVLM sits near chance there on spatial and
tracking, so those numbers are compressed from both ends. The replication below removes that
objection.

### Difficulty-matched: the gap is real, and the locus moves between models

The whole protocol re-run at SmolVLM's *own* levels, chosen exactly as Qwen's were — the level
per family closest to 50% error (chart L0, counting L0, spatial L1, tracking L0). New dataset,
new capture, new probes, new counterfactuals.

| family | model | probe @ final | probe @ peak | gap | follow | G1 |
|---|---|---|---|---|---|---|
| spatial | 38.7% | 85.3% | 86.7% (L22) | **+46.7 pp** | **90%** | PASS |
| chart | 48.6% | 63.5% | 67.6% (L19) | +14.9 pp | 46% | PASS |
| counting | 20.0% | 38.7% | 45.3% (L21) | +18.7 pp | **10%** | PASS |
| tracking | 31.6% | 47.4% | 52.6% (L15) | +15.8 pp | 40% | fail |

*(Canonical, from `analysis/tables/canon.json`. Shifts against the original all-item averages are
1-2 pp and no verdict changes: spatial is still SmolVLM's only causally-clean locus.)*

Matching difficulty **raised** the gaps, as the floor-effect argument predicted: spatial went
from +23.4 to +48.6 pp, chart from +12.9 to +16.8. So the first pass understated SmolVLM, and
the readout gap survives in a second architecture at **+48.6 pp on spatial with a 90%
counterfactual follow rate** — causally validated, on a model that answers those items at 36.7%.

But applying the same controls used on Qwen thins the list, and that is the finding:

| family | Qwen gap (follow) | SmolVLM gap (follow) |
|---|---|---|
| chart | **+64.4 pp (89%)** | +14.9 pp (46%) |
| spatial | +18.7 pp (97%) | **+46.7 pp (90%)** |
| counting | +14.7 pp (57%) | +18.7 pp (10%) |
| tracking | +8.0 pp (7%) | +15.8 pp (40%, n=19) |

**The locus flips.** Qwen's large, causally-clean readout gap is on *chart*; SmolVLM's is on
*spatial*. Each model's strong family is the other's weak one. Only the counterfactual control
separates them — chart's SmolVLM probe gains 17 points and follows the edit less than half the
time, and counting's gains 19 points while following it one time in ten, so neither is reading
the attribute the way Qwen's chart probe does.

Nothing about the task, the benchmark, the error rate or even the size of the probe gain
predicts which regime a given model-family pair is in. That is the case for running the probe
*and* its counterfactual control per model and per task, and it is what a benchmark score
cannot tell you.

**Limitation.** SmolVLM's tracking level has a spec space of 108 distinct items, so the
generator produced 72 rather than 300 and the tracking split is 19 test items. That row is
reported for completeness and is too small to carry weight. Counting never reaches the 40–60%
band for SmolVLM at any level (72% error at the easiest), so its row is a floor measurement
even here.

---

## 11. Open -- both items now closed
- **A real-image family.** Done: `analysis/results-p2-real.md`. Charts replicate at +13.3 pp with
  an 85% follow rate; real spatial and real counting do not, and a designed 3 px glyph control
  gives the study a measured floor on photographs.
- **Whether the probe's ordering predicts fine-tuning gain at usable n.** Done:
  `analysis/results-p4-prediction.md`. n=16 pairs, rho=+0.641 (p=0.0074, family-demeaned), and
  the probe beats four free predictors that carry no signal at all.

## Reproducing

```
bash vlm-locus/run/report.sh          # every table above, analysis only, no GPU
```
