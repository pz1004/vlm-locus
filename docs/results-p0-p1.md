# P0 / P1 results — the day-1 gate

Run against `runs/states_3b.npz` and the probe's own splits, 298 held-out items, one parser.
Decision rules are the ones fixed in `research-plan-v3.md` *before* the runs, not after.

---

## P1 — H-read: the head-tied readout does not close the low-budget gap

**Question.** LoRA's 8x label-efficiency advantage was attributed to one structural fact: it
inherits an output head that already knows what "45" is, while B-read learns an 18-way output
vocabulary from one or two examples per class. Give the readout the same head and the advantage
should transfer.

**Method.** `run/hread.py`. Freeze the final hidden state `h` and the unembedding `W_U`; learn
only a low-rank correction inside the model's own output space,
`logits = W_U(h + (alpha/r) h A B)`, argmax over the answer set. Two tyings: `pos` (one map per
answer position, against that position's candidate tokens) and `proto` (one map, against
per-answer prototypes `e_a = mean_t W_U[t_a]`). Rank chosen on the calibration split; a matched
B-read is fitted on the identical items in the same script so every row carries its own control.

**Result: no.** Single seed, so read small differences as noise.

| labels/family | 10 | 20 | 40 | 80 | 120 | 165 |
|---|---|---|---|---|---|---|
| H-read (`pos`) | 51.7 | 56.0 | 64.8 | 71.8 | 73.5 | 73.5 |
| B-read | 49.7 | 62.8 | 63.8 | 74.2 | 75.2 | **76.2** |
| LoRA | — | **75.2** | **81.9** | — | — | **83.2** |

H-read is indistinguishable from B-read across the whole curve and far below LoRA everywhere.
**Track A is dead at the n=20 gate (56.0% against a 68% threshold).**

**The mechanism test, which is why this is a finding and not a tuning failure.** At n=20, 23 of
chart's 73 test items have a gold class that never appears in training. B-read scores **0.0%**
on them -- a free |A|-way classifier structurally cannot emit a class it never saw. H-read
scores **8.7%**, against chart's 5.9% chance. The inherited head buys nothing exactly where it
was supposed to buy everything.

That is not because the head lacks geometry. It has plenty:

| geometry | Spearman rho vs numeric closeness |
|---|---|
| single digit tokens in `W_U` | **+0.879** (p=2e-15) |
| answer prototypes (mean-pooled) | +0.331 (p=3e-05) |
| residual-stream class centroids | +0.326 (p=1e-04) |

The head is strongly magnitude-ordered at the digit level; mean-pooling two digits to make a
prototype is what destroys the ordering. But `pos`, which avoids pooling and factors chart into
a 9-way tens digit and a 2-way ones digit, does no better. So the failure is not the prototype
construction either.

**What it means.** The frozen state and the frozen head cannot be aligned by a low-rank bridge
fitted on 20 examples. LoRA's advantage is not that it *inherits* the head -- it is that it
re-tunes 252 modules across 36 layers so the state comes to *align* with the head. That is a
different kind of change from any readout, and it is the honest reason no readout reaches it.

**What survives.** H-read is now the **sixth** readout architecture to land in the same
72.8-76.2% band, and it is the one that inherits the model's own output head -- the obvious
objection to "the frozen states have a ceiling of about 76%" is now tested and answered.

It also has one property no other readout has, worth a line in the paper: with `B = 0` it *is*
the base model, so it starts at 51.3% where B-read starts at chance, and the `init` column is a
free reconstruction check -- single-token families reproduce the model's own accuracy exactly
(spatial init 76.0% against the model's 76.0%).

---

## P0 — the +7 pp does not decompose, and that is the result

**Question.** The 83.2% LoRA adapts both stacks: `target_modules` was given as bare suffixes, so
PEFT matched the language model *and* the vision tower's MLPs (192 of 696 adapter tensors). So
"fine-tuning adds representation beyond the frozen-state ceiling" was never measured against
"fine-tuning learns a deeper readout of an unchanged representation".

**Method.** `run/lora.py --modules {lang,vis,both}`. `lang` and `vis` are the exact complement
halves of the reference run -- same data, same 656 items, same hyperparameters, same oracle epoch
rule, same parser. The run aborts if the regex lands on the wrong stack, so an empty half cannot
be mistaken for a result. `run/p0.py` does the paired tests on 298 shared items.

| config | chart | counting | spatial | tracking | ALL |
|---|---|---|---|---|---|
| base | 34.2% | 61.3% | 76.0% | 28.0% | 50.0% |
| frozen linear readout (no training) | 98.6% | 76.0% | 94.7% | 36.0% | 76.2% |
| **LoRA `lang`** (vision tower frozen) | 93.2% | 89.3% | 97.3% | 38.7% | **79.5%** |
| **LoRA `vis`** (language model frozen) | 100.0% | 88.0% | 98.7% | 42.7% | **82.2%** |
| LoRA `both` (the reference) | 100.0% | 88.0% | 97.3% | 48.0% | 83.2% |

**The stacks are redundant, not additive.**

| comparison | McNemar |
|---|---|
| `vis` vs `both` | **p = 0.72** -- indistinguishable |
| `lang` vs `both` | p = 0.052 |
| `lang` vs `vis` | p = 0.20 |
| `lang` vs base | **p = 2.5e-17** |
| `vis` vs base | **p = 1.8e-20** |

Per family, adapting both stacks is indistinguishable from adapting the better single one
(chart p=1, counting p=1, spatial p=1, tracking p=0.42). Each stack alone recovers 89-97% of the
total gain over base. So the planned "+x pp language, +y pp vision" split would misdescribe the
data, and `run/p0.py` deliberately does not print one.

**What this establishes, and it is stronger than the probe result.** `lang` holds the entire
vision tower frozen, so the image tokens entering the language model are *bit-identical* to the
base model's. On that unchanged visual representation the model goes **50.0% -> 79.5%**
(p = 2.5e-17), recovering 89% of everything fine-tuning recovers.

That is a presence claim made by the model itself rather than by a probe. A linear probe reading
98.6% off the final layer can always be answered with "your probe is doing work the model cannot
do". This cannot: if the information were absent from the frozen visual representation, no amount
of language-side adaptation could recover it, because there would be nothing to recover. The
logic is one-directional and that is the point -- `vis` also reaching 82.2% says the interface
can be repaired from either side, which is what an alignment failure looks like and what a
missing-information failure cannot look like.

One detail worth keeping: on chart, the frozen linear probe (98.6%) *beats* language-only
fine-tuning (93.2%). Thirty minutes of gradient descent on 252 modules extracts less from the
final hidden state than a logistic regression fitted on it in seconds.

**Against the pre-registered rule.** `lang` = 79.5% landed in the middle band (76-81% = "mixed,
report the split"), but the rule assumed the two stacks partition the gain. They do not, so the
finding replaces the rule's conclusion rather than selecting one of its branches.
