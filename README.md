# vlm-locus

A calibrated protocol for deciding **why** a vision–language model got a visual question wrong:
because the queried attribute never entered the representation (*perception* failure), or because
it is present in the final hidden state and the output head does not extract it (*readout*
failure). Benchmark accuracy cannot separate the two, and the distinction decides whether
supervised adaptation will help.

This repository is the complete reproducibility artefact: the protocol, the generators, the frozen
run artefacts, and the analysis that turns them into every reported number. **`docs/RESULTS.md`
holds the results in full and is itself generated from the artefacts**, so the repository does not
depend on any other document to be read or checked.

## The protocol in one page

For each (model, family) cell we capture the hidden state at the final prompt token, fit a linear
probe on the **final** layer — the post-final-norm vector the language-model head consumes, which
`run/assert_head.py` verifies rather than assumes — and compare what the probe decodes against
what the model emits on the same held-out items.

A probe beating a model is not by itself evidence of anything, so a cell counts as a **readout
locus** only if all three hold, each against its own control:

| condition | control | why |
|---|---|---|
| the attribute is decodable | the probe clears the 95th percentile of **its own** label-permutation null | a probe-minus-model gap has no meaning until you know how large it gets with the label–representation relation destroyed |
| the probe reads more than the model emits | exact paired test on the same items | an unpaired accuracy comparison confounds difficulty with locus |
| the decoding tracks the attribute | the **lower 95% bound** on the exact-counterfactual follow rate exceeds 50% | a probe can decode a *correlate* and still be useless; edits are pixel-guarded so exactly one field changes |

The null is what makes the negatives readable, and it cannot be a constant: its 95th percentile
runs from 8.6% to 61.3% across our grid as the class count and sample size vary. The protocol is
validated on a family designed to have nothing to read — a numeral rendered below the resolution
at which any model reads it — where the presence test fires at the nominal rate and the composed
verdict never fires.

Two things this buys that an accuracy comparison does not. Counting probes decode above their own
null in **all** cells and survive the counterfactual in **none**: the information is there, and
they are reading a correlate of the count. And the diagnostic's association with fine-tuning gain,
which is real in sample, buys **no detectable out-of-sample accuracy** — so it is reported as an
association and withdrawn as an instrument.

## Reproducing every number

Every number and figure comes from the frozen artefacts in `runs/`. **No GPU, no model download
and no dataset are needed for this path.**

```bash
pip install numpy scipy scikit-learn matplotlib
python3 run/canon.py        # -> out/canon.json, out/tables/*.tex
python3 run/results_md.py   # -> docs/RESULTS.md
python3 run/figs.py         # -> out/figs/*.pdf, *.png
```

`run/canon.py` is the single source of every reported number; nothing is transcribed by hand, and
both it and `run/results_md.py` are byte-stable for a given input, so **regenerating and diffing
is itself a check** that the committed outputs match the artefacts.

Output locations default to `out/` and are overridable with `VLM_LOCUS_JSON`, `VLM_LOCUS_TEX`,
`VLM_LOCUS_FIGS` and `VLM_LOCUS_DOCS`.

## Checking it

Three suites, all runnable from a fresh clone with no GPU. Each exits non-zero on failure.

```bash
python3 run/manifest.py          # what feeds what, and the load-path invariants
python3 run/verify_provenance.py # every number traces to the artefact that produces it
python3 run/verify_protocol.py   # the verdict has the properties it claims
```

`run/manifest.py` does not describe the artefact mapping, it **measures** it: it wraps file access,
runs the producers, and records the files they actually open, so the mapping cannot drift from the
code. It then enforces four invariants — no quarantined artefact in the load path, no
non-canonical lookalike being read, no producer reading a path another producer overwrites, and
agreement between the two places base accuracy is computed. `--json` emits the mapping.

`run/verify_protocol.py` checks the claims a reader would want to hold the protocol to: that no
verdict is decided by a couple of items, that the one prespecified threshold is swept rather than
asserted stable, that every locus survives multiplicity correction across the whole grid, that the
figures plot the calibrated quantities, and that nothing in the repository reads a path outside it.

On a GPU, `run/assert_head.py --model <id> [--load-4bit]` checks the claim the whole protocol rests
on: that the captured vector is the one the unembedding consumes. Run it on the quantised
configurations too — quantisation changes the numerics the assertion measures.

## Re-running the experiments

Needs a CUDA GPU (everything reported was produced on a single 16 GB RTX 4070 Ti SUPER) and the
full dependency set:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python3 gen/generate.py        # synthetic families, rendered from specs
python3 gen/build_real.py      # real families (needs COCO val2017 + ChartQA)
python3 gen/verify_real.py     # the per-family pixel-identity guards
bash    run/pipeline.sh        # capture -> probe -> counterfactual -> score
```

The real families are built on [COCO](https://cocodataset.org) *val2017* and
[ChartQA](https://github.com/vis-nlp/ChartQA), both public downloads. No image from either is
redistributed here, and no COCO annotation content is committed --- `gen/` reads COCO's
annotation files from the user's own download at build time.

Two committed files are *derived from* ChartQA annotations, and it is worth being exact about
them rather than filed under "not redistributed". `runs/chart_index.json` (742 charts) and
`runs/chart_index_nolabel.json` (27) each record, per source chart, its ChartQA identifier, its
category labels, its underlying values, and the bar geometry recovered from the rendered image.
They are committed because the counterfactual renderer and the per-family pixel-identity guards
need them, and because the reproduction path would not work without them.

ChartQA is distributed under **GPL-3.0**, and that is why this repository is too. Whether an
index of identifiers, labels and recovered geometry is a derivative work of the annotations it
was extracted from is a question the author is not equipped to answer; relicensing means it does
not have to be answered before anyone can redistribute this, because either way the whole thing
travels under one licence instead of two that would have to be reconciled first. ChartQA's own
terms still govern that content directly, upstream of anything granted here.

## Layout

| path | contents |
|---|---|
| `run/` | protocol, probes, counterfactuals, adaptation, analysis. `canon.py` emits every reported number |
| `gen/` | dataset generators and the per-family verification guards |
| `runs/` | frozen run artefacts (JSON/JSONL) and serialised probes — the inputs `canon.py` reads |
| `runs/superseded/` | artefacts that resemble a canonical one and are not, with the reason per file. Tracked deliberately: it is the audit trail for `manifest.py`'s first invariant |
| `docs/RESULTS.md` | the results in full, generated from `out/canon.json` |

## What is not in the repository

Excluded because they are large and reproducible from the code:

- **Captured hidden states** (`runs/*.npz`, 200–280 MB each) — regenerate with `run/capture.py`.
  Note that no producer reads these directly: the permutation nulls are precomputed into
  `runs/null_*.json` by `run/nullcal.py`, which is why the analysis path needs no GPU.
- **LoRA adapters** (`runs/*_adapter/`, 115–182 MB each) — every run's hyperparameters are recorded
  in the corresponding `runs/lora_*.json`, so no experimental detail is lost.
- **Datasets** (`data/`, 6.2 GB) — synthetic families regenerate from `gen/`; COCO and ChartQA are
  public downloads.
- **`runs/v1_contaminated/`** — a pre-`eager` capture, kept locally and deliberately *not* shipped.
  It stores states under filenames identical to the live ones, so shipping it would invite a
  producer being run against it. Capturing under one attention kernel and probing under another is
  a silent train/serve skew; it costs the robust probes at most 1.3 pp and the memorising tracking
  probe 62 pp, which is how it was found. `manifest.py` checks for exactly this.

All 94 artefacts the analysis actually reads *are* committed, which is why the reproduction path
above needs nothing else. `python3 run/manifest.py --json` lists them.

## Citation

Sooyoung Jang, Department of Computer Engineering, Hanbat National University
([ORCID 0000-0002-6931-9592](https://orcid.org/0000-0002-6931-9592)). Under review; citation
details will be added on acceptance.

## License

[GPL-3.0-only](LICENSE) © 2026 Sooyoung Jang — the code, the generators and the run artefacts
committed here.

This program is free software: you can redistribute it and/or modify it under the terms of
version 3 of the GNU General Public License as published by the Free Software Foundation. It is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE)
for the full text.

**Why GPL and not something permissive.** Two committed files are derived from ChartQA
annotations, and ChartQA is GPL-3.0 — see
[Re-running the experiments](#re-running-the-experiments) for exactly what those files contain.
Copyleft propagates, so the repository that carries them is GPL-3.0 as well. It is
`GPL-3.0-only` rather than `-or-later` because ChartQA ships the plain licence text without the
"or any later version" clause: that forward option was never granted upstream, so it is not
claimed here.

**On the earlier releases.** Commits up to and including `ce1c88c` were published under MIT.
That grant cannot be withdrawn from anyone who already has those snapshots, and this section
does not attempt to; it governs this commit onward.

**Third-party material is not covered by any of the above.** COCO *val2017* and ChartQA are
downloaded by the user under their own terms, and no image from either is redistributed here.
The model weights (Qwen2.5-VL, InternVL3, SmolVLM) carry their own licenses and are fetched
from their upstream repositories.
