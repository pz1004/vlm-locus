# vlm-locus

Code, dataset generators and frozen run artefacts for **"Perception or Readout? Locating
Vision–Language Model Failures with Exact Counterfactuals and a Measured Control Floor."**

When a vision–language model answers a visual question wrongly, the error may be a *perception*
failure — the queried attribute never entered the representation — or a *readout* failure — the
attribute is present but the output head does not extract it. Benchmark accuracy cannot separate
the two, yet the distinction decides whether supervised adaptation will help. This repository
implements a protocol that separates them, and reproduces every number in the paper.

## Reproducing the paper's tables and figures

Every table and figure comes from the frozen artefacts in `runs/`. **No GPU, no model download and
no dataset are needed for this path** — `canon.py` and `figs.py` import only numpy, scipy,
scikit-learn and matplotlib:

```bash
pip install numpy scipy scikit-learn matplotlib
python3 run/canon.py     # -> out/tables/*.tex, out/canon.json
python3 run/figs.py      # -> out/figs/*.pdf, *.png
```

`run/canon.py` is the single source of every reported number; nothing is transcribed by hand.
`python3 run/canon.py --audit` additionally prints the canonical final-layer probe against the
alternative selected-layer estimator for all 28 cells (§11 of the paper).

Output locations default to `out/` and are overridable with `VLM_LOCUS_TEX`, `VLM_LOCUS_FIGS` and
`VLM_LOCUS_JSON`.

## Re-running the experiments

Needs a CUDA GPU (everything reported was produced on a single 16 GB RTX 4070 Ti SUPER) and the
full dependency set:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Then build the datasets and run a stage:

```bash
python3 gen/generate.py        # synthetic families, rendered from specs
python3 gen/build_real.py      # real families (needs COCO val2017 + ChartQA)
python3 gen/verify_real.py     # the per-family pixel-identity guards
bash    run/pipeline.sh        # capture -> probe -> counterfactual -> score
```

The real families are built on [COCO](https://cocodataset.org) *val2017* and
[ChartQA](https://github.com/vis-nlp/ChartQA); both are public downloads and neither is
redistributed here.

## Layout

| path | contents |
|---|---|
| `run/` | protocol, probes, counterfactuals, adaptation, analysis. `canon.py` emits every reported number |
| `gen/` | dataset generators and the per-family verification guards |
| `runs/` | frozen run artefacts (JSON/JSONL) plus serialised probes — the inputs `canon.py` reads |
| `docs/` | the development record: per-stage results, controls, and the defects the controls caught |

## What is not in the repository

Three classes of artefact are excluded because they are large and reproducible from the code:

- **Captured hidden states** (`runs/*.npz`, 200–280 MB each) — regenerate with `run/capture.py`.
- **LoRA adapters** (`runs/*_adapter/`, 115–182 MB each) — every run's hyperparameters are recorded
  in the corresponding `runs/lora_*.json`, so no experimental detail is lost.
- **Datasets** (`data/`, 6.2 GB) — synthetic families regenerate from `gen/`; COCO and ChartQA are
  public downloads.

The small artefacts that the reported numbers actually depend on *are* included, which is why the
table-and-figure path above needs nothing else.

## Citation

Sooyoung Jang, Department of Computer Engineering, Hanbat National University
([ORCID 0000-0002-6931-9592](https://orcid.org/0000-0002-6931-9592)). Under review; citation
details will be added on acceptance.

## License

[MIT](LICENSE) © 2026 Sooyoung Jang. This covers the contents of this repository — the code, the
generators and the run artefacts committed here.

It does not extend to the third-party datasets the real-image families are built on. COCO
*val2017* and ChartQA are downloaded by the user under their own terms and are not redistributed
here; the model weights (Qwen2.5-VL, InternVL3, SmolVLM) likewise carry their own licenses and are
fetched from their upstream repositories.
