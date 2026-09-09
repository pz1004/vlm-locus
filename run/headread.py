# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""The model's own head, restricted to the probe's answer set. No GPU, no forward pass.

The paper compares a closed-set linear probe against the model's free-form generation and reads
the difference as a readout failure. A reviewer's objection is that this conflates two things: the
head may fail to *extract* the attribute, or it may extract it and lose the comparison because it
was decoding over the whole vocabulary while the probe chose among four classes. The two have
different consequences and the existing experiments cannot separate them.

They can be separated for free. run/assert_head.py establishes that the cached final state is the
vector the frozen unembedding consumes, so applying that unembedding to the same cached states and
taking the argmax over *the probe's own answer tokens* gives the model's head the probe's
candidate set and nothing else. Three readouts on identical items and an identical split:

  model     free-form generation, parsed          (what the paper reports)
  head      frozen unembedding, answer set only   (this file)
  probe     fitted linear readout                 (what the paper reports)

head >> model says a large part of the reported gap is candidate-space mismatch. head ~ model says
the head genuinely fails to extract, which is the paper's claim.

Two limits, both structural rather than incidental. The state is captured at the final prompt
token, so the unembedding gives a distribution over the FIRST answer token; the comparison is
exact only where distinct answers have distinct first tokens. That holds for the counting and
spatial families and fails for chart and glyph, whose answers are multi-token and collide ("20"
and "25" share a first token in all three tokenisers). Those families are reported on the
coarsened task instead -- both readouts predicting the first token, which for chart is the tens
digit -- and marked as such. Second, only bf16 configurations are scored: the nf4 states were
captured under quantised weights and the unembedding here is the released bf16 one.
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
import torch
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
# tag -> (model id, cache dir name, head tensor key). Qwen ties its head to the embedding.
Q3 = ("Qwen/Qwen2.5-VL-3B-Instruct", "models--Qwen--Qwen2.5-VL-3B-Instruct",
      "model.embed_tokens.weight")
SM = ("HuggingFaceTB/SmolVLM-Instruct", "models--HuggingFaceTB--SmolVLM-Instruct",
      "lm_head.weight")
IV = ("OpenGVLab/InternVL3-2B-hf", "models--OpenGVLab--InternVL3-2B-hf",
      "language_model.lm_head.weight")
TAGS = {"3b": Q3, "real": Q3, "realchart": Q3, "smolm": SM, "smol_real_3b": SM,
        "smol_real_chart": SM, "ivl_real_3b": IV, "ivl_real_chart": IV}

# where the model's own generations live, mirroring run/canon.py's GEN
GEN = {"3b": "runs/branches6_test.jsonl", "smolm": "runs/smolm_gen.jsonl"}


def raw_gen(tag):
    """The model's generated string per item id, not the precomputed correctness flag."""
    f = GEN.get(tag, f"runs/{tag}_gen.jsonl")
    if not os.path.exists(f):
        return {}
    out = {}
    for line in open(f):
        r = json.loads(line)
        out[r["id"]] = str(r["out"]["none"] if "out" in r else r.get("gen_vis", ""))
    return out


def coarse_model(txt, fam, tk):
    """The model's answer at the head's granularity: its first answer token.

    On chart the paper credits the model for reading the value and declining the rounding
    instruction (run/fix_chart_scoring.py), so the same rounding is applied here before
    coarsening -- otherwise this comparison would grade the model more harshly than the
    tables do and the gap it reports would be an artefact of the grader.
    """
    t = txt.strip()
    if fam == "chart":
        # run/fix_chart_scoring.py's extraction, verbatim: the first number in the string,
        # clamped to the axis range, rounded to the nearest 5. Anything stricter would score
        # "50%" and "11.5." as unparseable, which the paper's own grader does not.
        m = re.search(r"-?\d+(?:\.\d+)?", t)
        if not m:
            return None
        t = str(int(round(min(max(float(m.group()), 0.0), 100.0) / 5) * 5))
    ids = tk.encode(t, add_special_tokens=False)
    return ids[0] if ids else None


TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]),
              glyph=lambda m: int(m["attribute"]["value"]))


def head_rows(cache, key, ids):
    """Just the unembedding rows for these token ids, sliced out of the checkpoint."""
    d = glob.glob(f"{HUB}/{cache}/snapshots/*/")[0]
    for f in sorted(glob.glob(d + "*.safetensors")):
        with safe_open(f, framework="pt") as h:
            if key in h.keys():
                sl = h.get_slice(key)
                return torch.stack([sl[i:i + 1][0] for i in ids]).float().numpy()
    raise SystemExit(f"{key} not in {cache}")


def main(tags):
    # the paper's own model accuracy, for the families scored exactly -- taken rather than
    # recomputed, so this table cannot disagree with Tables 4 and 6 about the model
    C = json.load(open("out/canon.json"))
    CANON = {(r["tag"], r["family"]): r["model_acc"] for r in C["synthetic"] + C["real"]}
    out = []
    print(f"{'cell':26s}{'n':>4s}{'k':>3s}{'exact':>7s}{'model':>7s}{'head':>7s}{'probe':>7s}"
          f"   note")
    for tag in tags:
        mid, cache, key = TAGS[tag]
        tk = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        st = f"runs/states_{tag}.npz"
        if not os.path.exists(st):
            continue
        V = np.load(st)["vis"].astype(np.float32)
        meta = json.load(open(st.replace(".npz", "_meta.json")))
        G = raw_gen(tag)
        for fam in sorted({m["family"] for m in meta} & set(TARGET)):
            idx = np.array([i for i, m in enumerate(meta) if m["family"] == fam])
            y = np.array([TARGET[fam](meta[i]) for i in idx])
            keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= 8], dtype=int)
            idx, y = idx[keep], y[keep]
            if len(idx) < 20:
                continue
            tr, rest = train_test_split(np.arange(len(idx)), test_size=0.45, random_state=0,
                                        stratify=y)
            _, te = train_test_split(rest, test_size=0.55, random_state=0, stratify=y[rest])
            answers = sorted({str(v) for v in y})
            first = {a: tk.encode(a, add_special_tokens=False)[0] for a in answers}
            exact = len(set(first.values())) == len(answers) and all(
                len(tk.encode(a, add_special_tokens=False)) == 1 for a in answers)

            # the head, over the probe's answer tokens only
            ids = sorted(set(first.values()))
            W = head_rows(cache, key, ids)
            S = V[idx[te], -1]
            pick = np.argmax(S @ W.T, axis=1)
            hpred = [ids[p] for p in pick]
            gold_first = [first[str(v)] for v in y[te]]
            head_acc = float(np.mean([a == b for a, b in zip(hpred, gold_first)]))

            # the probe, same split, same recipe -- scored on the same granularity as the head
            pipe = make_pipeline(StandardScaler(),
                                 PCA(n_components=min(64, len(tr) - 1), random_state=0),
                                 LogisticRegression(max_iter=1000, C=0.5))
            pipe.fit(V[idx[tr], -1], y[tr])
            ppred = pipe.predict(S)
            probe_exact = float(np.mean(ppred == y[te]))
            probe_acc = probe_exact if exact else float(
                np.mean([first[str(a)] == b for a, b in zip(ppred, gold_first)]))

            # the model, at whichever granularity this family is scored on
            if exact:
                model_acc = CANON.get((tag, fam))
            else:
                mp = [coarse_model(G.get(meta[i]["id"], ""), fam, tk) for i in idx[te]]
                have = [(a, b) for a, b in zip(mp, gold_first) if a is not None]
                model_acc = (float(np.mean([a == b for a, b in have]))
                             if len(have) == len(mp) else None)
            note = "exact" if exact else f"coarsened to first token ({len(ids)} of {len(answers)})"
            out.append(dict(tag=tag, family=fam, n=len(te), classes=len(answers),
                            model=model_acc, head=head_acc, probe=probe_acc,
                            probe_exact=probe_exact, exact=bool(exact), head_classes=len(ids)))
            m = "   --" if model_acc is None else f"{100 * model_acc:5.1f}"
            print(f"{tag + '/' + fam:26s}{len(te):4d}{len(answers):3d}"
                  f"{('  yes' if exact else '   no'):>7s}{m:>7s}"
                  f"{100 * head_acc:7.1f}{100 * probe_acc:7.1f}   {note}", flush=True)
    json.dump(out, open("runs/headread.json", "w"), indent=1)
    print("  wrote runs/headread.json")


if __name__ == "__main__":
    main(sys.argv[1:] or list(TAGS))
