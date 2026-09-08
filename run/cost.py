"""Measure what each branch actually costs on this GPU.

Every cost claim so far has been arithmetic on max_new_tokens: "CoT costs 14x because 140
tokens vs 10". That ignores prefill, ignores that B-prior scores the entire answer space
against two images (2|A| forward passes -- 36 on chart), and ignores that B-look pays a
second full prefill. A cost-matched claim built on assumed costs is not cost-matched.

Recorded per branch, per item: wall-clock (cuda-synchronised), forward passes, prefill
tokens, decode tokens. The probe read is recorded separately as router overhead, since the
router pays it once regardless of which branch it then selects.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import branches as B


class Meter:
    """Counts forward passes with a module hook.

    Patching model.forward instead would break generate(): transformers inspects the forward
    signature to validate model kwargs, and a (*a, **k) wrapper makes every real kwarg look
    unused.
    """
    def __init__(self, model):
        self.n = 0
        def hook(*_a, **_k): self.n += 1
        self.h = model.register_forward_pre_hook(hook)
    def reset(self): self.n = 0
    def restore(self): self.h.remove()


@torch.inference_mode()
def gen_fused(model, proc, image, question, dev, layer, mx=10):
    """Generate the base answer AND read the probe layer in ONE forward pass.

    Charging the probe read as a separate pass (it costs 0.83x a base call) is an artifact of
    how the grid script is written, not a property of the method: the state the probe needs is
    the last prompt token at layer L, which the generation prefill already computes. Fusing
    them is what a deployment would do, and it is the number the cost claim has to use.
    """
    enc = proc(text=[B.build_prompt(proc, question)], images=[image], return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                         output_hidden_states=True, return_dict_in_generate=True,
                         pad_token_id=proc.tokenizer.eos_token_id)
    hs = out.hidden_states[0][layer][0, -1]
    n = enc.input_ids.shape[1]
    txt = proc.tokenizer.decode(out.sequences[0, n:], skip_special_tokens=True).strip()
    return txt, hs


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0


def main(a):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=dev, attn_implementation="eager").eval()
    IMG_TOK = model.config.image_token_id
    P = np.load(a.probes, allow_pickle=True).item()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl")))}
    grey = Image.new("RGB", (448, 448), (127, 127, 127))
    meter = Meter(model)
    rows = []

    for fam, pf in P.items():
        ids = json.load(open(a.ids))[fam] if a.ids else pf["test_ids"]
        ids = ids[:a.per_family]
        comp, mean, scale = np.array(pf["components"]), np.array(pf["mean"]), np.array(pf["scale"])
        coef, inter = np.array(pf["coef"]), np.array(pf["intercept"])
        dirs = np.array(pf["directions"])
        for k, iid in enumerate(ids):
            r = recs[iid]; img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
            space = r["answer_space"]
            enc0 = proc(text=[B.build_prompt(proc, r["question"])], images=[img],
                        return_tensors="pt").to(dev)
            n_prompt = int(enc0.input_ids.shape[1])

            def measure(name, fn, prefills, decode_budget):
                meter.reset()
                out, dt = timed(fn)
                rows.append(dict(family=fam, id=iid, branch=name, sec=dt, fwd=meter.n,
                                 prompt_tok=n_prompt, prefill_tok=n_prompt * prefills,
                                 decode_budget=decode_budget, space=len(space)))
                return out

            if k == 0:   # warm-up, not recorded
                B.gen(model, proc, img, r["question"], dev)

            measure("none", lambda: B.gen(model, proc, img, r["question"], dev), 1, 10)

            # router overhead: one hidden-state read to get the probe features
            def probe_read():
                with torch.inference_mode():
                    e = proc(text=[B.build_prompt(proc, r["question"])], images=[img],
                             return_tensors="pt").to(dev)
                    return model(**e, output_hidden_states=True).hidden_states[pf["layer"]][0, -1]
            hs = measure("_probe", probe_read, 1, 0).detach().float().cpu().numpy()
            measure("none+probe", lambda: gen_fused(model, proc, img, r["question"], dev,
                                                    pf["layer"]), 1, 10)
            logits = coef @ (((hs - mean) / scale - np.array(pf["pca_mean"])) @ comp.T) + inter

            measure("steer", lambda: B.gen(model, proc, img, r["question"], dev,
                    hook=dict(layer=pf["layer"] - 1,
                              fn=B.steer_hook(dirs[int(np.argmax(logits))], a.alpha))), 1, 10)
            measure("prior", lambda: (B.cand_logits(model, proc, img, r["question"], space, dev),
                                      B.cand_logits(model, proc, grey, r["question"], space, dev)),
                    2 * len(space), 0)
            measure("look", lambda: B.gen(model, proc, B.crop_for(r, img), r["question"], dev), 1, 10)
            measure("attn", lambda: B.gen_attn(model, proc, img, r["question"], dev, IMG_TOK,
                                               a.gamma, referents=r["referents"]), 1, 10)
            cot_q = (r["question"] + " Think step by step, briefly, then end with "
                     "'ANSWER: <your answer>'.")
            measure("cot", lambda: B.gen(model, proc, img, cot_q, dev, mx=140), 1, 140)
        print(f"  {fam} done", flush=True)

    meter.restore()
    json.dump(rows, open(a.out, "w"))
    # ---- report
    import collections
    by = collections.defaultdict(list)
    for r in rows: by[r["branch"]].append(r)
    base = np.median([r["sec"] for r in by["none"]])
    print(f"\n{'branch':8s}{'sec':>8s}{'x none':>8s}{'fwd':>6s}{'prefill tok':>12s}{'decode':>8s}")
    for b in ("none", "none+probe", "steer", "attn", "look", "cot", "prior", "_probe"):
        g = by[b]
        if not g: continue
        s = np.median([r["sec"] for r in g])
        print(f"{b:8s}{s:8.3f}{s/base:8.2f}{np.median([r['fwd'] for r in g]):6.0f}"
              f"{np.median([r['prefill_tok'] for r in g]):12.0f}"
              f"{np.median([r['decode_budget'] for r in g]):8.0f}")
    print(f"\nper family (seconds, median):")
    fams = sorted({r["family"] for r in rows})
    print(f"{'branch':8s}" + "".join(f"{f:>11s}" for f in fams))
    for b in ("none", "none+probe", "steer", "attn", "look", "cot", "prior", "_probe"):
        print(f"{b:8s}" + "".join(
            f"{np.median([r['sec'] for r in by[b] if r['family']==f]):11.3f}" for f in fams))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--out", default="runs/cost_3b.json")
    p.add_argument("--alpha", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--per_family", type=int, default=12)
    p.add_argument("--ids", default="")
    main(p.parse_args())
