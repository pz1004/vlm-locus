"""Label-free routing features: the experiment the whole method claim now rests on.

The bind: LRC's routing features come from a probe fitted on labelled calibration data. But if
you HAVE that data you can fit B-read, which answers at 88.9% for 1.01x and beats every router
configuration outright (W4 and W5 both fail once B-read is admitted). And if you DON'T have it,
LRC has no features. Either way the router is redundant -- unless it can route on signals that
need no labels.

Everything here is computable from the model itself plus the task's answer space, which the
task statement gives you:

  p_max      max_a softmax over the answer space, from the model's own candidate log-probs
  ent        entropy of that distribution -- decoder uncertainty, no probe involved
  top2       log p(top-1) - log p(top-2)
  marg_self  log p_vis(a_hat) - log p_blind(a_hat) for the model's own answer a_hat.
             This is the plan's evidence margin computed on the DECODER instead of the probe:
             how much the image moved the model's own belief in its own answer.
  blind_max  max softmax over the answer space with the image blindfolded -- prior
             answerability, measured rather than assumed
  agree      whether the free-form greedy answer matches the constrained argmax
  gen_lp     mean log-prob of the emitted answer tokens, from the generation pass (free)

The last one costs nothing: it rides on the base pass. The candidate-space features cost 2|A|
forwards as implemented, batchable to ~2. Both are reported so the router can be priced under
either implementation.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import branches as B


@torch.inference_mode()
def gen_with_lp(model, proc, image, question, dev, mx=10):
    enc = proc(text=[B.build_prompt(proc, question)], images=[image], return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=mx, do_sample=False, output_scores=True,
                         return_dict_in_generate=True,
                         pad_token_id=proc.tokenizer.eos_token_id)
    n = enc.input_ids.shape[1]
    seq = out.sequences[0, n:]
    lps = []
    for t, sc in enumerate(out.scores):
        if t >= seq.shape[0]: break
        lp = torch.log_softmax(sc[0].float(), -1)
        lps.append(float(lp[seq[t]]))
    txt = proc.tokenizer.decode(seq, skip_special_tokens=True).strip()
    return txt, (float(np.mean(lps)) if lps else -20.0)


def main(a):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=dev, attn_implementation="eager").eval()
    P = np.load(a.probes, allow_pickle=True).item()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl")))}
    grey = Image.new("RGB", (448, 448), (127, 127, 127))
    f = open(a.out, "w"); t0 = time.time(); n = 0
    for fam, pf in P.items():
        if a.families and fam not in a.families: continue
        ids = json.load(open(a.ids))[fam] if a.ids else pf["test_ids"]
        for iid in ids:
            r = recs[iid]; img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
            space = r["answer_space"]
            txt, gen_lp = gen_with_lp(model, proc, img, r["question"], dev)
            lv = B.cand_logits(model, proc, img,  r["question"], space, dev)
            lb = B.cand_logits(model, proc, grey, r["question"], space, dev)
            f.write(json.dumps(dict(id=iid, family=fam, answer=r["answer"], space=space,
                                    gen=B.parse_in_space(txt, space), gen_lp=gen_lp,
                                    lv=lv.tolist(), lb=lb.tolist())) + "\n")
            f.flush(); n += 1
            if n % 50 == 0: print(f"  {n}  {(time.time()-t0)/n:.2f}s/item", flush=True)
    f.close(); print(f"wrote {a.out}: {n} items in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--out", default="runs/unsup_test_3b.jsonl")
    p.add_argument("--ids", default="")
    p.add_argument("--families", nargs="*", default=None,
                   help="restrict to these families; the rest of the dataset is unchanged, so a "
                        "partial re-run merges with the existing output")
    main(p.parse_args())
