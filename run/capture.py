"""Capture hidden states for the presence test (gate G1).

For every item, two forward passes -- the real image and a mid-grey blindfold -- with
`output_hidden_states`. We keep the residual stream at the **last prompt token**, the position
from which the answer is generated, for every layer. That is the state the router would read
at inference, so it is the state the probe must be fitted on.

Storage: L layers x D dims x float16 per item per condition. For Qwen2.5-VL-3B that is
37 x 2048 x 2 bytes = 152 KB per item per condition, so 800 items fit in ~240 MB.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadm as L


@torch.inference_mode()
def last_token_states(model, proc, image, question, dev):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=[image], return_tensors="pt").to(dev)
    out = model(**enc, output_hidden_states=True)
    hs = torch.stack([h[0, -1] for h in out.hidden_states])          # [L+1, D]
    return hs.to(torch.float16).cpu().numpy()


def main(a):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    # eager, to match every downstream consumer. Capturing under sdpa and then applying the
    # probe to eager activations (which is what branches.py, unsup.py and cfprobe.py produce)
    # is a silent train/serve skew. It costs the robust probes <=1.3 pp -- chart 98.6->99.0,
    # spatial 97.3->97.0, counting 69.3->68.0 -- and the memorising tracking probe 62 pp,
    # 90.7->29.0, which is how the skew was found.
    model = L.load(a.model, dev, a.load_4bit).eval()
    recs = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))]
    recs = [r for r in recs if r["cf_of"] is None]
    if a.limit:
        keep, per = [], {}
        for r in recs:
            per[r["family"]] = per.get(r["family"], 0) + 1
            if per[r["family"]] <= a.limit: keep.append(r)
        recs = keep
    grey = Image.new("RGB", (448, 448), (127, 127, 127))
    V = B = None
    meta = []
    t0 = time.time()
    for i, r in enumerate(recs):
        img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
        hv = last_token_states(model, proc, img,  r["question"], dev)
        hb = last_token_states(model, proc, grey, r["question"], dev)
        if V is None:
            V = np.zeros((len(recs), *hv.shape), np.float16)
            B = np.zeros_like(V)
            print(f"  states {hv.shape} (layers x dim); buffer "
                  f"{2*V.nbytes/2**20:.0f} MB", flush=True)
        V[i], B[i] = hv, hb
        meta.append(dict(id=r["id"], family=r["family"], arm=r["prior_arm"],
                         level=r["difficulty"].get("level"), answer=r["answer"],
                         answer_space=r["answer_space"], chance=r["chance"],
                         attribute=r["attribute"]))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(recs)}  {el/(i+1):.2f}s/item  eta {(len(recs)-i-1)*el/(i+1)/60:.1f} min",
                  flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, vis=V, blind=B)
    json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"))
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/2**20:.0f} MB) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-4bit", action="store_true",
                   help="load in nf4; needed for 7B on 16 GB, and available on "
                        "smaller models so the scale axis holds precision constant")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--out", default="runs/states_3b.npz")
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
