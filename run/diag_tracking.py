# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Is tracking failing at parsing or at tracking?

Three probes on the same images, from easiest to hardest:
  A. read the start panel   -- pure perception, no tracking at all
  B. read the swap labels   -- pure OCR of the instruction panels
  C. the real question      -- perception + tracking

If A fails, the family measures panel-parsing and the rendering has to change. If A and B
pass while C fails, the family measures what it claims to.
"""
import json, os, re, sys, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

@torch.inference_mode()
def gen(model, proc, img, q, dev, mx=24):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    p = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    e = proc(text=[p], images=[img], return_tensors="pt").to(dev)
    o = model.generate(**e, max_new_tokens=mx, do_sample=False,
                       pad_token_id=proc.tokenizer.eos_token_id)
    return proc.tokenizer.decode(o[0, e.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main(data, n=40):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                        device_map=dev).eval()
    recs = [json.loads(l) for l in open(os.path.join(data, "manifest.jsonl"))
            if json.loads(l)["family"] == "tracking" and json.loads(l)["cf_of"] is None][:n]
    QA = ("In the leftmost panel, labelled 'start', the red ball is drawn under one cup. "
          "Which cup is it? Answer with a number only.")
    QB = ("How many panels are in this image, counting the leftmost 'start' panel and the "
          "rightmost 'end' panel? Answer with a number only.")
    QC = ("Read the text label written at the top of the second panel from the left. "
          "Reply with that text exactly.")
    res = []
    t0 = time.time()
    for r in recs:
        img = Image.open(os.path.join(data, r["image"])).convert("RGB")
        s = r["spec"]
        a = gen(model, proc, img, QA, dev, 8)
        b = gen(model, proc, img, QB, dev, 8)
        c = gen(model, proc, img, QC, dev, 16)
        d = gen(model, proc, img, r["question"], dev, 8)
        num = lambda t: (m.group(0) if (m := re.search(r"\d+", t)) else None)
        i, j = s["swaps"][0]
        res.append(dict(
            id=r["id"], level=r["difficulty"]["level"], n_swaps=len(s["swaps"]),
            start_true=s["start"] + 1, start_read=num(a), start_ok=num(a) == str(s["start"] + 1),
            panels_true=len(s["swaps"]) + 2, panels_read=num(b),
            panels_ok=num(b) == str(len(s["swaps"]) + 2),
            label_true=f"swap {i+1}<->{j+1}", label_read=c,
            label_ok=(str(i + 1) in c and str(j + 1) in c),
            answer=r["answer"], task_read=num(d), task_ok=num(d) == r["answer"]))
    n = len(res)
    print(f"\n{n} tracking items, {(time.time()-t0)/n:.2f}s/item\n")
    print(f"  A. reads the start panel correctly   {sum(x['start_ok'] for x in res):3d}/{n}"
          f"   ({100*sum(x['start_ok'] for x in res)/n:.0f}%)   <- pure perception")
    print(f"  B. counts the panels correctly       {sum(x['panels_ok'] for x in res):3d}/{n}"
          f"   ({100*sum(x['panels_ok'] for x in res)/n:.0f}%)")
    print(f"  C. reads a swap label correctly      {sum(x['label_ok'] for x in res):3d}/{n}"
          f"   ({100*sum(x['label_ok'] for x in res)/n:.0f}%)   <- pure OCR")
    print(f"  D. answers the tracking question     {sum(x['task_ok'] for x in res):3d}/{n}"
          f"   ({100*sum(x['task_ok'] for x in res)/n:.0f}%)")
    both = [x for x in res if x["start_ok"] and x["label_ok"]]
    if both:
        print(f"\n  on the {len(both)} items where perception AND OCR both succeed, "
              f"tracking accuracy is {100*sum(x['task_ok'] for x in both)/len(both):.0f}%")
    print("\n  sample start-panel reads (true -> read):")
    for x in res[:8]:
        print(f"    {x['id']}  {x['start_true']} -> {x['start_read']!r:>6s}"
              f"   label {x['label_true']!r} -> {x['label_read']!r}")
    json.dump(res, open("runs/diag_tracking.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sweep")
