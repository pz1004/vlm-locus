"""Assert that the state the probe is fitted on is the vector the language-model head consumes.

The whole protocol rests on this: run/capture.py keeps every entry of `out.hidden_states` at the
final prompt token, and run/layers.py reads index -1. The claim is that index -1 is the
*post-final-norm* vector, so applying the frozen unembedding to it reproduces the model's own
logits -- as opposed to the pre-norm residual stream, which it would not.

That is a property of the model implementation, not of our code, so it is asserted rather than
assumed. One forward pass per model. The models run in bf16, which carries about three decimal
digits, so the check is not exact equality but a discriminative one: index -1 must agree with the
model's own logits to bf16 tolerance and must agree far better than the layer below it, which is
the pre-norm state and reproduces nothing.
"""
from __future__ import annotations
import argparse, os, sys
import torch
from PIL import Image
from transformers import AutoProcessor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadm as L


@torch.inference_mode()
def check(model_id, load_4bit=False, dev="cuda"):
    proc = AutoProcessor.from_pretrained(model_id)
    model = L.load(model_id, dev, load_4bit).eval()
    img = Image.new("RGB", (448, 448), (127, 127, 127))
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "What is in this image?"}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=[img], return_tensors="pt").to(dev)
    out = model(**enc, output_hidden_states=True)

    h = out.hidden_states[-1][0, -1]                 # what run/layers.py reads
    pre = out.hidden_states[-2][0, -1]               # the layer below, for contrast
    W = model.get_output_embeddings()                # the frozen unembedding
    logits = out.logits[0, -1].float()
    lo_post, lo_pre = W(h).float(), W(pre).float()

    def rel(a, b): return float((a - b).abs().max() / b.abs().max())
    r_post, r_pre = rel(lo_post, logits), rel(lo_pre, logits)
    agree_post = int((lo_post.argmax() == logits.argmax()).item())
    print(f"  hidden_states[-1] -> unembedding : max rel. err {r_post:.2e}   argmax match {agree_post}")
    print(f"  hidden_states[-2] -> unembedding : max rel. err {r_pre:.2e}   (contrast)")
    # bf16 leaves ~1e-3 of relative error; what identifies the head's input is that index -1
    # lands within that while the layer below is off by order 1.
    ok = r_post < 1e-2 and agree_post and r_pre > 100 * r_post
    print(f"  ratio pre/post = {r_pre / r_post:.0f}x")
    print(f"  {'PASS' if ok else 'FAIL'}: index -1 {'is' if ok else 'is NOT'} the head's input")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--load-4bit", action="store_true")
    a = p.parse_args()
    print(a.model)
    raise SystemExit(0 if check(a.model, a.load_4bit) else 1)
