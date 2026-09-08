"""Gate G3: the oracle over corrections. Is there anything for a router to recover?

Four branches, each a re-implementation of a published correction family, run on every item:

  B-none   the base answer
  B-steer  add alpha * (probe direction for the probe's predicted class) to the residual
           stream at the probe's layer, at every position from the last prompt token onward,
           at a magnitude matched to that layer's residual norm            [GRASP, RUDDER]
  B-prior  contrastive rescoring of the answer space: logit_vis - lam * logit_blind, using
           the same mid-grey blindfold as the probe's reference            [DEGAP, TriCD, VCD]
  B-look   crop to the referent boxes (or a centre zoom where the family has none), upscale,
           re-read                                                          [HiDe, SPARC]

The oracle is `any branch correct`. It is the ceiling on any router over this library, and it
is the number that decides whether LRC can exist. A specificity control accompanies B-steer:
the same push toward a *wrong* class at matched norm. If the answer follows wherever it is
pushed, the branch is a puppet string and is discarded for that cell rather than interpreted.
"""
from __future__ import annotations
import argparse, json, os, re, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def parse_in_space(text, space):
    """Match the first token in `text` that is a member of THIS item's answer space.

    The previous per-family regexes hardcoded an alphabet (chart used [A-G]) and silently
    scored every answer outside it as wrong. At chart level 3 the space is A..I, so 59% of
    the model's replies -- every "H" and "I" -- were counted as errors. Deriving the pattern
    from the item's own answer_space makes that class of bug impossible.
    """
    t = text.strip()
    up = t.upper()
    # longest-first so "10" is preferred over "1", and multi-word options match whole
    for cand in sorted(space, key=len, reverse=True):
        import re as _re
        if _re.search(r"(?<![A-Za-z0-9])" + _re.escape(cand.upper()) + r"(?![A-Za-z0-9])", up):
            return cand
    return None


PARSE = {"counting": lambda t: (m.group(0) if (m := re.search(r"\d+", t)) else None),
         "tracking": lambda t: (m.group(0) if (m := re.search(r"\d+", t)) else None),
         "spatial":  lambda t: next((w for w in re.findall(r"[a-z]+", t.lower())
                                     if w in ("left", "right", "above", "below")), None),
         "chart":    lambda t: next((c for c in re.findall(r"\b([A-L])\b", t.upper())), None)}


def build_prompt(proc, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def gen(model, proc, image, question, dev, mx=10, hook=None):
    enc = proc(text=[build_prompt(proc, question)], images=[image], return_tensors="pt").to(dev)
    h = None
    if hook is not None:
        layer = model.model.language_model.layers[hook["layer"]]
        h = layer.register_forward_hook(hook["fn"])
    try:
        out = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                             pad_token_id=proc.tokenizer.eos_token_id)
    finally:
        if h is not None: h.remove()
    return proc.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def cand_logits(model, proc, image, question, cands, dev):
    """Summed log-prob of each candidate string as the reply."""
    ps = build_prompt(proc, question)
    out = []
    for c in cands:
        ep = proc(text=[ps], images=[image], return_tensors="pt").to(dev)
        ef = proc(text=[ps + c], images=[image], return_tensors="pt").to(dev)
        n = ep.input_ids.shape[1]
        lg = model(**ef).logits[0]
        lp = torch.log_softmax(lg[:-1].float(), -1)
        ids = ef.input_ids[0]
        out.append(float(lp[torch.arange(n - 1, ids.shape[0] - 1), ids[n:]].sum()))
    return np.array(out)


def attn_bias_hooks(model, img_mask, gamma):
    """B-attn: add a constant gamma to the attention logits of every image-token KEY, at every
    layer. The decoder receives an additive float mask, so a positive bias on those columns is
    exactly an attention re-weighting toward the image, with no change to the attention maths.
    Requires attn_implementation='eager' -- sdpa takes a causal fast path and materialises no
    mask, so every branch is run under eager to keep the comparison numerically identical."""
    handles = []
    def make(_l):
        def pre(_mod, args, kwargs):
            am = kwargs.get("attention_mask")
            if am is None or not torch.is_tensor(am):
                return None
            k = am.shape[-1]
            km = torch.zeros(k, dtype=torch.bool, device=am.device)
            n = min(k, img_mask.numel())
            km[:n] = img_mask[:n]
            bias = torch.zeros(k, dtype=am.dtype, device=am.device)
            bias[km] = gamma
            allowed = (am > -1e30).to(am.dtype)
            kwargs["attention_mask"] = am + bias.view(1, 1, 1, -1) * allowed
            return (args, kwargs)
        return pre
    for lay in model.model.language_model.layers:
        handles.append(lay.register_forward_pre_hook(make(lay), with_kwargs=True))
    return handles


def referent_token_mask(enc, img_token_id, referents, W_img=448, H_img=448, merge=2):
    """Map referent boxes to the image-token grid.

    A uniform boost over all 256 image tokens is inert (1/32 items changed at gamma<=1) and
    harmful above that -- which is what Fox reports for coarse magnitude boosting. The branch
    only becomes distinct if it boosts the tokens the question is ABOUT. Qwen2.5-VL emits
    image_grid_thw in 14-px patches and merges 2x2, so token (r, c) covers a 28-px cell.
    """
    ids = enc.input_ids[0]
    pos = (ids == img_token_id).nonzero(as_tuple=True)[0]
    mask = torch.zeros_like(ids, dtype=torch.bool)
    if pos.numel() == 0 or not referents:
        return (ids == img_token_id)                      # fall back to the whole image
    gh, gw = int(enc["image_grid_thw"][0][1]) // merge, int(enc["image_grid_thw"][0][2]) // merge
    if gh * gw != pos.numel():
        return (ids == img_token_id)
    cell_x, cell_y = W_img / gw, H_img / gh
    hit = torch.zeros(gh * gw, dtype=torch.bool)
    for x0, y0, x1, y1 in referents:
        c0, c1 = max(0, int(x0 // cell_x)), min(gw - 1, int(x1 // cell_x))
        r0, r1 = max(0, int(y0 // cell_y)), min(gh - 1, int(y1 // cell_y))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                hit[r * gw + c] = True
    if not hit.any():
        return (ids == img_token_id)
    mask[pos] = hit.to(mask.device)
    return mask


@torch.inference_mode()
def gen_attn(model, proc, image, question, dev, img_token_id, gamma, mx=10, referents=None):
    enc = proc(text=[build_prompt(proc, question)], images=[image], return_tensors="pt").to(dev)
    img_mask = referent_token_mask(enc, img_token_id, referents)
    hs = attn_bias_hooks(model, img_mask, gamma)
    try:
        out = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                             pad_token_id=proc.tokenizer.eos_token_id)
    finally:
        for h in hs: h.remove()
    return proc.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def steer_hook(direction, alpha):
    d = torch.as_tensor(direction)
    def fn(_mod, _inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = d.to(hs.device, hs.dtype)
        scale = hs[:, -1:, :].norm(dim=-1, keepdim=True) * alpha
        hs = hs + v.view(1, 1, -1) * scale
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
    return fn


def crop_for(rec, img):
    r = rec["referents"]
    if r:
        x0 = max(0, min(b[0] for b in r) - 24); y0 = max(0, min(b[1] for b in r) - 24)
        x1 = min(img.width, max(b[2] for b in r) + 24); y1 = min(img.height, max(b[3] for b in r) + 24)
        if x1 - x0 < 48 or y1 - y0 < 48: return img
        return img.crop((x0, y0, x1, y1)).resize((448, 448), Image.LANCZOS)
    w, h = img.size                                   # centre zoom for families with no boxes
    return img.crop((int(.08*w), int(.08*h), int(.92*w), int(.92*h))).resize((448, 448), Image.LANCZOS)


def main(a):
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16,
                                                        device_map=dev,
                                                        attn_implementation="eager").eval()
    IMG_TOK = model.config.image_token_id
    P = np.load(a.probes, allow_pickle=True).item()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl")))}
    grey = Image.new("RGB", (448, 448), (127, 127, 127))
    f = open(a.out, "w"); t0 = time.time(); n = 0
    for fam, pf in P.items():
        ids = json.load(open(a.ids))[fam] if a.ids else pf["test_ids"]
        ids = ids[:a.limit] if a.limit else ids
        comp = np.array(pf["components"]); mean = np.array(pf["mean"]); scale = np.array(pf["scale"])
        coef = np.array(pf["coef"]); inter = np.array(pf["intercept"]); classes = pf["classes"]
        dirs = np.array(pf["directions"])
        for iid in ids:
            r = recs[iid]; img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
            space = r["answer_space"]; par = lambda t: parse_in_space(t, space)

            base = gen(model, proc, img, r["question"], dev)
            # probe prediction from this item's visual state at the probe layer
            with torch.inference_mode():
                enc = proc(text=[build_prompt(proc, r["question"])], images=[img],
                           return_tensors="pt").to(dev)
                hs = model(**enc, output_hidden_states=True).hidden_states[pf["layer"]][0, -1]
                hs = hs.detach().float().cpu().numpy()
            z = ((hs - mean) / scale - np.array(pf["pca_mean"])) @ comp.T
            logits = coef @ z + inter
            pred = classes[int(np.argmax(logits))]
            wrong_cls = int(np.argmin(logits))

            st = gen(model, proc, img, r["question"], dev,
                     hook=dict(layer=pf["layer"] - 1,
                               fn=steer_hook(dirs[int(np.argmax(logits))], a.alpha)))
            spec = gen(model, proc, img, r["question"], dev,
                       hook=dict(layer=pf["layer"] - 1,
                                 fn=steer_hook(dirs[wrong_cls], a.alpha)))
            lv = cand_logits(model, proc, img,  r["question"], space, dev)
            lb = cand_logits(model, proc, grey, r["question"], space, dev)
            prior = space[int(np.argmax(lv - a.lam * lb))]
            look = gen(model, proc, crop_for(r, img), r["question"], dev)
            attn = gen_attn(model, proc, img, r["question"], dev, IMG_TOK, a.gamma,
                            referents=r["referents"])
            # B-cot: the only branch that changes the COMPUTATION rather than the evidence
            # pathway. Attention editing, steering, prior suppression and re-perception all
            # act on how the image reaches the decoder; none of them gives the model more
            # steps. On tracking, none/steer/look agree on correctness 100% of the time, and
            # the failure there is composition over swaps, so extra steps is the one
            # mechanistically different thing left to try.
            cot_q = (r["question"] + " Think step by step, briefly, then end with "
                     "'ANSWER: <your answer>'.")
            cot_raw = gen(model, proc, img, cot_q, dev, mx=140)
            tail = cot_raw.split("ANSWER:")[-1] if "ANSWER:" in cot_raw else cot_raw[-40:]

            got = dict(none=par(base), steer=par(st), prior=prior, look=par(look),
                       attn=par(attn), cot=par(tail))
            rec = dict(id=iid, family=fam, answer=r["answer"], probe_pred=str(pred),
                       arm=r["prior_arm"], out=got,
                       ok={k: (v == r["answer"]) for k, v in got.items()},
                       spec_steer=par(spec), spec_class=classes[wrong_cls])
            rec["oracle"] = any(rec["ok"].values())
            f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
            if n % 40 == 0:
                print(f"  {n}  {(time.time()-t0)/n:.2f}s/item", flush=True)
    f.close(); print(f"wrote {a.out}: {n} items in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--out", default="runs/branches_3b.jsonl")
    p.add_argument("--alpha", type=float, default=0.30)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default="")
    p.add_argument("--gamma", type=float, default=1.0)
    main(p.parse_args())
