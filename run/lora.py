"""The same-supervision-tier competitor: LoRA fine-tuning on the probe's own labels.

B-read (emit the probe's argmax) needs a labelled calibration set. So does fine-tuning. A
reviewer will ask why you would read a frozen model's latents instead of simply training the
model to say the right thing, and the honest answer has to be measured, not argued.

This trains on the *exact* item ids the probe was fitted on -- `train_ids` in probes_3b.npy --
and evaluates on the *exact* ids run/branches.py scores, with the same prompt builder and the
same answer parser. Nothing about the comparison is free to drift.

Inference cost afterwards is 1.00x base (the adapter is merged), which is *cheaper* than
B-read's 1.01x. So if B-read still wins it wins on accuracy against a strictly cheaper
competitor, and the training cost is pure overhead on top.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branches as B
import loadm as L


def build(proc, img, question, answer=None):
    """Tokenise one example. With `answer`, returns labels masked to the answer tokens only."""
    prompt = B.build_prompt(proc, question)
    if answer is None:
        return proc(text=[prompt], images=[img], return_tensors="pt")
    full = prompt + answer + proc.tokenizer.eos_token
    enc = proc(text=[full], images=[img], return_tensors="pt")
    n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc["labels"] = labels
    return enc


@torch.inference_mode()
def evaluate(model, proc, recs, ids_by_fam, dev, tag):
    # gradient checkpointing forces use_cache=False, which makes generate() quadratic.
    # Turn it off for the duration of the eval and restore it after.
    ckpt = getattr(model, "is_gradient_checkpointing", False)
    if ckpt: model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval(); out = {}; per = {}
    for fam, ids in ids_by_fam.items():
        ok = []
        for iid in ids:
            r = recs[iid]
            img = Image.open(os.path.join(DATA, r["image"])).convert("RGB")
            enc = build(proc, img, r["question"]).to(dev)
            # 40, not 10: a fine-tuned model answers tersely (0-2 unparsed per 225) but the
            # base model prefixes its answer, and at 10 tokens the [base] line reads 6.7 pp low
            # on real charts. The reported gains take base from run/score.py, which already used
            # 40, so this only fixes lora.py's own base line -- but a misleading printout is how
            # a wrong number gets copied into a table.
            g = model.generate(**enc, max_new_tokens=40, do_sample=False,
                               pad_token_id=proc.tokenizer.eos_token_id)
            txt = proc.tokenizer.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            got = B.parse_in_space(txt, r["answer_space"])
            good = got == str(r["answer"])
            # the real-chart family asks for the nearest multiple of 5; the fine-tune is trained
            # on those exact strings while the base model answers 73 for a bar worth 75. Scoring
            # both on the value read keeps the measured gain from being an artefact of the
            # rounding instruction (see run/fix_chart_scoring.py).
            if not good and fam == "chart" and len(r["answer_space"]) > 18:
                m = re.search(r"-?\d+(?:\.\d+)?", txt or "")
                if m:
                    v = min(max(float(m.group()), 0.0), 100.0)
                    good = str(int(round(v / 5) * 5)) == str(r["answer"])
            per[iid] = dict(pred=got, ok=bool(good))
            ok.append(per[iid]["ok"])
        out[fam] = float(np.mean(ok))
        print(f"  [{tag}] {fam:10s} n={len(ids):3d}  {100*out[fam]:5.1f}%", flush=True)
    out["ALL"] = float(np.mean([a for f, ids in ids_by_fam.items() for a in [out[f]] * len(ids)]))
    print(f"  [{tag}] {'ALL':10s}          {100*out['ALL']:5.1f}%", flush=True)
    if ckpt: model.gradient_checkpointing_enable(); model.config.use_cache = False
    json.dump(per, open(f"runs/lora_items_{tag}.json", "w"))
    return out


def main(a):
    global DATA; DATA = a.data
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl")))}
    P = np.load(a.probes, allow_pickle=True).item()

    # label budget is capped *per family* so it lines up with run/datasize.py's x-axis
    rng0 = np.random.default_rng(0)
    train_ids = []
    for f in P:
        t = list(P[f]["train_ids"])
        if a.n_per_family and a.n_per_family < len(t):
            t = [t[j] for j in rng0.choice(len(t), size=a.n_per_family, replace=False)]
        train_ids += t
    test_by_fam = {}
    for line in open(a.test):
        r = json.loads(line); test_by_fam.setdefault(r["family"], []).append(r["id"])
    print(f"train {len(train_ids)} items (the probe's own train split)   "
          f"test {sum(len(v) for v in test_by_fam.values())}\n")

    model = L.load(a.model, dev, a.load_4bit)

    if a.eval_base:
        # Tagged per run. This used to write the bare "base", i.e. runs/lora_items_base.json,
        # which every --eval-base run overwrote: only the last survived, and run/p0.py then
        # intersected its ids with a *different* dataset's. Real-chart and synthetic-chart items
        # share a naming scheme (chart_000123), so 21 ids collided by name and the intersection
        # silently shrank from 298 items to 21 rather than raising. p0.py now takes its base from
        # the canonical scored generations instead; this file is kept only for inspection.
        print("base model (no adapter):")
        evaluate(model, proc, recs, test_by_fam, dev, f"base_{a.tag}")

    # --modules splits the default target set by stack. The default list matches by *suffix*, so
    # it lands on the language model AND on the vision tower's MLPs (192 of 696 adapter tensors
    # in the 83.2% run). That confounds "fine-tuning adds representation" with "fine-tuning adds
    # a better readout": `lang` and `vis` are the exact complement halves of `both`, so their
    # results decompose it. Regex form is matched with re.fullmatch on the module path.
    SUF = "(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    TARGETS = {"both": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
               "lang": rf".*language_model\..*\.{SUF}",
               "vis":  rf".*visual\..*\.{SUF}"}
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS[a.modules])
    if a.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    # verify the split landed where it was meant to; a silently empty stack would look like a
    # result rather than a typo.
    hit = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    nv = sum("visual" in n for n in hit); nl = len(hit) - nv
    print(f"  adapted modules: language={nl}  vision={nv}   (--modules {a.modules})", flush=True)
    # `both` takes whatever the suffix list reaches. On Qwen and SmolVLM that is language plus
    # the vision MLPs; InternViT names its blocks differently and contributes nothing, so the
    # count is reported rather than asserted -- an aborted run would be worse than a documented
    # difference in what "both" covers per architecture.
    want = {"both": nl > 0, "lang": (nl > 0 and nv == 0), "vis": nv > 0}
    if not want[a.modules]:
        sys.exit(f"! --modules {a.modules} matched language={nl} vision={nv}; aborting")
    if a.modules == "both" and nv == 0:
        print("  note: no vision-tower modules matched; this run adapts the language stack only",
              flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = a.epochs * len(train_ids) // a.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1),
                                                pct_start=0.1)
    rng = np.random.default_rng(0)
    t0 = time.time(); k = 0
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(len(train_ids))
        run = []
        for n, j in enumerate(order):
            iid = train_ids[j]; r = recs[iid]
            img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
            enc = build(proc, img, r["question"], str(r["answer"])).to(dev)
            loss = model(**enc).loss / a.accum
            loss.backward(); run.append(float(loss) * a.accum)
            if (n + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                                if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); k += 1
                if k % 10 == 0:
                    print(f"  ep{ep} step {k}/{steps}  loss {np.mean(run[-80:]):.3f}  "
                          f"{time.time()-t0:.0f}s", flush=True)
        print(f"epoch {ep} done  loss {np.mean(run):.3f}  {time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % a.eval_every and ep != a.epochs - 1: continue
        res = evaluate(model, proc, recs, test_by_fam, dev, f"{a.tag}-ep{ep}")
        json.dump(dict(epoch=ep, acc=res, train_s=time.time() - t0,
                       n_train=len(train_ids), r=a.r, lr=a.lr),
                  open(a.out.replace(".json", f"_ep{ep}.json"), "w"), indent=1)
    model.save_pretrained(a.out.replace(".json", "_adapter"))
    json.dump(dict(final=res, train_s=time.time() - t0, n_train=len(train_ids),
                   epochs=a.epochs, r=a.r, lr=a.lr), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}   total train {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-4bit", action="store_true",
                   help="QLoRA: 4-bit frozen base, needed for 7B on 16 GB")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data", default="data/cal_3b")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--test", default="runs/branches6_test.jsonl")
    p.add_argument("--out", default="runs/lora_3b.json")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--eval-base", action="store_true")
    p.add_argument("--n-per-family", type=int, default=0,
                   help="cap labelled training items per family; 0 = use the whole split")
    p.add_argument("--tag", default="lora")
    p.add_argument("--modules", choices=["both", "lang", "vis"], default="both",
                   help="which stack LoRA adapts: both (default, the 83.2% reference), the "
                        "language model only, or the vision tower only")
    p.add_argument("--eval-every", type=int, default=1,
                   help="evaluate every N epochs (the last is always evaluated); an eval costs "
                        "~4 min, which dominates a small-budget run")
    main(p.parse_args())
