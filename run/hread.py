"""H-read: a readout parameterised *through* the model's own output head.

The five readouts in run/readout.py all learn `R^2048 -> R^|A|` from scratch and throw the LM
head away; `fuse` mixes probe and model logits at the output, which is not the same thing. That
is the one structural advantage LoRA has and the probe never got: LoRA inherits a head that
already knows what "45" is, so it only has to learn *where to look*, while the probe has to
learn an 18-way output vocabulary from one or two examples per class. §6 of the results doc
shows exactly that signature -- the probe's deficit is largest at n=20 (-16.7) and shrinks to
-8.0 at n=165.

H-read keeps the frozen final hidden state h and the frozen unembedding W_U, and learns only a
low-rank correction inside the model's own output space:

    logits_p = W_U[cand_p] (h + (alpha/r) h A_p B_p)          A_p: D x r,  B_p: r x D
    score(a) = sum_p  log_softmax(logits_p)[a_p]

Two properties fall out of the parameterisation and both matter:

  * B_p initialised to zero means H-read *starts* at the frozen head's own restricted decode.
    Before it sees a single label it is the base model, where B-read is at chance. That is the
    whole low-budget argument.
  * Answers are digit-factored by the tokeniser ('45' -> ['4','5']; chart's 18 answers share
    only 9 first tokens), so a position-1-only readout cannot separate them. Both positions are
    read from the *same* h by two independent maps, so the joint is representable and inference
    still costs one forward pass -- 1.00x, cheaper than B-read's 1.01x, and equal to merged LoRA.

Two ways to tie, and the difference is not cosmetic:

  `pos`   one map per answer position, scored against W_U rows for the tokens at that position.
          Exact for single-token answer sets (spatial inits at 76.0%, the model's own accuracy)
          but wrong for mixed-length ones: position 2 is read from the *prompt-final* state, and
          the frozen head has no business decoding a second digit from it. Counting inits at
          9.3%, below its 11% chance, which is the signature of a basis the head cannot supply.

  `proto` one map, scored against a per-answer prototype e_a = mean of W_U rows over the answer's
          tokens. '45' and '40' get distinct but semantically adjacent class vectors, so the head
          supplies the class structure for multi-token answers too, at one position.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FINAL = -1          # head-tying is only meaningful at the layer W_U consumes


def load_wu(model_id, cache=None):
    """The unembedding. Qwen2.5-VL ties embeddings, so W_U is model.embed_tokens.weight."""
    from safetensors import safe_open
    import glob
    snap = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*/"))
    if not snap: sys.exit(f"! no local snapshot for {model_id}")
    idx = json.load(open(snap[0] + "model.safetensors.index.json"))["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
    with safe_open(snap[0] + idx[key], framework="pt") as f:
        return f.get_tensor(key).float()


def answer_tokens(tok, answers, eos):
    """Token id per answer per position, right-padded with eos. Returns [n_ans, L]."""
    ids = [tok.encode(a, add_special_tokens=False) for a in answers]
    L = max(len(i) for i in ids)
    return np.array([i + [eos] * (L - len(i)) for i in ids]), L


def prototypes(W, tgt, eos):
    """Per-answer class vector: the mean of its answer tokens' unembedding rows, padding
    excluded. The head supplies the geometry -- '45' lands near '40' -- instead of |A|
    independent columns learned from one or two examples each."""
    E = []
    for row in tgt:
        t = [i for i in row if i != eos] or [row[0]]
        E.append(W[t].mean(0))
    return torch.stack(E)


class HRead(torch.nn.Module):
    """Low-rank correction into a frozen head, one map per answer position."""

    def __init__(self, D, r, L, alpha=None):
        super().__init__()
        self.s = (alpha or 2 * r) / r
        self.A = torch.nn.Parameter(torch.randn(L, D, r) / D ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(L, r, D))       # zero -> starts as the base model
        self.b = torch.nn.ParameterList()                       # per-candidate bias, added lazily

    def forward(self, h, W):
        """h [n, D]; W list of L tensors [|cand_p|, D]. Returns list of L logit tensors."""
        if len(self.b) == 0:
            for w in W: self.b.append(torch.nn.Parameter(torch.zeros(len(w))))
        return [(h + self.s * (h @ self.A[p]) @ self.B[p]) @ W[p].T + self.b[p]
                for p in range(len(W))]


def scorer(m, W, tgt, eos, mode):
    """Close over the frozen head and return h -> [n, n_ans] answer scores."""
    if mode == "proto":
        E = prototypes(W, tgt, eos)
        return lambda h: m(h, [E])[0]
    L = tgt.shape[1]
    cand = [np.unique(tgt[:, p]) for p in range(L)]
    Wc = [W[c] for c in cand]
    pos = [torch.as_tensor(np.searchsorted(cand[p], tgt[:, p])) for p in range(L)]  # ans -> col
    def f(h):
        lg = m(h, Wc)
        return sum(torch.log_softmax(lg[p], -1)[:, pos[p]] for p in range(L))
    return f


def fit(X, y, Xv, yv, W, tgt, r, eos, mode, epochs=400, lr=3e-3, wd=1e-3, seed=0):
    """Fit one family. Training uses the same scoring rule as inference, so the objective and
    the reported metric cannot drift apart."""
    torch.manual_seed(seed)
    L = 1 if mode == "proto" else tgt.shape[1]
    m = HRead(X.shape[1], r, L)
    scores = scorer(m, W, tgt, eos, mode)
    with torch.no_grad(): scores(X[:1])                 # materialise the lazy bias
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = -1.0, None, 0
    for e in range(epochs):
        m.train(); opt.zero_grad()
        torch.nn.functional.cross_entropy(scores(X), y).backward()
        opt.step()
        if e % 5: continue
        m.eval()
        with torch.no_grad():
            acc = float((scores(Xv).argmax(1) == yv).float().mean())
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 12: break
    m.load_state_dict(best_state)
    return m, best, scores


def main(a):
    V = np.load(a.states)["vis"]
    meta = json.load(open(a.states.replace(".npz", "_meta.json")))
    P = np.load(a.probes, allow_pickle=True).item()
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(a.model).tokenizer
    W = load_wu(a.model)
    print(f"states {V.shape}  W_U {tuple(W.shape)}  head-tied at layer {V.shape[1]-1}\n")

    rng = np.random.default_rng(a.seed)
    out, tot_h, tot_b, tot_i, n_tot = {}, 0, 0, 0, 0
    for f in P:
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        ans = np.array([str(meta[i]["answer"]) for i in idx])
        # same >=8-member class filter fit_probes.py applies, so the splits line up
        keep = np.array([c for c in range(len(ans)) if (ans == ans[c]).sum() >= 8])
        idx, ans = idx[keep], ans[keep]
        classes = sorted(set(ans))
        cls = {c: j for j, c in enumerate(classes)}
        tgt, L = answer_tokens(tok, classes, tok.eos_token_id)

        ids = np.array([meta[i]["id"] for i in idx])
        tr_all = np.where(np.isin(ids, P[f]["train_ids"]))[0]
        te = np.where(np.isin(ids, P[f]["test_ids"]))[0]
        cal = np.setdiff1d(np.arange(len(ids)), np.concatenate([tr_all, te]))
        tr = tr_all if not a.n else tr_all[rng.choice(len(tr_all), min(a.n, len(tr_all)), False)]

        Xa = torch.as_tensor(V[idx, FINAL].astype(np.float32))
        ya = torch.as_tensor(np.array([cls[c] for c in ans]))

        # r chosen on the calibration split -- never on test
        best = (-1, None, None)
        for r in a.ranks:
            m, ca, sc = fit(Xa[tr], ya[tr], Xa[cal], ya[cal], W, tgt, r, tok.eos_token_id,
                            a.mode, epochs=a.epochs, lr=a.lr, seed=a.seed)
            if ca > best[0]: best = (ca, r, (m, sc))
        cal_acc, r_sel, (m, sc) = best
        with torch.no_grad():
            h_acc = float((sc(Xa[te]).argmax(1) == ya[te]).float().mean())
            m0 = HRead(Xa.shape[1], r_sel, 1 if a.mode == "proto" else L)   # B=0: frozen head
            i_acc = float((scorer(m0, W, tgt, tok.eos_token_id, a.mode)(Xa[te]).argmax(1)
                           == ya[te]).float().mean())

        # matched B-read on the identical items, so every row has its own control
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(64, len(tr) - 1), random_state=0),
                             LogisticRegression(max_iter=1000, C=0.5))
        Xn = V[idx, FINAL].astype(np.float32)
        pipe.fit(Xn[tr], ans[tr])
        b_acc = float(pipe.score(Xn[te], ans[te]))

        seen = set(ya[tr].tolist())
        un = torch.as_tensor([int(c.item() not in seen) for c in ya[te]]).bool()
        with torch.no_grad(): hp = sc(Xa[te]).argmax(1)
        bp = np.array([cls[c] for c in pipe.predict(Xn[te])])
        okh, okb = (hp == ya[te]).numpy(), (bp == ya[te].numpy())
        sp = dict(n_unseen=int(un.sum()),
                  h_unseen=float(okh[un].mean()) if un.any() else float("nan"),
                  b_unseen=float(okb[un].mean()) if un.any() else float("nan"),
                  h_seen=float(okh[~un].mean()), b_seen=float(okb[~un].mean()))
        print(f"    unseen-class items {sp['n_unseen']:3d}/{len(te)}   "
              f"H-read {100*sp['h_unseen']:5.1f}%  B-read {100*sp['b_unseen']:5.1f}%   "
              f"| seen: H {100*sp['h_seen']:5.1f}%  B {100*sp['b_seen']:5.1f}%")
        out[f] = dict(mode=a.mode, split=sp, n_train=len(tr), n_test=len(te), classes=len(classes), L=int(L),
                      r=r_sel, cal=cal_acc, hread=h_acc, bread=b_acc, init=i_acc)
        tot_h += h_acc * len(te); tot_b += b_acc * len(te); tot_i += i_acc * len(te)
        n_tot += len(te)
        print(f"  {f:10s} n_tr={len(tr):3d} |A|={len(classes):2d} L={L} r={r_sel:2d}  "
              f"init {100*i_acc:5.1f}%   H-read {100*h_acc:5.1f}%   B-read {100*b_acc:5.1f}%")

    out["ALL"] = dict(hread=tot_h / n_tot, bread=tot_b / n_tot, init=tot_i / n_tot, n=n_tot)
    print(f"\n  {'ALL':10s} n={n_tot:3d}                    init {100*tot_i/n_tot:5.1f}%   "
          f"H-read {100*tot_h/n_tot:5.1f}%   B-read {100*tot_b/n_tot:5.1f}%")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--states", default="runs/states_3b.npz")
    p.add_argument("--probes", default="runs/probes_3b.npy")
    p.add_argument("--out", default="runs/hread_3b.json")
    p.add_argument("--n", type=int, default=0, help="labels per family; 0 = the whole train split")
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["pos", "proto"], default="proto",
                   help="how the answer set is tied to the head: one map per answer position, "
                        "or one map against per-answer prototypes (default)")
    main(p.parse_args())
