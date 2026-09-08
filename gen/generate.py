"""Exact-label item generators for the four LRC task families.

Architecture: every item is a **spec** (pure data: positions, colours, values, swaps) plus a
pure **render** of that spec. Three consequences, all of which the experiment needs:

  * the label is derived from the spec by an independent function, never from the drawing
    code, so `verify.py` can re-derive it and catch a renderer that lies;
  * a counterfactual is a one-field edit of the spec re-rendered, so the pair differs by a
    minimal, *stated* change and nothing else — including the background, which is redrawn
    from the same seed;
  * the whole set regenerates bit-exactly from (master seed, config, this file's hash).

Each item carries the attribute label, the answer space (so per-item chance is computable),
referent boxes where meaningful, a difficulty vector, and a prior-answerability arm holding
the answer space fixed across arms.

    python gen/generate.py --out data/pilot --n 200 --seed 20260905
"""
from __future__ import annotations
import argparse, hashlib, io, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = H = 448
FAMILIES = ("counting", "spatial", "chart", "tracking")
FAMILY_ID = {f: i for i, f in enumerate(FAMILIES)}

PALETTE = {"red": (211, 47, 47), "blue": (25, 87, 190), "green": (46, 139, 66),
           "yellow": (222, 178, 20), "purple": (123, 66, 171), "orange": (223, 116, 24)}
SHAPES = ("circle", "square", "triangle")
COUNT_SPACE = [str(i) for i in range(13)]
COUNT_CANON, COUNT_ANTI = (2, 6), (8, 13)      # same answer space, different prior mass
SPATIAL_SPACE = ["left", "right", "above", "below"]
# Arms manipulate which *class* of relation is true, not which answer. Horizontal
# relations are the high-frequency class in VQA training data; vertical the low.
# Each arm admits two answers, so the arm is not recoverable from a single answer,
# and the answer space is identical (4 options) in both arms.


def rng_for(master, family, idx):
    return np.random.default_rng(np.random.SeedSequence([master, FAMILY_ID[family], idx]))


def _font(size):
    try:    return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


def _background(seed, w=W, h=H):
    r = np.random.default_rng(seed)
    bg = Image.fromarray(np.asarray(r.integers(150, 230, size=(2, 2, 3)), dtype=np.uint8),
                         "RGB").resize((w, h), Image.BICUBIC)
    noise = r.normal(0, 7, size=(h, w, 1)).repeat(3, axis=2)
    arr = np.clip(np.asarray(bg, np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(0.4))


def _shape(d, kind, box, fill, outline=(40, 40, 40)):
    x0, y0, x1, y1 = box
    if kind == "circle":   d.ellipse(box, fill=fill, outline=outline, width=2)
    elif kind == "square": d.rectangle(box, fill=fill, outline=outline, width=2)
    else:                  d.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)],
                                     fill=fill, outline=outline)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _place(rng, n, size_rng, max_overlap, tries=240):
    boxes = []
    for _ in range(n):
        for _t in range(tries):
            s = int(rng.integers(*size_rng))
            x0 = int(rng.integers(4, W - s - 4)); y0 = int(rng.integers(4, H - s - 4))
            b = [x0, y0, x0 + s, y0 + s]
            if all(_iou(b, o) <= max_overlap for o in boxes):
                boxes.append(b); break
        else: return []
    return boxes

# ================================================================= counting
LEVELS = 5   # 0 = easiest, 4 = hardest; calibrated per family by run/calibrate.py

def spec_counting(rng, arm, level):
    hard = level >= 3
    color = str(rng.choice(list(PALETTE))); shape = str(rng.choice(SHAPES))
    lo, hi = COUNT_CANON if arm == "canonical" else COUNT_ANTI
    k = int(rng.integers(lo, hi))
    # Sample the TOTAL number of objects from an arm-independent distribution, then split it
    # into k targets and the rest distractors. Without this the anti arm carries both a
    # different prior and a heavier perceptual load, and the two are not separable.
    dist_sim = [0.0, 0.35, 0.7, 1.0, 1.0][level]     # P(distractor shares one attribute)
    base_tot = 8 + 3 * level
    total = int(rng.integers(base_tot, base_tot + 6))
    n_d = max(2, total - k)
    smin = 62 - 7 * level
    size_rng = (smin, smin + 16)
    boxes = _place(rng, k + n_d, size_rng, 0.02 + 0.05 * level)
    if not boxes: return None
    others = [c for c in PALETTE if c != color]
    oshapes = [s2 for s2 in SHAPES if s2 != shape]
    dist = []
    for b in boxes[k:]:
        if rng.random() >= dist_sim:                 # shares neither attribute: pop-out
            dist.append([b, str(rng.choice(others)), str(rng.choice(oshapes))])
        elif rng.random() < 0.5:                     # shares shape only
            dist.append([b, str(rng.choice(others)), shape])
        else:                                        # shares colour only
            dist.append([b, color, str(rng.choice(oshapes))])
    return dict(family="counting", bg=int(rng.integers(1 << 31)), color=color, shape=shape,
                targets=boxes[:k], distractors=dist, hard=bool(hard), level=int(level),
                size_min=size_rng[0], n_d=n_d, dist_sim=float(dist_sim))

def label_counting(s):
    k = len(s["targets"])
    return dict(answer=str(k), answer_space=COUNT_SPACE,
                attribute=dict(count=k, color=s["color"], shape=s["shape"], n_distractors=len(s["distractors"])),
                referents=[list(map(int, b)) for b in s["targets"]],
                question=f"How many {s['color']} {s['shape']}s are in the image? "
                         f"Answer with a number from 0 to 12.")

def render_counting(s):
    img = _background(s["bg"]); d = ImageDraw.Draw(img, "RGBA")
    for b in s["targets"]: _shape(d, s["shape"], b, PALETTE[s["color"]])
    for b, c, sh in s["distractors"]: _shape(d, sh, b, PALETTE[c])
    return img

def cf_counting(s):
    if len(s["targets"]) == 0: return None
    t = dict(s); t["targets"] = s["targets"][:-1]; return t

# ================================================================= spatial
def spec_spatial(rng, arm, level):
    hard = level >= 3
    ca, cb = [str(x) for x in rng.choice(list(PALETTE), size=2, replace=False)]
    sa, sb = [str(x) for x in rng.choice(SHAPES, size=2, replace=True)]
    horizontal = (arm == "canonical")
    a_first = bool(rng.random() < 0.5)          # which side/level object A takes
    g0 = max(8, 190 - 44 * level)
    gap = int(rng.integers(g0, g0 + 34))
    sz = int(rng.integers(46, 70)) - 5 * level; c = W // 2
    lo, hi = c - gap // 2 - sz, c + gap // 2
    if horizontal:
        y = int(rng.integers(120, H - 120)); xa, xb = (lo, hi) if a_first else (hi, lo)
        ba, bb = [xa, y, xa + sz, y + sz], [xb, y, xb + sz, y + sz]
    else:
        x = int(rng.integers(120, W - 120)); ya, yb = (lo, hi) if a_first else (hi, lo)
        ba, bb = [x, ya, x + sz, ya + sz], [x, yb, x + sz, yb + sz]
    # Clutter may never reproduce a target's (colour, shape) pair: a second "purple circle"
    # makes the question referentially ambiguous, and the model's failure would then be
    # scored as a perception error when the item simply has no unique answer.
    taken = {(ca, sa), (cb, sb)}
    clutter = []
    if level > 0:
        # Distractors that share exactly one attribute with a target force the referent to be
        # resolved before the relation can be read. Gap and clutter-count alone topped out at
        # 25% error, because two isolated salient objects make the relation trivially visible.
        near = [(ca, s2) for s2 in SHAPES if (ca, s2) not in taken] + \
               [(c2, sa) for c2 in PALETTE if (c2, sa) not in taken] + \
               [(cb, s2) for s2 in SHAPES if (cb, s2) not in taken] + \
               [(c2, sb) for c2 in PALETTE if (c2, sb) not in taken]
        for b in _place(rng, 3 * level, (max(22, sz - 8 * level), max(30, sz - 4 * level)), 0.05):
            c, sh = near[int(rng.integers(0, len(near)))] if rng.random() < 0.8 else \
                    (str(rng.choice(list(PALETTE))), str(rng.choice(SHAPES)))
            if (c, sh) not in taken:
                clutter.append([b, c, sh])
    return dict(family="spatial", bg=int(rng.integers(1 << 31)), ca=ca, sa=sa, cb=cb, sb=sb,
                ba=ba, bb=bb, clutter=clutter, hard=bool(hard), level=int(level), gap=gap,
                axis="horizontal" if horizontal else "vertical")

def label_spatial(s):
    if s["axis"] == "horizontal":
        ans = "left" if s["ba"][0] < s["bb"][0] else "right"
    else:
        ans = "above" if s["ba"][1] < s["bb"][1] else "below"
    return dict(answer=ans, answer_space=SPATIAL_SPACE,
                attribute=dict(relation=ans, axis=s["axis"], a=f"{s['ca']} {s['sa']}",
                               b=f"{s['cb']} {s['sb']}", ax=s["ba"][0], ay=s["ba"][1],
                               bx=s["bb"][0], by=s["bb"][1], gap=s["gap"]),
                referents=[s["ba"], s["bb"]],
                question=f"Where is the {s['ca']} {s['sa']} relative to the {s['cb']} {s['sb']}? "
                         f"Answer with one word: left, right, above, or below.")

def render_spatial(s):
    img = _background(s["bg"]); d = ImageDraw.Draw(img, "RGBA")
    for b, c, sh in s["clutter"]: _shape(d, sh, b, PALETTE[c])
    _shape(d, s["sa"], s["ba"], PALETTE[s["ca"]])
    _shape(d, s["sb"], s["bb"], PALETTE[s["cb"]])
    return img

def cf_spatial(s):
    t = dict(s); t["ba"], t["bb"] = list(s["bb"]), list(s["ba"]); return t

# ================================================================= chart
# Extremum-finding is saturated: Qwen2.5-VL-3B scores 100% at every difficulty level, including
# 12 categories with five near-ties. The apparent 50% error at level 3 was an answer-parsing
# bug, not difficulty. So chart moves to the harder variant the family definition already
# names -- value read-off, which requires reading the y-axis rather than comparing bar heights.
# Difficulty is gridline and tick availability, plus the number of bars.
CHART_SPACE = [str(v) for v in range(10, 100, 5)]          # 18 options, chance 5.6%
CHART_LEVEL = [dict(n=4, grid=True,  ticks=10, ymax=100),
               dict(n=5, grid=True,  ticks=20, ymax=100),
               dict(n=6, grid=False, ticks=20, ymax=100),
               dict(n=7, grid=False, ticks=25, ymax=100),
               dict(n=8, grid=False, ticks=50, ymax=100)]

def spec_chart(rng, arm, level):
    cfg = CHART_LEVEL[level]
    n = cfg["n"]
    # the queried value is an exact multiple of 5 so the label is exact; the prior arm
    # controls whether it is also a multiple of 10, which is the a-priori likelier answer,
    # with the answer space held identical across arms
    pool10 = [v for v in range(10, 100, 10)]
    pool5  = [v for v in range(15, 100, 10)]
    target = int(rng.choice(pool10 if arm == "canonical" else pool5))
    vals = [int(rng.choice(range(10, 100, 5))) for _ in range(n)]
    q = int(rng.integers(0, n)); vals[q] = target
    return dict(family="chart", vals=vals, q=q, hard=bool(level >= 3), level=int(level),
                grid=cfg["grid"], ticks=cfg["ticks"], ymax=cfg["ymax"])

def label_chart(s):
    cats = [chr(ord("A") + i) for i in range(len(s["vals"]))]
    return dict(answer=str(s["vals"][s["q"]]), answer_space=CHART_SPACE,
                attribute=dict(value=int(s["vals"][s["q"]]), bar=cats[s["q"]],
                               values=s["vals"], n_cats=len(s["vals"])),
                referents=[],
                question=f"What is the value of bar {cats[s['q']]}? The value is a multiple of 5 "
                         f"between 10 and 95. Answer with a number only.")

def cf_chart(s):
    t = dict(s); v = list(s["vals"])
    cur = v[s["q"]]
    v[s["q"]] = cur + 10 if cur <= 80 else cur - 10
    t["vals"] = v; return t

def render_chart(s):
    v = np.asarray(s["vals"], float); cats = [chr(ord("A") + i) for i in range(len(v))]
    fig, ax = plt.subplots(figsize=(4.48, 4.48), dpi=100)
    ax.bar(cats, v, color="#4a76a8", edgecolor="#22364d", linewidth=0.8)
    ax.set_ylabel("value"); ax.set_ylim(0, s["ymax"])
    if s["ticks"]: ax.set_yticks(list(range(0, int(s["ymax"]) + 1, s["ticks"])))
    else:          ax.set_yticks([0, int(s["ymax"])])
    ax.grid(axis="y", alpha=0.35) if s["grid"] else ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.canvas.draw()
    a = np.asarray(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                   fig.canvas.tostring_argb()))
    img = Image.fromarray(np.roll(a, -1, axis=2)[..., :3].copy(), "RGB").resize((W, H), Image.LANCZOS)
    plt.close(fig)
    return img

# ================================================================= tracking
def _replay(start, swaps):
    p = start
    for i, j in swaps:
        if p == i: p = j
        elif p == j: p = i
    return p

def spec_tracking(rng, arm, level):
    """Difficulty scales cups AND swaps, because the spec space has to be larger than the
    number of items drawn from it. The previous schedule (3 cups, 1+level swaps) gives 108
    distinct specs at level 1; 300 items were drawn from it, so 65 of 75 held-out items had a
    pixel-identical twin in the probe's training split and the tracking probe scored 100% on
    those and 30% -- below its 33% chance -- on the 10 clean ones. It had memorised, not read.
    3+level cups with 2+level swaps gives 6,912 specs at level 1."""
    hard = level >= 3
    n_cups = 3 + level
    n_swaps = 2 + level
    for _ in range(80):
        start = int(rng.integers(0, n_cups))
        swaps = [[int(a), int(b)] for a, b in
                 (rng.choice(n_cups, size=2, replace=False) for _ in range(n_swaps))]
        if (_replay(start, swaps) == start) == (arm == "canonical"):
            return dict(family="tracking", n_cups=n_cups, start=start, swaps=swaps,
                        hard=bool(hard), level=int(level))
    return None

def label_tracking(s):
    end = _replay(s["start"], s["swaps"])
    space = [str(i + 1) for i in range(s["n_cups"])]
    return dict(answer=str(end + 1), answer_space=space,
                attribute=dict(start=s["start"] + 1, end=end + 1, swaps=s["swaps"],
                               n_cups=s["n_cups"], returned=bool(end == s["start"])),
                referents=[],
                question=f"A ball starts under one cup. Following the swaps left to right, which "
                         f"cup is the ball under at the end? Answer with a number from 1 to {s['n_cups']}.")

def render_tracking(s):
    """Panels stacked as rows, not columns. The column layout gave each panel ~89 px of width
    for four cups; the model could count the panels only 22% of the time, so a large share of
    'tracking' errors were layout-parsing errors. Rows give each panel the full 448 px."""
    n_cups, swaps = s["n_cups"], s["swaps"]
    rows = len(swaps) + 2
    ph = H // rows
    img = Image.new("RGB", (W, H), (245, 245, 247)); d = ImageDraw.Draw(img)
    f = _font(max(11, min(17, ph // 4)))
    def cups(y0, hidden, label):
        d.rectangle([0, y0, W - 1, y0 + ph - 2], fill=(255, 255, 255), outline=(150, 150, 158))
        d.text((6, y0 + 4), label, fill=(20, 20, 20), font=f)
        avail = W - 120
        step = avail // n_cups
        cw = min(step - 10, ph - 22)
        cy = y0 + ph // 2
        for c in range(n_cups):
            cx = 100 + c * step + (step - cw) // 2
            d.rectangle([cx, cy - cw // 2, cx + cw, cy + cw // 2],
                        fill=(196, 148, 92), outline=(90, 60, 30), width=2)
            d.text((cx + cw // 2 - 4, cy + cw // 2 + 1), str(c + 1), fill=(20, 20, 20), font=f)
            if hidden == c:
                r = cw // 3
                d.ellipse([cx + cw // 2 - r, cy - r, cx + cw // 2 + r, cy + r],
                          fill=(214, 60, 60), outline=(120, 20, 20), width=2)
    cups(0, s["start"], "START")
    for k, (i, j) in enumerate(swaps):
        y0 = (k + 1) * ph
        d.rectangle([0, y0, W - 1, y0 + ph - 2], fill=(238, 240, 246), outline=(150, 150, 158))
        d.text((6, y0 + ph // 2 - 7), f"step {k+1}:  swap cup {i+1} and cup {j+1}",
               fill=(20, 20, 20), font=f)
    cups((len(swaps) + 1) * ph, None, "END  (ball hidden)")
    return img

def cf_tracking(s):
    if len(s["swaps"]) < 2: return None
    t = dict(s); t["swaps"] = s["swaps"][:-1]; return t


SPEC   = dict(counting=spec_counting, spatial=spec_spatial, chart=spec_chart, tracking=spec_tracking)
LABEL  = dict(counting=label_counting, spatial=label_spatial, chart=label_chart, tracking=label_tracking)
RENDER = dict(counting=render_counting, spatial=render_spatial, chart=render_chart, tracking=render_tracking)
CF     = dict(counting=cf_counting, spatial=cf_spatial, chart=cf_chart, tracking=cf_tracking)
CF_KIND = dict(counting="remove_one_target", spatial="swap_positions",
               chart="shift_queried_value_by_10", tracking="drop_last_swap")


def _record(iid, fam, spec, path, arm, cf_of=None):
    lab = LABEL[fam](spec)
    return dict(id=iid, family=fam, image=path, question=lab["question"], answer=lab["answer"],
                answer_space=lab["answer_space"], chance=1.0 / len(lab["answer_space"]),
                attribute=lab["attribute"], referents=lab["referents"],
                difficulty=dict(hard=spec.get("hard", False), level=spec["level"],
                                **{k: v for k, v in spec.items()
                                   if k in ("gap", "n_d", "size_min", "n_cups")}),
                prior_arm=arm, cf_kind=CF_KIND[fam], cf_of=cf_of, spec=spec)


def build(out, n_per_cell, master, families=FAMILIES, with_cf=True, levels="sweep"):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    manifest, stats = [], {}
    for fam in families:
        made = idx = tries = n_cf = n_dup = 0
        seen = set()               # rendered-image hashes: two specs that draw the same picture
                                   # are the same item, however different the dicts look
        cnt = dict(canonical=0, anti=0, hard=0, easy=0)
        while made < n_per_cell and tries < n_per_cell * 40:
            tries += 1
            rng = rng_for(master, fam, idx); idx += 1
            arm = "canonical" if made % 2 == 0 else "anti"
            lv = levels[fam] if isinstance(levels, dict) else int(rng.integers(0, LEVELS)) \
                 if levels == "sweep" else int(levels)
            if levels == "sweep": lv = made % LEVELS
            spec = SPEC[fam](rng, arm, lv)
            if spec is None: continue
            img = RENDER[fam](spec)
            buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
            ih = hashlib.sha256(buf.getvalue()).hexdigest()
            if ih in seen:                      # a duplicate image is a train/test leak waiting
                n_dup += 1; continue            # to happen -- reject it, and say so below
            seen.add(ih)
            iid = f"{fam}_{made:06d}"
            p = os.path.join("images", iid + ".png")
            img.save(os.path.join(out, p), optimize=True)
            manifest.append(_record(iid, fam, spec, p, arm))
            cnt[arm] += 1; cnt["hard" if spec["level"] >= 3 else "easy"] += 1
            if with_cf:
                cs = CF[fam](spec)
                if cs is not None:
                    cid = iid + "_cf"; cp = os.path.join("images", cid + ".png")
                    RENDER[fam](cs).save(os.path.join(out, cp), optimize=True)
                    manifest.append(_record(cid, fam, cs, cp, arm, cf_of=iid)); n_cf += 1
            made += 1
        stats[fam] = dict(made=made, tries=tries, cf=n_cf, dup_rejected=n_dup,
                          distinct_space_hit=bool(made < n_per_cell), **cnt)
        if made < n_per_cell:
            print(f"  !! {fam}: only {made}/{n_per_cell} distinct items exist at this level "
                  f"({n_dup} duplicate renders rejected). The spec space is smaller than the "
                  f"requested sample; raise the difficulty level or lower --n.")
    with open(os.path.join(out, "manifest.jsonl"), "w") as f:
        for r in manifest: f.write(json.dumps(r) + "\n")
    cfg = dict(master_seed=master, n_per_cell=n_per_cell, families=list(families), canvas=[W, H],
               with_cf=with_cf, levels=levels,
               generator_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    return manifest, stats, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pilot"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--no-cf", action="store_true")
    ap.add_argument("--levels", default="sweep",
                    help='"sweep" (cycle 0-4), an int, or a json dict {family: level}')
    a = ap.parse_args()
    lv = a.levels
    if lv not in ("sweep",):
        lv = json.loads(lv) if lv.strip().startswith("{") else int(lv)
    man, st, cfg = build(a.out, a.n, a.seed, tuple(a.families), with_cf=not a.no_cf, levels=lv)
    print(f"wrote {len(man)} records ({sum(s['made'] for s in st.values())} base + "
          f"{sum(s['cf'] for s in st.values())} counterfactual) -> {a.out}")
    for f, s in st.items():
        print(f"  {f:9s} base={s['made']:4d} cf={s['cf']:4d} tries={s['tries']:4d} "
              f"canon={s['canonical']:3d} anti={s['anti']:3d} hard={s['hard']:3d} easy={s['easy']:3d}")
    print("  generator sha:", cfg["generator_sha256"])
