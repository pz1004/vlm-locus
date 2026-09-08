"""Real chart images with exact bar-value counterfactuals.

Chart-value reading is where the synthetic effect is concentrated (+64 pp, 89% follow, decodable
at 98.6% from the final layer), so a real counterpart is the load-bearing test of whether that
finding is about VLMs or about our renderer. ChartQA supplies real published charts (Statista,
Pew, OWID) together with the underlying data table, which is what makes an exact edit possible.

The edit is **additive only**: a bar is raised, never lowered. Lowering one would require
reconstructing whatever sits behind it -- gridlines, background gradients -- and a reconstruction
is not an exact edit. Raising it paints a rectangle from the old top to the new top in the bar's
own colour, so every pixel outside that rectangle is bit-identical and the guard is trivial.

Charts are admitted only if the detected bar heights are an affine function of the table values
(max residual under 2% of the tallest bar). That single test rejects 3-D bars, stacked bars, log
axes, broken axes and mis-parsed tables without having to detect any of them by name.
"""
from __future__ import annotations
import csv
import numpy as np
from PIL import Image
from scipy import ndimage

GRID = 5                                        # answers are rounded to a multiple of this
VALS = [str(v) for v in range(0, 101, GRID)]    # 21 classes, cf. the synthetic family's 18


def background(A):
    c, n = np.unique(A.reshape(-1, 3), axis=0, return_counts=True)
    return c[n.argmax()]


def find_bars(im, min_w=5, min_h=8, fill=0.90):
    """Axis-aligned solid rectangles sharing a common baseline: the bars of one series.
    Returns (bars, baseline, colour), bars sorted left to right as (x0, x1, ytop, colour)."""
    A = np.asarray(im.convert("RGB")).astype(np.int16)
    bg = background(A)
    cols, cnt = np.unique(A.reshape(-1, 3), axis=0, return_counts=True)
    keep = (cnt > min_w * min_h) & (np.abs(cols - bg).sum(1) > 40)
    keep &= (cols.max(1) - cols.min(1)) > 12     # drop greys: axes, text, gridlines
    if not keep.any(): return [], None, None
    best = None
    order = np.argsort(-cnt[keep])
    for c in cols[keep][order][:12]:
        m = np.abs(A - c).sum(2) < 30
        lbl, n = ndimage.label(m)
        if not n: continue
        rects = []
        for i, sl in enumerate(ndimage.find_objects(lbl)):
            ys, xs = sl
            h, w = ys.stop - ys.start, xs.stop - xs.start
            if w < min_w or h < min_h: continue
            if (lbl[sl] == i + 1).sum() / (w * h) < fill: continue
            rects.append((xs.start, xs.stop, ys.start, ys.stop))
        if len(rects) < 3: continue
        base = {}
        for r in rects: base.setdefault(round(r[3] / 2) * 2, []).append(r)
        b, grp = max(base.items(), key=lambda kv: len(kv[1]))
        if len(grp) < 3: continue
        if best is None or len(grp) > len(best[0]):
            best = (sorted(grp), b, tuple(int(v) for v in c))
    if best is None: return [], None, None
    grp, b, c = best
    return [(r[0], r[1], r[2], c) for r in grp], b, c


def read_table(path):
    """(categories, values) for a single-series table, else (None, None)."""
    try:
        rows = list(csv.reader(open(path, encoding="utf-8", errors="ignore")))
    except OSError:
        return None, None
    rows = [r for r in rows if any(x.strip() for x in r)]
    # exactly two columns: category + one value. A wider table is a multi-series chart, where
    # bar detection finds one series and strip_labels therefore misses the other's printed
    # values -- the chart would still show numbers, and the two series would not be separable.
    if len(rows) < 4 or len(rows[0]) != 2: return None, None
    body = rows[1:]
    if set(len(r) for r in body) != {2}: return None, None
    cats, vals = [], []
    for r in body:
        try: v = float(r[1].replace(",", "").replace("%", "").strip())
        except ValueError: return None, None
        cats.append(r[0].strip()); vals.append(v)
    return cats, vals


def calibrate(bars, baseline, values, tol=0.02):
    """Fit height = a*value + b and accept only a tight fit. This is the admission test."""
    if len(bars) != len(values) or len(bars) < 3: return None
    h = np.array([baseline - t for _, _, t, _ in bars], float)
    v = np.array(values, float)
    if v.max() - v.min() < 1e-6 or h.max() - h.min() < 4: return None
    a, b = np.polyfit(v, h, 1)
    if a <= 0: return None
    if np.max(np.abs(a * v + b - h)) > max(tol * h.max(), 2.5): return None
    return float(a), float(b)


def has_data_labels(im, bars, band=24, thresh=4):
    """True if any bar has dark ink directly above its top -- a printed data label.

    Charts that print their values turn "read the bar against the axis" into OCR of a number, and
    raising such a bar destroys its label, so the counterfactual would swap the task rather than
    change the attribute: the original is answerable by reading text, the edit is not. The
    synthetic chart family prints no values, so the real one must not either. Rejecting these is
    what keeps the two families measuring the same thing.
    """
    A = np.asarray(im.convert("L")).astype(np.int16)
    H, W = A.shape
    for x0, x1, ytop, _ in bars:
        y1, y0 = max(ytop - 2, 0), max(ytop - band, 0)
        if y1 <= y0: continue
        if (A[y0:y1, x0:x1] < 120).sum() > thresh: return True
    return False


def strip_labels(im, bars, band=26, pad=6):
    """Paint over the printed value above every bar, uniformly.

    ChartQA charts print their values, which turns "read the bar against the axis" into OCR and
    makes the raised bar the only unlabelled one. Erasing the labels from EVERY bar removes both
    problems at once: the task becomes the synthetic family's task, and no bar is distinguishable
    from its neighbours. The erase happens before the counterfactual is derived, so the pair is
    still exact -- the guard compares the two images we actually use, and the only difference
    between them is the raised rectangle.

    The band is clipped to the midpoints between neighbouring bars so one bar's erasure cannot
    reach into another's, and it is filled with the image's background colour.
    """
    A = np.asarray(im.convert("RGB")).copy()
    H, W, _ = A.shape
    bg = background(A.astype(np.int16))
    xs = sorted((b[0] + b[1]) / 2 for b in bars)
    for x0, x1, ytop, _ in bars:
        c = (x0 + x1) / 2
        lo = max([m for m in xs if m < c], default=-1e9)
        hi = min([m for m in xs if m > c], default=1e9)
        a0 = int(max(0, max(x0 - pad, (lo + c) / 2 if lo > -1e8 else 0)))
        a1 = int(min(W, min(x1 + pad, (hi + c) / 2 if hi < 1e8 else W)))
        y1, y0 = max(ytop - 2, 0), max(ytop - band, 0)
        if y1 > y0 and a1 > a0: A[y0:y1, a0:a1] = bg
    return Image.fromarray(A)


def raise_bar(im, bar, baseline, a, b, new_value):
    """Paint the bar up to `new_value`. Additive only: returns (image, mask) with the mask the
    added rectangle exactly, or (None, None) if the new top would leave the image."""
    x0, x1, ytop, colour = bar
    new_top = int(round(baseline - (a * new_value + b)))
    if new_top >= ytop - 2 or new_top < 1: return None, None
    out = np.asarray(im.convert("RGB")).copy()
    out[new_top:ytop, x0:x1] = colour
    mask = np.zeros(out.shape[:2], np.uint8)
    mask[new_top:ytop, x0:x1] = 255
    return Image.fromarray(out), Image.fromarray(mask)
