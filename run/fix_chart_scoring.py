"""Score the chart family on the value the model read, not on whether it rounded.

The real-chart question asks for the nearest multiple of 5, and `parse_in_space` accepts only an
exact member of that space. But the model answers 73 for a bar worth 75, 29 for 30, 19 for 20 --
it is reading the chart and declining the rounding instruction. Those all scored WRONG, which
understates model accuracy and inflates probe-minus-model by exactly the amount the instruction
was disobeyed.

The probe cannot make this mistake: it emits a class from the answer space by construction. So
the comparison was between a probe restricted to legal answers and a model penalised for
producing illegal ones. Rounding the model's numeric output removes that asymmetry, and makes
the gap measurement conservative in the direction of the claim rather than generous.

Only the chart family is affected. Counting's unparsed answers ('10' where the truth is 5) are
genuine errors and are left alone.
"""
from __future__ import annotations
import glob, json, re, sys

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def rounded_ok(text, gold, lo=0.0, hi=100.0):
    m = NUM.search(text or "")
    if not m: return False
    v = min(max(float(m.group()), lo), hi)
    return str(int(round(v / 5) * 5)) == str(gold)


def main(pattern="runs/*_gen.jsonl"):
    for f in sorted(glob.glob(pattern)):
        rs = [json.loads(l) for l in open(f)]
        if not any(r.get("family") == "chart" and r.get("chance", 0) < 0.05 for r in rs):
            continue                       # only the 21-value real-chart family
        before = sum(r["gen_correct"] for r in rs if r["family"] == "chart")
        n = 0
        for r in rs:
            if r["family"] != "chart": continue
            n += 1
            if not r["gen_correct"]:
                r["gen_correct"] = rounded_ok(r.get("gen_vis"), r["answer"])
            r["scored"] = "nearest-5"
        after = sum(r["gen_correct"] for r in rs if r["family"] == "chart")
        with open(f, "w") as fh:
            for r in rs: fh.write(json.dumps(r) + "\n")
        print(f"  {f:36s} chart {100*before/n:5.1f}% -> {100*after/n:5.1f}%  (n={n})")


if __name__ == "__main__":
    main(*sys.argv[1:])
