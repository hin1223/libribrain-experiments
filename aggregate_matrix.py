"""Aggregates every results/baseline-{X}avg/baseline-{X}avg-hpo-{seed}-test{Y}.json
file into a compact per-(train level, test level) summary: mean, SE, n.

Run on ARC (where the files live):
    python aggregate_matrix.py > matrix_summary.json
"""
import glob
import json
import re
import statistics as st
from collections import defaultdict

RESULTS_ROOT = "/data/engs-pnpl-hl/results"

fname_re = re.compile(r"baseline-(\d+)avg-hpo-(\d+)-test(\d+)\.json$")

cells = defaultdict(list)  # (X, Y) -> [f1_macro, ...]

paths = glob.glob(f"{RESULTS_ROOT}/baseline-*avg/baseline-*avg-hpo-*-test*.json")
for p in paths:
    m = fname_re.search(p)
    if not m:
        continue
    X, seed, Y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        continue
    entry = data.get(str(Y))
    if entry is None:
        continue
    cells[(X, Y)].append(entry["f1_macro"])

summary = {}
for (X, Y), vals in cells.items():
    m = st.mean(vals)
    se = st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    summary.setdefault(str(X), {})[str(Y)] = {"mean": round(m, 4), "se": round(se, 4), "n": len(vals)}

print(json.dumps(summary, indent=2, sort_keys=True))
