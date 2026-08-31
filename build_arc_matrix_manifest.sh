#!/bin/bash
# Generates manifest.txt: CONFIG_NAME SEED TEST_LEVEL
#
# One line per (existing baseline-Xavg checkpoint, seed, test level Y) with
# Y <= X — the full lower-triangular-plus-diagonal "train X, test Y" transfer
# matrix (Y=X included, evaluated via the same evaluate_averaging harness as
# every other cell rather than relying on the training run's own test metric),
# using only checkpoints that are already trained (no new training runs).
# Skips any (X, seed, Y) whose output json already exists (e.g. the Y=50
# column, already computed for most levels). Sorted seed-ascending first so an array
# job submitted in manifest order sweeps breadth (all X, all Y) for seed 0
# before deepening into seed 1, 2, ...
set -e
cd /data/engs-pnpl-hl/libribrain-experiments

LEVELS=(1 5 10 15 20 25 30 40 50 60 75 85 100 125 150 200)
CKPT_ROOT=/data/engs-pnpl-hl/checkpoints
RESULTS_ROOT=/data/engs-pnpl-hl/results
OUT=manifest.txt
> "$OUT"

for X in "${LEVELS[@]}"; do
  CONFIG_NAME="baseline-${X}avg"
  CFG_DIR="${CKPT_ROOT}/${CONFIG_NAME}"
  [ -d "$CFG_DIR" ] || continue
  for RUN_DIR in "${CFG_DIR}"/${CONFIG_NAME}-hpo-*; do
    [ -d "$RUN_DIR" ] || continue
    ls "${RUN_DIR}"/best-*.ckpt >/dev/null 2>&1 || continue
    SEED=$(basename "$RUN_DIR" | sed "s/^${CONFIG_NAME}-hpo-//")
    for Y in "${LEVELS[@]}"; do
      [ "$Y" -le "$X" ] || continue
      OUTFILE="${RESULTS_ROOT}/${CONFIG_NAME}/${CONFIG_NAME}-hpo-${SEED}-test${Y}.json"
      [ -f "$OUTFILE" ] && continue
      echo "${CONFIG_NAME} ${SEED} ${Y}" >> "$OUT"
    done
  done
done

sort -k2,2n -k1,1V -k3,3n -o "$OUT" "$OUT"
echo "Generated $(wc -l < "$OUT") jobs -> $OUT"
