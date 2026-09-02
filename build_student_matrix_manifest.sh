#!/bin/bash
# Generates manifest.txt: CONFIG_NAME CKPT_PATH TEST_LEVEL
#
# Discovers every existing student-Xavg* checkpoint (all 10 KD/stochastic/
# scheduled/FiLM/nofilm categories, all levels, all seeds already trained)
# and queues an evaluate_averaging.py call at every one of the 16 test
# levels, skipping any (checkpoint, test level) pair whose output already
# exists. Unlike build_arc_matrix_manifest.sh, this discovers checkpoint
# paths directly via glob rather than reconstructing them from
# CONFIG_NAME+RUN_INDEX, since student-Xavg run names embed an alpha tag
# (e.g. student-50avg-a05-hpo-5) that the simpler baseline-Xavg naming
# doesn't have.
set -e
cd /data/engs-pnpl-hl/libribrain-experiments

LEVELS=(1 5 10 15 20 25 30 40 50 60 75 85 100 125 150 200)
CKPT_ROOT=/data/engs-pnpl-hl/checkpoints
RESULTS_ROOT=/data/engs-pnpl-hl/results
OUT=student_manifest.txt
> "$OUT"

for CKPT in $(find "$CKPT_ROOT" -path "*/student-*avg*/*-hpo-*/best-*.ckpt" 2>/dev/null); do
  RUN_DIR=$(dirname "$CKPT")
  RUN_NAME=$(basename "$RUN_DIR")
  CONFIG_DIR=$(dirname "$RUN_DIR")
  CONFIG_NAME=$(basename "$CONFIG_DIR")

  # skip if no matching config file (shouldn't happen, but be defensive)
  [ -f "configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml" ] || continue

  for Y in "${LEVELS[@]}"; do
    OUTFILE="${RESULTS_ROOT}/${CONFIG_NAME}/${RUN_NAME}-test${Y}.json"
    [ -f "$OUTFILE" ] && continue
    echo "${CONFIG_NAME} ${CKPT} ${Y}" >> "$OUT"
  done
done

echo "Generated $(wc -l < "$OUT") jobs -> $OUT"
echo "Checkpoints discovered: $(find "$CKPT_ROOT" -path "*/student-*avg*/*-hpo-*/best-*.ckpt" 2>/dev/null | wc -l)"
