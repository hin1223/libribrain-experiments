#!/bin/bash
# Submits Baseline CE (step-matched) + KD (fixed) + Scheduled KD+FiLM training
# runs for averaging levels 80/85/90/95, via run_arc.sh (partition=short,
# time=11:58:00 — same as every other student-Xavg distill run).
#
# Usage: bash submit_new_levels.sh <first_seed> <last_seed>
#   bash submit_new_levels.sh 0 4   # first batch: 4 levels x 3 categories x 5 seeds = 60 jobs
#   bash submit_new_levels.sh 5 9   # remaining batch: another 60 jobs (120 total)
set -e
FIRST=${1:?usage: submit_new_levels.sh <first_seed> <last_seed>}
LAST=${2:?usage: submit_new_levels.sh <first_seed> <last_seed>}
LEVELS=(80 85 90 95)

for X in "${LEVELS[@]}"; do
  for SEED in $(seq "$FIRST" "$LAST"); do
    # 1. Baseline CE, step-matched (bypasses the KD loss; alpha is unused but
    #    must be set for the script's --alpha-override argument)
    CONFIG_NAME=student-${X}avg RUN_NAME_PREFIX=student-${X}avg-abo \
      ALPHA=0.5 ALPHA_TAG=05 BASELINE_ONLY=1 RUN_INDEX=${SEED} \
      sbatch run_arc.sh

    # 2. KD, fixed averaging, alpha=0.5 (config default)
    CONFIG_NAME=student-${X}avg RUN_NAME_PREFIX=student-${X}avg \
      ALPHA=0.5 ALPHA_TAG=05 RUN_INDEX=${SEED} \
      sbatch run_arc.sh

    # 3. Scheduled KD + FiLM, alpha=0.6 (config default); the search-space's
    #    temperature axis has 5 values ([1,2,3,4,6], fastest-varying), so
    #    seed*5+1 pins temperature=2.0 for every seed, matching how every
    #    other level's -scheduled sweep was run.
    SCHED_RUN_INDEX=$((SEED * 5 + 1))
    CONFIG_NAME=student-${X}avg-scheduled RUN_NAME_PREFIX=student-${X}avg-scheduled \
      ALPHA=0.6 ALPHA_TAG=06 RUN_INDEX=${SCHED_RUN_INDEX} \
      sbatch run_arc.sh
  done
done
