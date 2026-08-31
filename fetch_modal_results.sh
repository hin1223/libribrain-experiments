#!/bin/bash
# Downloads results/<config>/ one config directory at a time, so a single
# problematic subdirectory (triggering modal's "[Errno 21] Is a directory"
# bug on the bulk download) can't block the rest. Continues past failures
# and prints a summary of what failed at the end for a targeted retry.
set -u
cd ~/Desktop/Python/libribrain-experiments
mkdir -p modal_backup_results
FAILED=()

CONFIGS=(
  baseline-1avg baseline-5avg baseline-10avg baseline-15avg baseline-20avg
  baseline-25avg baseline-30avg baseline-40avg baseline-50avg baseline-60avg
  baseline-75avg baseline-85avg baseline-100avg baseline-125avg
  baseline-150avg baseline-200avg
  student-50avg student-50avg-stochastic student-50avg-stochastic-nofilm
  student-50avg-scheduled student-50avg-scheduled-nofilm
  student-10avg student-10avg-stochastic student-10avg-stochastic-nofilm
  student-10avg-scheduled student-10avg-scheduled-nofilm
  student-25avg student-25avg-stochastic student-25avg-stochastic-nofilm
  student-25avg-scheduled student-25avg-scheduled-nofilm
  student-75avg student-75avg-stochastic student-75avg-stochastic-nofilm
  student-75avg-scheduled student-75avg-scheduled-nofilm
)

for CFG in "${CONFIGS[@]}"; do
  echo "### ${CFG} ###"
  rm -rf "modal_backup_results/${CFG}"
  if modal volume get libribrain-vol "results/${CFG}" "modal_backup_results/${CFG}" 2>&1; then
    echo "OK: ${CFG}"
  else
    echo "FAILED: ${CFG}"
    FAILED+=("${CFG}")
  fi
done

echo
echo "===== SUMMARY ====="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All config directories downloaded successfully."
else
  echo "Failed (${#FAILED[@]}): ${FAILED[*]}"
  echo "Retry these individually, or fall back to per-file streaming for them."
fi
