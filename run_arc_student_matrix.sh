#!/bin/bash
#SBATCH --job-name=libribrain-student-matrix
#SBATCH --partition=short
#SBATCH --time=00:06:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-g003,htc-g004,htc-g032,htc-g033,htc-g034,htc-g035,htc-g037,htc-g038
#SBATCH --output=/data/engs-pnpl-hl/logs/student-matrix-%A_%a.out
#SBATCH --error=/data/engs-pnpl-hl/logs/student-matrix-%A_%a.err

source /data/engs-pnpl-hl/miniconda3/etc/profile.d/conda.sh
conda activate libribrain

cd /data/engs-pnpl-hl/libribrain-experiments

MANIFEST=${MANIFEST:?must set MANIFEST}
OFFSET=${OFFSET:-0}
LINE_NUM=$((SLURM_ARRAY_TASK_ID + OFFSET))
LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
CONFIG_NAME=$(awk '{print $1}' <<< "$LINE")
CKPT=$(awk '{print $2}' <<< "$LINE")
TEST_LEVEL=$(awk '{print $3}' <<< "$LINE")

RUN_NAME=$(basename "$(dirname "$CKPT")")

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: ${CKPT}, aborting"
    exit 1
fi

# baseline (-abo-) runs bypass distillation entirely (ClassificationModule);
# every KD/stochastic/scheduled run needs DistillationModule (or a subclass)
if [[ "$RUN_NAME" == *-abo-* ]]; then
    MODULE=classification
else
    MODULE=distillation
fi

python -m libribrain_experiments.evaluate_averaging \
    --config configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml \
    --checkpoint "$CKPT" \
    --module $MODULE \
    --split test \
    --levels $TEST_LEVEL \
    --n-pool 200 \
    --output /data/engs-pnpl-hl/results/${CONFIG_NAME}/${RUN_NAME}-test${TEST_LEVEL}.json
