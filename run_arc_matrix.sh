#!/bin/bash
#SBATCH --job-name=libribrain-matrix
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-g003,htc-g004,htc-g032,htc-g033,htc-g034,htc-g035,htc-g037,htc-g038
#SBATCH --output=/data/engs-pnpl-hl/logs/matrix-%A_%a.out
#SBATCH --error=/data/engs-pnpl-hl/logs/matrix-%A_%a.err

source /data/engs-pnpl-hl/miniconda3/etc/profile.d/conda.sh
conda activate libribrain

cd /data/engs-pnpl-hl/libribrain-experiments

MANIFEST=${MANIFEST:?must set MANIFEST}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
CONFIG_NAME=$(awk '{print $1}' <<< "$LINE")
RUN_INDEX=$(awk '{print $2}' <<< "$LINE")
TEST_LEVEL=$(awk '{print $3}' <<< "$LINE")

RUN_NAME=${CONFIG_NAME}-hpo-${RUN_INDEX}
CKPT_DIR=/data/engs-pnpl-hl/checkpoints/${CONFIG_NAME}/${RUN_NAME}
CKPT=$(ls ${CKPT_DIR}/best-*.ckpt | head -1)

if [ -z "$CKPT" ]; then
    echo "No checkpoint found in ${CKPT_DIR}, aborting"
    exit 1
fi

# n-pool fixed at 200 (>= the largest train level) so every test level in the
# sweep (always < its train level, so <= 150) fits under the pooling cap.
python -m libribrain_experiments.evaluate_averaging \
    --config configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml \
    --checkpoint "$CKPT" \
    --module classification \
    --split test \
    --levels $TEST_LEVEL \
    --n-pool 200 \
    --output /data/engs-pnpl-hl/results/${CONFIG_NAME}/${RUN_NAME}-test${TEST_LEVEL}.json
