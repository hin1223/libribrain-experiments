#!/bin/bash
#SBATCH --job-name=libribrain-traintest
#SBATCH --partition=medium
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-g003,htc-g004,htc-g032,htc-g033,htc-g034,htc-g035,htc-g037,htc-g038
#SBATCH --output=/data/engs-pnpl-hl/logs/traintest-%A.out
#SBATCH --error=/data/engs-pnpl-hl/logs/traintest-%A.err

source /data/engs-pnpl-hl/miniconda3/etc/profile.d/conda.sh
conda activate libribrain

cd /data/engs-pnpl-hl/libribrain-experiments

CONFIG_NAME=${CONFIG_NAME:?must set CONFIG_NAME}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-$CONFIG_NAME}
TEST_LEVEL=${TEST_LEVEL:-50}

python -m libribrain_experiments.hpo \
    --config configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml \
    --search-space configs/phoneme/${CONFIG_NAME}/search-space.yaml \
    --run-index $RUN_INDEX \
    --run-name ${RUN_NAME_PREFIX} \
    --project-name libribrain-experiments

RUN_NAME=${RUN_NAME_PREFIX}-hpo-${RUN_INDEX}
CKPT_DIR=/data/engs-pnpl-hl/checkpoints/${CONFIG_NAME}/${RUN_NAME}
CKPT=$(ls ${CKPT_DIR}/best-*.ckpt | head -1)

if [ -z "$CKPT" ]; then
    echo "No checkpoint found in ${CKPT_DIR}, aborting eval step"
    exit 1
fi

python -m libribrain_experiments.evaluate_averaging \
    --config configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml \
    --checkpoint "$CKPT" \
    --module classification \
    --split test \
    --levels $TEST_LEVEL \
    --n-pool 100 \
    --output /data/engs-pnpl-hl/results/${CONFIG_NAME}/${RUN_NAME}-test${TEST_LEVEL}.json
