#!/bin/bash
#SBATCH --job-name=libribrain-hpo
#SBATCH --partition=medium
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-g003,htc-g004,htc-g032,htc-g033,htc-g034,htc-g035,htc-g037,htc-g038
#SBATCH --output=/data/engs-pnpl-hl/logs/hpo-%A.out
#SBATCH --error=/data/engs-pnpl-hl/logs/hpo-%A.err

source /data/engs-pnpl-hl/miniconda3/etc/profile.d/conda.sh
conda activate libribrain

cd /data/engs-pnpl-hl/libribrain-experiments

CONFIG_NAME=${CONFIG_NAME:-averaging-ablation}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-$CONFIG_NAME}

CMD="python -m libribrain_experiments.hpo \
    --config configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml \
    --search-space configs/phoneme/${CONFIG_NAME}/search-space.yaml \
    --run-index $RUN_INDEX \
    --run-name ${RUN_NAME_PREFIX} \
    --project-name libribrain-experiments"

if [ -n "$TRACK_TEST_PER_EPOCH" ]; then
    CMD="$CMD --track-test-per-epoch"
fi

eval $CMD
