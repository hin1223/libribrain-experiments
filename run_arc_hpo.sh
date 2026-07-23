#!/bin/bash
#SBATCH --job-name=libribrain-hpo
#SBATCH --partition=short
#SBATCH --time=11:58:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/engs-pnpl-hl/logs/hpo-%A.out
#SBATCH --error=/data/engs-pnpl-hl/logs/hpo-%A.err

source /data/engs-pnpl-hl/miniconda3/etc/profile.d/conda.sh
conda activate libribrain

cd /data/engs-pnpl-hl/libribrain-experiments

CONFIG_NAME=${CONFIG_NAME:-averaging-ablation}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-$CONFIG_NAME}

python -m libribrain_experiments.hpo \
    --config configs/phoneme/${CONFIG_NAME}/base-config-arc.yaml \
    --search-space configs/phoneme/${CONFIG_NAME}/search-space.yaml \
    --run-index $RUN_INDEX \
    --run-name ${RUN_NAME_PREFIX} \
    --project-name libribrain-experiments
