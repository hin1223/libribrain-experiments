#!/bin/bash
#SBATCH --job-name=libribrain-distill
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-g[003-004,032-038,046-049]
#SBATCH --output=/data/engs-pnpl-hl/logs/distill-%A.out
#SBATCH --error=/data/engs-pnpl-hl/logs/distill-%A.err

source /data/engs-pnpl-hl/miniconda3/etc/profile.d/conda.sh
conda activate libribrain

cd /data/engs-pnpl-hl/libribrain-experiments

python -m libribrain_experiments.distill \
    --config configs/phoneme/student-50avg-stochastic/base-config-arc.yaml \
    --search-space configs/phoneme/student-50avg-stochastic/search-space.yaml \
    --run-index $RUN_INDEX \
    --run-name student-50avg-stochastic \
    --project-name libribrain-experiments
