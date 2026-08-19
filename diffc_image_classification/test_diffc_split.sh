#!/bin/bash
#SBATCH --job-name=test_diffc_split
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_diffc_split_%j.out
#SBATCH --error=logs/test_diffc_split_%j.err

set -eo pipefail

WORK=/home/woody/rlvl/rlvl172v
PROJECT_DIR="$WORK/revelio/diffc_image_classification"

mkdir -p "$PROJECT_DIR/logs"

module load python
conda activate "$WORK/conda_envs/revelio"

export HF_HOME="$WORK/hf_cache"
export HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

python test_diffc_split.py
