#!/bin/bash
#SBATCH --job-name=diffc_pet_upft1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/diffc_upft1_%j.out
#SBATCH --error=logs/diffc_upft1_%j.err

set -eo pipefail

WORK=/home/woody/rlvl/rlvl172v
PROJECT_DIR="$WORK/revelio/diffc_image_classification"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$WORK/revelio/DiffC_outputs/"

module load python
conda activate "$WORK/conda_envs/revelio"

export HF_HOME="$WORK/hf_cache"
export HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR="$WORK/wandb"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Starting up_ft:1 training"

nvidia-smi

python smoke_test_dit.py

echo "up_ft:1 training finished"
