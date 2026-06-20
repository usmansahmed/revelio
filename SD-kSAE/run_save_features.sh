#!/bin/bash -l
#SBATCH --job-name=revelio_save_features
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
unset SLURM_EXPORT_ENV

module load python
conda activate $WORK/conda_envs/revelio

export HF_HOME=$WORK/hf_cache
export HF_DATASETS_CACHE=$WORK/hf_cache/datasets
export TRANSFORMERS_CACHE=$WORK/hf_cache/transformers
export HUGGINGFACE_HUB_CACHE=$WORK/hf_cache/hub

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR=$WORK/wandb

mkdir -p logs
mkdir -p $WANDB_DIR

cd $WORK/revelio/SD-kSAE

python run_save_features.py
