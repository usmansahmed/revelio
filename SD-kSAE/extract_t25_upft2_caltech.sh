#!/bin/bash -l
#SBATCH --job-name=extract_t25_upft2_caltech
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:30:00
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

echo "Starting feature extraction..."

python extract_feature.py \
  --dataset_name dpdl-benchmark/caltech101 \
  --model_name runwayml/stable-diffusion-v1-5 \
  --timestep 25 \
  --block_name up_blocks[2] \
  --image_size 256 \
  --max_batch_size 32 \
  --save_path $WORK/revelio/SD-kSAE/caltech101/SDv1-5/timestep_25/up_blocks_2

echo "Feature extraction completed."
