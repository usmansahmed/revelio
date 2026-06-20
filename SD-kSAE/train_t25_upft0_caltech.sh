#!/bin/bash -l
#SBATCH --job-name=train_t25_upft0_caltech
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
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
mkdir -p $WORK/revelio/SD-kSAE/Checkpoints
mkdir -p $WORK/revelio/SD-kSAE/features/caltech101/SDv1-5/step25_upft0

cd $WORK/revelio/SD-kSAE

echo "Starting full k-SAE training..."

python train_ksae.py \
  --model_name $WORK/revelio/SD-kSAE/caltech101/SDv1-5/timestep_25/up_blocks_0 \
  --feature_dir $WORK/revelio/SD-kSAE/features/caltech101/SDv1-5/step25_upft0 \
  --module_name up_blocks_0 \
  --dataset_name dpdl-benchmark/caltech101 \
  --d_in 1280 \
  --expansion_factor 64 \
  --k 32 \
  --lr 0.0004 \
  --lr_scheduler_name constantwithwarmup \
  --batch_size 8192 \
  --lr_warm_up_steps 500 \
  --total_training_tokens 83886080 \
  --dead_feature_threshold 1e-6 \
  --device cuda \
  --checkpoint_path $WORK/revelio/SD-kSAE/Checkpoints \
  --dtype float32

echo "Full k-SAE training completed."
