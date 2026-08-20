#!/bin/bash
#SBATCH --job-name=diffc_caltech_bottleneck
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/diffc_bottleneck_%j.out
#SBATCH --error=logs/diffc_bottleneck_%j.err

set -eo pipefail

WORK=/home/woody/rlvl/rlvl172v
PROJECT_DIR="$WORK/revelio/diffc_image_classification"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$WORK/revelio/DiffC_outputs"

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
echo "Starting bottleneck:0 training"

nvidia-smi

python train.py \
    --dataset_flag "dpdl-benchmark/caltech101" \
    --output_dir "$WORK/revelio/DiffC_outputs" \
    --model_name "runwayml/stable-diffusion-v1-5" \
    --diffusion_timestep 25 \
    --diffusion_layer "bottleneck:0" \
    --learning_rate 1e-4 \
    --num_epochs 90 \
    --batch_size 16 \
    --num_classes 101 \
    --prompt_type "empty" \
    --pooling_strategy "GAP" \
    --dropout_rate 0.0

echo "Bottleneck training finished"
