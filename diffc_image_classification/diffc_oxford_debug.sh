#!/bin/bash -l

#SBATCH --job-name=diffc_pet_debug
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
unset SLURM_EXPORT_ENV

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

module load python

conda activate "$WORK/conda_envs/revelio"

export HF_HOME="$WORK/hf_cache"
export HF_DATASETS_CACHE="$WORK/hf_cache/datasets"
export TRANSFORMERS_CACHE="$WORK/hf_cache/transformers"
export HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"

# Use cached models/datasets on compute node
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Avoid W&B internet problems
export WANDB_MODE=offline
export WANDB_DIR="$WORK/wandb"

mkdir -p logs
mkdir -p "$WANDB_DIR"

# Change this path if train.py is in another directory
cd "$WORK/revelio/diffc_image_classification"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"

python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# Smoke-test training
# ------------------------------------------------------------

python train.py \
    --dataset_flag "timm/oxford-iiit-pet" \
    --output_dir "$WORK/revelio/DiffC_outputs/debug" \
    --model_name runwayml/stable-diffusion-v1-5 \
    --diffusion_timestep 25 \
    --diffusion_layer "up_ft:1" \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --batch_size 2 \
    --prompt_type empty \
    --pooling_strategy GAP \
    --dropout_rate 0.5 \
    --num_classes 37

echo "End time: $(date)"
echo "Smoke test completed successfully."
