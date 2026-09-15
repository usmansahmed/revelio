#!/bin/bash
#SBATCH --job-name=qualitative_sae_intervention_multistep
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:00:05
#SBATCH --output=logs/qualitative_sae_intervention_multistep_%j.out
#SBATCH --error=logs/qualitative_sae_intervention_multistep_%j.err

set -eo pipefail

WORK=/home/woody/rlvl/rlvl172v
PROJECT_DIR="$WORK/revelio/diffc_image_classification"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$WORK/revelio/qualitative_interventions/keeshond_from_pug_multistep"

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

python run_qualitative_sae_intervention_multistep.py \
    --config qualitative_intervention_multistep_config.json
