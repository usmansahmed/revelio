#!/bin/bash
#SBATCH --job-name=bestcase_intervention_generation
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/bestcase_intervention_generation_%j.out
#SBATCH --error=logs/bestcase_intervention_generation_%j.err

set -eo pipefail

WORK=/home/woody/rlvl/rlvl172v
PROJECT_DIR="$WORK/revelio/diffc_image_classification"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$WORK/revelio/causal_results/image512/best_case_generation"

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

python bestcase_intervention_generation.py \
  --diffc-dir /home/woody/rlvl/rlvl172v/revelio/diffc_image_classification \
  --sd-ksae-dir /home/woody/rlvl/rlvl172v/revelio/SD-kSAE \
  --diffc-checkpoint /home/woody/rlvl/rlvl172v/revelio/DiffC_outputs/image512/timm-oxford-iiit-pet/runwayml-stable-diffusion-v1-5/diffusion_step_25/layer_up_ft:1/prompt_empty/pool_GAP/dropout_0.0/best_classifier.pt \
  --ksae-checkpoint /home/woody/rlvl/rlvl172v/revelio/SD-kSAE/Checkpoints/image512/ihny7k5a/final_k_sparse_autoencoder_/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/oxfordpet/SDv1-5/timestep_25/up_blocks_1/image512_10_up_blocks_1_81920.pt \
  --reference-cache /home/woody/rlvl/rlvl172v/revelio/causal_results/image512/sufficiency/reference_activation_medians.pt \
  --output-dir /home/woody/rlvl/rlvl172v/revelio/causal_results/image512/best_case_generation \
  --dataset-flag timm/oxford-iiit-pet \
  --model-name runwayml/stable-diffusion-v1-5 \
  --diffusion-timestep 25 \
  --diffusion-layer up_ft:1 \
  --diffc-dropout-rate 0.0 \
  --min-purity 0.8 \
  --min-valid 10 \
  --control-min-purity 0.8 \
  --control-min-valid 10 \
  --min-reference-count 5 \
  --insertion-feature-ranking purity_reference_activation \
  --target-classes all \
  --max-source-images-per-target 64 \
  --max-insert-per-image 3 \
  --insertion-scales 0.5,1.0,1.5,2.0 \
  --classifier-chunk-size 8 \
  --require-originally-correct \
  --require-no-target-features-active \
  --min-margin-difference 0.0 \
  --best-cases-per-target 2 \
  --top-n 10 \
  --stats-batch-size 8 \
  --generation-dtype float16 \
  --difference-map-scale 10.0 \
  --seed 42 \
  --noise-seed 42