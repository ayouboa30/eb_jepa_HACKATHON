#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/lustre/work/vivatech-equipe7/%u/logs/train_vae_%j.out

echo "Job started on $(hostname) at $(date)"

# Get repo directory (assuming submit from eb_jepa)
cd /lustre/work/vivatech-equipe7/$USER/eb_jepa

# Source environment
source env.sh

# Ensure results and logs directories exist
mkdir -p results logs

# Run training — 200 epochs, small model (~5 min on GPU)
echo "Starting training (50 epochs = 5000 steps, small model)..."
uv run python scripts/train_spectral_state_vae.py \
    --source edf \
    --manifest /lustre/work/vivatech-equipe7/$USER/eeg_audit/output/edf_inventory.csv \
    --max-edf-files -1 \
    --batch-size 16 \
    --latent-dim 64 \
    --hidden-dim 128 \
    --steps 5000 \
    --steps-per-epoch 100 \
    --warmup-epochs 10 \
    --ramp-epochs 10 \
    --physics-min-scale 0.5 \
    --physics-max-scale 1.0 \
    --lr 3e-4 \
    --beta 1e-5 \
    --lambda-phase 0.1 \
    --lambda-bandpower 0.5 \
    --save-checkpoint results/vae_checkpoint_final.pt \
    --save-interval 1000

echo "Job finished at $(date)"
