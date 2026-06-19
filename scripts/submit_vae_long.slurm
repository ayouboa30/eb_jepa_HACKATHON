#!/bin/bash
#SBATCH --job-name=vae_long
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/work/vivatech-equipe7/%u/logs/vae_long_%j.out

echo "Job started on $(hostname) at $(date)"
cd /lustre/work/vivatech-equipe7/$USER/eb_jepa
source env.sh

uv run python scripts/train_spectral_state_vae.py \
    --source edf \
    --manifest /lustre/work/vivatech-equipe7/$USER/eeg_audit/output/edf_inventory.csv \
    --steps 4000 \
    --steps-per-epoch 100 \
    --save-checkpoint results/vae_checkpoint_4k.pt \
    --save-interval 5000

echo "Job finished at $(date)"
