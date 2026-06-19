#!/bin/bash
#SBATCH --job-name=vae_out_of_cohort
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=/lustre/work/vivatech-equipe7/%u/logs/out_of_cohort_%j.out

echo "Job started on $(hostname) at $(date)"
cd /lustre/work/vivatech-equipe7/$USER/eb_jepa
source env.sh

uv run python scripts/test_out_of_cohort.py \
    --data-root /lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED \
    --manifest /lustre/work/vivatech-equipe7/$USER/eeg_audit/output/edf_inventory.csv \
    --vae-ckpt results/vae_checkpoint_final.pt

echo "Job finished at $(date)"
