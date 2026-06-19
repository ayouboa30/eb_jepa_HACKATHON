#!/bin/bash
#SBATCH --job-name=jepa_ssl_long
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/work/vivatech-equipe7/%u/logs/jepa_ssl_long_%j.out

echo "Job started on $(hostname) at $(date)"
cd /lustre/work/vivatech-equipe7/$USER/eb_jepa
source env.sh

uv run python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml

echo "Job finished at $(date)"
