#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --job-name=training_diffusion_ts
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail

mkdir -p jobs
source .venv/bin/activate
srun python3 -m scripts.training.train_diffusion_ts
