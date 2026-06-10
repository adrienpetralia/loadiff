#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=calibrate_plaus_env
#SBATCH -o ./jobs/calibrate_plaus_env_%j.out
#SBATCH -e ./jobs/calibrate_plaus_env_%j.err

mkdir -p jobs
source .venv/bin/activate

srun python -m scripts.postprocessing.calibrate_plausibility_envelopes \
    dataset.name=cer_bis calibration_split=train
