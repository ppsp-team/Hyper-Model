#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=HypKur
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=40
#SBATCH --nodes=1
#SBATCH --mem-per-cpu 4GB

# loading proper modules
module load StdEnv/2020
module load python/3.9

# start virtual environment
source ~/PY39/bin/activate

# start pipeline
python run_kuramoto_parallel.py
