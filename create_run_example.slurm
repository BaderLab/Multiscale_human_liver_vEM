#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=create_run_example
#SBATCH --cpus-per-task=32
#SBATCH --mem=190000M
#SBATCH --time=3:00:00
#SBATCH --account=def-gbader
#SBATCH --output=example_%j_%N.txt
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc opencv/4.8.1
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r /nnUNet/requirements.txt
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun python mito_er_analysis.py

echo "done!!!!!!!"