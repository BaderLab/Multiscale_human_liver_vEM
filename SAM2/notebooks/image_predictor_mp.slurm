#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=SAM2_infer
#SBATCH --gres=gpu:t4:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=127000M
#SBATCH --time=3:00:00
#SBATCH --output=/home/codee/scratch/segment-anything-2/log/SAM2infer_%j_%N.txt
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load opencv/4.8.1
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r /home/codee/scratch/segment-anything-2/requirements.txt

export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MAIN_NODE=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

log_dir="/home/codee/scratch/segment-anything-2/log"

echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

echo "$SLURM_NODEID Launching python script"

#srun python /home/codee/scratch/nnUNet/nnunetv2/run/run_training.py 025 2d 0 -p nnUNetResEncUNetMPlans --npz -num_gpus 4 > $log_dir/nnunettrain6.26_3
srun python /home/codee/scratch/segment-anything-2/notebooks/image_predictor_mp.py > $log_dir/sam2infer_0912
echo "done!!!!!!!"