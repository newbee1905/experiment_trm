#!/bin/bash
#SBATCH --job-name=sudoku-act
#SBATCH --qos=xbatch
#SBATCH --gpus=v100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/pretrain_deepspeed_%j.out
#SBATCH --error=logs/pretrain_deepspeed_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s221056384@deakin.edu.au

module load NVHPC/24.9-CUDA-12.6.0
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
set -euo pipefail

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

srun torchrun \
	--nproc_per_node=4 \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	train_sudoku.py
