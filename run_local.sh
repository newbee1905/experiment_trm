#!/bin/bash

export OMP_NUM_THREADS=4 

if command -v nvidia-smi &> /dev/null; then
	NUM_GPUS=$(nvidia-smi -L | wc -l)
else
	NUM_GPUS=1
	echo "nvidia-smi not found. Defaulting to 1 process (CPU/MPS)."
fi

echo "Launching SudokuACT training on $NUM_GPUS GPUs..."

torchrun \
	--standalone \
	--nproc_per_node=$NUM_GPUS \
	trainer.py
