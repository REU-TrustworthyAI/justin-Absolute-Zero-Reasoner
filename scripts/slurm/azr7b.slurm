#!/bin/bash  -l

#SBATCH --job-name=azr-7b		# Name for your job

#SBATCH --account=stf		# Project account to run your job under
#SBATCH --partition=ckpt-all 		# Partition to run your job on

#SBATCH --output=./logs/%x_%j.out		# Output file
#SBATCH --error=./logs/%x_%j.err		# Error file

#SBATCH --gpus-per-node=h200:4
#SBATCH --nodes=1                # Request one node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=16        # Request CPUs for the task
#SBATCH --mem=64G

#SBATCH --time=3-00:00:00		# Time limit

source ~/miniconda3/etc/profile.d/conda.sh
conda activate azr

unset ROCR_VISIBLE_DEVICES
cd /gscratch/scrubbed/jchae3/justin-Absolute-Zero-Reasoner
bash scripts/selfplay/coder7b.sh
