#!/bin/bash -l

#SBATCH --job-name=azr-train-a100		# Name for your job

#SBATCH --account=reu-aisocial		# Project account to run your job under
#SBATCH --partition=tier3 		# Partition to run your job on

#SBATCH --output=./logs/%x_%j.out		# Output file
#SBATCH --error=./logs/%x_%j.err		# Error file

#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1                # Request one node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=16        # Request CPUs for the task
#SBATCH --mem=64G

#SBATCH --time=1-00:00:00		# Time limit

spack load /tzo3mmi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate azr

unset ROCR_VISIBLE_DEVICES
cd ~/justin-Absolute-Zero-Reasoner
bash scripts/selfplay/coder3b2.sh
