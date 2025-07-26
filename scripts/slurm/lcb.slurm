#!/bin/bash -l
#SBATCH --job-name=lcb-generation
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/lcb_%A_%a.out
#SBATCH --error=logs/lcb_%A_%a.err

cd /gscratch/scrubbed/jchae3/justin-Absolute-Zero-Reasoner 
conda activate azr
bash evaluation/code_eval/scripts/run_lcb_gen.sh \
        --model andrewzh/Absolute_Zero_Reasoner-Coder-7b \
	-n 512 \
	-t 1.0 \
	-p 1.0 \
	-b 128 \
	-k 8096	\
	-g 2
