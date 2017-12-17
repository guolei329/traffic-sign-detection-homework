#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=CV3
#SBATCH --output=test.out
module load python3/intel/3.6.3 
module load pytorch/python3.6/0.3.0_4
python3 evaluate.py --model=model_100.pth
exit
