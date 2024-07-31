#!/bin/bash

#SBATCH --job-name="KeyserSoze"
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --mem=20GB
#SBATCH --output=output.out
#SBATCH --signal=INT@60

bash commands.txt
