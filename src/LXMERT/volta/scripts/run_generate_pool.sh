#!/bin/bash
#SBATCH --job-name=gen-pool
#SBATCH --ntasks=1
#SBATCH --mem 2GB
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-4:00:00

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(builtin cd ..; pwd)
export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=/home/sxk199/miniconda3/envs/multimodal/lib:$LD_LIBRARY_PATH
export PYTHONWARNINGS="ignore"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate genvlm
python generate_pool.py mscoco
conda deactivate