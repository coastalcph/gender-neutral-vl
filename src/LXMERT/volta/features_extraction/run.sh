#!/bin/bash
#SBATCH --job-name=lmdb
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-12:00:00

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(builtin cd ..; pwd)
. /etc/profile.d/modules.sh
#module load anaconda3/5.3.1
#module load cuda/11.4
eval "$(conda shell.bash hook)"
conda activate multimodal

python h5_to_lxmert_splits.py \
 --h5_tr /projects/nlp/data/emanuele/data/mscoco/features/train2014_boxes36.h5 \
 --h5_val /projects/nlp/data/emanuele/data/mscoco/features/val2014_boxes36.h5

conda deactivate
