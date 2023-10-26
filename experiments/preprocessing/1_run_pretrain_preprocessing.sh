#!/bin/bash
#SBATCH --job-name=pretrain_prep
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

export PYTHONPATH=$(builtin cd ..; pwd)

. ../../main.config

task=0
task_name=preprocessing
echo "Task ${task}: ${task_name}"


. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate genvlm

cd $CODE_DIR

name=coco
output=${BASE_DIR}/data/pretrain/${name}
logs=logs/${task_name}/${name}
mkdir -p $output

for split in "train" "val"; do
    echo "Processing COCO (LXMERT splits) ${split}"
    python preprocessing/preprocess_coco_lxmert.py \
    --input_dir ${BASE_DIR}/data/coco/lxmert  \
    --output ${output} \
    --split ${split} \
    --princeton_demo ${BASE_DIR}/data/coco/COCO2014_VAL_DEMOGRAPHICS \
    --coco_karpathy ${BASE_DIR}/data/coco/karpathy
done

name=cc3m
output=${BASE_DIR}/data/pretrain/${name}
logs=logs/${task_name}/${name}
mkdir -p $output

for split in "train" "val"; do
    echo "Processing Conceptual Captions 3M ${split}"
    python preprocess_cc3m.py \
    --input_dir ${BASE_DIR}/data/volta/conceptual_captions/annotations \
    --output ${output} \
    --split ${split} 
done

conda deactivate