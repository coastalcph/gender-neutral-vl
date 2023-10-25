#!/bin/bash
#SBATCH --job-name=pretrain_prep
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

export PYTHONPATH=$(builtin cd ..; pwd)

CODE_DIR=/home/sxk199/projects/multimodal-gender-bias/src/preprocessing
BASE_DIR=/projects/nlp/data/data/multimodal-gender-bias

task=0
task_name=preprocessing
echo "Task ${task}: ${task_name}"


. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate multimodal

cd $CODE_DIR

name=coco
input=${BASE_DIR}/data/${name}/lxmert
output=${BASE_DIR}/data/pretrain/${name}/gender-neutral
logs=logs/${task_name}/${name}

mkdir -p $output

for split in "train" "val"; do
    echo ${name} ${split}
    python gender_neutral.py ${name} ${split} ${input} ${output}
done


name=cc3m
input=${BASE_DIR}/data/volta/conceptual_captions/annotations
output=${BASE_DIR}/data/pretrain/${name}/gender-neutral
logs=logs/${task_name}/${name}

mkdir -p $output

for split in "train" "val"; do
    echo ${name} ${split}
    python gender_neutral.py ${name} ${split} ${input} ${output}
done

conda deactivate