#!/bin/bash
#SBATCH --job-name=albef_14m-bias
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --output=eval_bias_albef_14m.out
#SBATCH --error=eval_bias_albef_14m.err

export PYTHONPATH=$(builtin cd ..; pwd)

. ../../main.config



name=albef_14m
task_name=eval_intrinsic_bias
echo "Task: ${task_name}"

split=val
configs=${CODE_DIR}/ALBEF/configs/config_bert.json
ckpt=${CKPT_DIR}/ALBEF.pth

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate genvlm

cd $CODE_DIR

###########################
### CONCEPTUAL CAPTIONS ###
###########################

DATASET=cc3m
IMGS_DIR=${BASE_DIR}/data/volta/conceptual_captions/images/${split}
CAPS_DIR=${BASE_DIR}/data/pretrain/${DATASET}/gender-neutral

output=${OUTS_DIR}/${task_name}/${name}/${DATASET}
mkdir -p $output

# 1. Predict (see Section 5.3, MLM task)
args="""
--output-path $output \
--checkpoint_path $ckpt \
--config $configs \
--split $split \
--batch-size 1 \
--dataset $DATASET \
--features_path $IMGS_DIR \
--dataset-splits-dir $CAPS_DIR 
"""
python3 ALBEF/predict.py $args


# 2. Eval intrinsic bias & intrinsic bias amplification (see Section 4.1 and 4.2)
python3 bias/intrinsic_bias/1_eval_intrinsic_bias.py \
 --model ${name} \
 --dataset ${DATASET} \
 --output_dir ${output} \
 --file ${output}/${split}predictions_tokens.json \
 --annotations ${CAPS_DIR}/mapped_captions_${split}.csv

python3 bias/intrinsic_bias/2_eval_intrinsic_bias_amp.py \
 --model ${name} \
 --dataset ${DATASET} \
 --output_dir ${output} \
 --file ${output}/${split}predictions_tokens.json \
 --annotations ${output}/mapped_captions_${split}_pred.csv



###########################
###         COCO        ###
###########################

DATASET=coco
IMGS_DIR=${BASE_DIR}/data/volta/mscoco/images/${split}2014
CAPS_DIR=${BASE_DIR}/data/pretrain/${DATASET}/gender-neutral

output=${OUTS_DIR}/${task_name}/${name}/${DATASET}
mkdir -p $output

# 1. Predict (see Section 5.3, MLM task)
args="""
--output-path $output \
--checkpoint_path $ckpt \
--config $configs \
--split $split \
--batch-size 1 \
--dataset $DATASET \
--features_path $IMGS_DIR \
--dataset-splits-dir $CAPS_DIR 
"""
python3 ALBEF/predict.py $args


# 2. Eval intrinsic bias & intrinsic bias amplification (see Section 4.1 and 4.2)
python3 bias/intrinsic_bias/1_eval_intrinsic_bias.py \
 --model ${name} \
 --dataset ${DATASET} \
 --output_dir ${output} \
 --file ${output}/${split}predictions_tokens.json \
 --annotations ${CAPS_DIR}/mapped_captions_${split}.csv

python3 bias/intrinsic_bias/2_eval_intrinsic_bias_amp.py \
 --model ${name} \
 --dataset ${DATASET} \
 --output_dir ${output} \
 --file ${output}/${split}predictions_tokens.json \
 --annotations ${output}/mapped_captions_${split}_pred.csv


conda deactivate



