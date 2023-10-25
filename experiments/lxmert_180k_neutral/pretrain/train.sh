#!/bin/bash
#SBATCH --job-name=lxmert180k_neutral
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=0-08:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

export PYTHONPATH=$(builtin cd ..; pwd)

CODE_DIR=/home/sxk199/projects/multimodal-gender-bias/src/LXMERT
BASE_DIR="/projects/nlp/data/data/multimodal-gender-bias"
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/outputs
ANNOS_DIR=${BASE_DIR}/data/volta/mscoco/annotations
FEATS_DIR=${BASE_DIR}/data/volta/mscoco/resnet101_faster_rcnn_genome_imgfeats

WANDB_ENT="coastal-multimodal-gb"
WANDB_PROJ="MM-GB"

name=lxmert_180k
task=Pretrain_COCO_neutral
configs=volta/config/original_lxmert.json
ckpt=${CKPT_DIR}/cFGANaAtmN
output=${OUTS_DIR}/${task}/${name}
logs=logs/${task}/${name}

mkdir -p $output

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate multimodal


cd $CODE_DIR
time python3 pretrain.py \
  --config_file ${configs} \
  --from_pretrained ${ckpt} \
  --task ${task} \
  --timestamp \
  --train_batch_size 256 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 20 \
  --learning_rate 5e-5 \
  --adam_epsilon 1e-6 \
  --adam_betas 0.9 0.999 \
  --weight_decay 0.01 \
  --warmup_proportion 0.05 \
  --p_neutral_cap 0.15 \
  --clip_grad_norm 1.0 \
  --objective 1 \
  --num_train_epochs 1 \
  --last_epoch 20 \
  --annotations_path ${ANNOS_DIR} \
  --features_path ${FEATS_DIR} \
  --output_dir ${output} \
  --logdir ${logs} \
  --use_wandb \
  --wandb_project ${WANDB_PROJ} \
  --wandb_entity ${WANDB_ENT} \
  --wandb_run_name ${name}-${task} \

conda deactivate
