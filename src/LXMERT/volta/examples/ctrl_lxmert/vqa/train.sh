#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=2-00:00:00

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(builtin cd ..; pwd)
export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=/home/sxk199/miniconda3/envs/multimodal/lib:$LD_LIBRARY_PATH
export PYTHONWARNINGS="ignore"

TASK=1
MODEL=ctrl_lxmert
CONFIG_DIR=/image/nlp-datasets/laura/configs/volta
MODEL_CONFIG=ctrl_lxmert
TASKS_CONFIG=ctrl_trainval_tasks
PRETRAINED=checkpoints/conceptual_captions/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin

OUTPUT_DIR=checkpoints/vqa/${MODEL}
LOGGING_DIR=logs/vqa

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal

cd ../../..
python train_task.py \
	--config_file ${CONFIG_DIR}/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file ${CONFIG_DIR}/config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
	--output_dir ${OUTPUT_DIR} \
	--logdir ${LOGGING_DIR} \
#	--resume_file ${OUTPUT_DIR}/VQA_${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate
