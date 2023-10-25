#!/bin/bash
#SBATCH --job-name=gqa_lxmert3m
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

export PYTHONPATH=$(builtin cd ..; pwd)

CODE_DIR=/home/sxk199/projects/multimodal-gender-bias/src/LXMERT
BASE_DIR="/projects/nlp/data/data/multimodal-gender-bias"
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/outputs

WANDB_ENT="coastal-multimodal-gb"
WANDB_PROJ="MM-GB"

name=lxmert_3m
task=15
task_name=GQA
configs=volta/config/lxmert.json
task_config_file=volta/config_tasks/all_trainval_tasks.yml
ckpt=${CKPT_DIR}/fYBrp01t8M
output=${OUTS_DIR}/${task_name}/${name}
logs=logs/${task_name}/${name}

echo "Task ${task}: ${task_name}"

mkdir -p $output

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate multimodal

cd $CODE_DIR
python train_task.py \
	--config_file ${configs} --from_pretrained ${ckpt} \
	--tasks_config_file ${task_config_file} \
	--task $task \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias \
	--weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
	--output_dir ${output} \
	--logdir ${logs} \
	--use_wandb \
	--wandb_project ${WANDB_PROJ} \
	--wandb_entity ${WANDB_ENT} \
  	--wandb_run_name ${name}-${task_name}

conda deactivate
