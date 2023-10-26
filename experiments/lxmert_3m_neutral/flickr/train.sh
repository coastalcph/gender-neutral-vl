#!/bin/bash
#SBATCH --job-name=flickr_lxmert3m_neutral
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

export PYTHONPATH=$(builtin cd ..; pwd)

. ../../main.config

name=lxmert_3m_neutral
task=8
task_name=Retrieval_Flickr
configs=volta/config/lxmert.json
task_config_file=volta/config_tasks/all_trainval_tasks.yml
ckpt=${OUTS_DIR}/Pretrain_CC3M_neutral/lxmert_3m/pytorch_model_0.bin
output=${OUTS_DIR}/${task_name}/${name}
logs=logs/${task_name}/${name}

echo "Task ${task}: ${task_name}"

mkdir -p $output

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate genvlm

cd $CODE_DIR
python LXMERT/train_task.py \
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
