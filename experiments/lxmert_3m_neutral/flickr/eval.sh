#!/bin/bash
#SBATCH --job-name=flickr_lxmert3m_neutral-eval
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err

export PYTHONPATH=$(builtin cd ..; pwd)

CODE_DIR=/home/sxk199/projects/multimodal-gender-bias/src/LXMERT
BASE_DIR="/projects/nlp/data/data/multimodal-gender-bias"
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/outputs

WANDB_ENT="coastal-multimodal-gb"
WANDB_PROJ="MM-GB"

name=lxmert_3m_neutral
task=8
task_name=Retrieval_Flickr
configs=volta/config/lxmert.json
ckpt=${OUTS_DIR}/${task_name}/${name}/pytorch_model_best.bin
output=${OUTS_DIR}/${task_name}/${name}
logs=logs/${task_name}/${name}

echo "Task ${task}: ${task_name}"

# VALIDATION
task_config_file=volta/config_tasks/all_trainval_tasks.yml

python eval_retrieval.py \
	--config_file ${configs} \
	--from_pretrained ${ckpt} \
	--tasks_config_file ${task_config_file} \
	--task $task \
	--batch_size 1 \
	--output_dir ${output} \
	--save_name "val"

# TEST
task_config_file=volta/config_tasks/all_test_tasks.yml

python eval_retrieval.py \
	--config_file ${configs} \
	--from_pretrained ${ckpt} \
	--tasks_config_file ${task_config_file} \
	--task $task \
	--batch_size 1 \
	--output_dir ${output} \
	--save_name "test"

conda deactivate
