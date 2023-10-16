#!/bin/bash
#SBATCH --job-name=albef4m_coco_neutral
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -p gpu --gres=gpu:a100:4
#SBATCH --time=02:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

CODE_DIR=/home/pmh864/projects/multimodal-gender-bias/src/ALBEF
ENVS_DIR=/home/pmh864/envs
BASE_DIR="/projects/nlp/data/data/multimodal-gender-bias"
CKPT_DIR=${BASE_DIR}/checkpoints
OUTS_DIR=${BASE_DIR}/outputs
WANDB_ENT="coastal-multimodal-gb"
WANDB_PROJ="MM-GB"

name=albef_4m_coco_neutral
task=VQA
configs=configs/${task}.yaml
ckpt=${OUTS_DIR}/Pretrain_COCO_neutral/albef_4m/checkpoint_00.pth 
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.3
eval "$(conda shell.bash hook)"
source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=4 --master_port $(($RANDOM + $RANDOM)) VQA.py \
    --distributed \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --wandb_project ${WANDB_PROJ} \
    --wandb_entity ${WANDB_ENT} \
    --wandb_run ${name}-${task} \
    # --resume \
    # --device cpu

deactivate
