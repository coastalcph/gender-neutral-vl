#!/bin/bash
#SBATCH --job-name=albef14m_coco_neutral
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -p gpu --gres=gpu:a100:4
#SBATCH --time=08:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

CODE_DIR=${CODE_DIR}/ALBEF


name=albef_14m_coco_neutral
task=NLVR
configs=configs/${task}.yaml
ckpt=${OUTS_DIR}/Pretrain_COCO_neutral_s1234/albef_14m/checkpoint_00.pth 
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.3
eval "$(conda shell.bash hook)"
source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=4 --master_port $(($RANDOM + $RANDOM)) NLVR.py \
    --distributed \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --wandb_project ${WANDB_PROJ} \
    --wandb_entity ${WANDB_ENT} \
    --wandb_run ${name}-${task} \
    # --seed 42 \
    # --resume \
    # --device cpu

deactivate
