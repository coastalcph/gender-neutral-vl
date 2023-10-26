#!/bin/bash
#SBATCH --job-name=blip_coco_neutral
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err

CODE_DIR=${CODE_DIR}/BLIP


name=blip
task=Retrieval_Flickr
configs=configs/${task}.yaml
ckpt=${OUTS_DIR}/${task}/${name}/checkpoint_best.pth 
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.3
eval "$(conda shell.bash hook)"
source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
python train_retrieval.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --wandb_project ${WANDB_PROJ} \
    --wandb_entity ${WANDB_ENT} \
    --wandb_run ${name}-${task} \
    --evaluate \
    # --resume \
    # --device cpu

deactivate
