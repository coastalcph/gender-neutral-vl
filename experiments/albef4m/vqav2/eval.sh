#!/bin/bash
#SBATCH --job-name=albef4m
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err

CODE_DIR=${CODE_DIR}/ALBEF


name=albef_4m
task=VQA
configs=configs/${task}.yaml
ckpt=${OUTS_DIR}/${task}/${name}/checkpoint_07.pth 
output=${OUTS_DIR}/${task}/${name}

mkdir -p $output

. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/11.3
eval "$(conda shell.bash hook)"
source ${ENVS_DIR}/albef/bin/activate

cd $CODE_DIR
python VQA.py \
    --config $configs \
    --checkpoint $ckpt \
    --output_dir $output \
    --wandb_project ${WANDB_PROJ} \
    --wandb_entity ${WANDB_ENT} \
    --wandb_run ${name}-${task} \
    --evaluate
    # --resume \
    # --device cpu

deactivate
