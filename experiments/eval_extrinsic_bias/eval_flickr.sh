#!/bin/bash
#SBATCH --job-name=flickr_dba
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --output=eval_bias_flickr.out
#SBATCH --error=eval_bias_flickr.err

export PYTHONPATH=$(builtin cd ..; pwd)

. ../../main.config

results_to_eval=${OUTS_DIR}/Retrieval_Flickr
input_dir=${DATA_DIR}/flickr

task_name=Retrieval_Flickr
echo "Task eval: ${task_name}"

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate genvlm

cd $CODE_DIR
#Image retrieval
python bias/extrinsic_bias_amp/flickr_dba.py ${results_to_eval} ${input_dir} IR
#Text retrieval
python bias/extrinsic_bias_amp/flickr_dba.py ${results_to_eval} ${input_dir} TR
conda deactivate