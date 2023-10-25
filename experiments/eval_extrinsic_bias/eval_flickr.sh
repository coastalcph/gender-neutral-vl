#!/bin/bash
#SBATCH --job-name=flickr_dba
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --output=eval_bias_flickr.out
#SBATCH --error=eval_bias_flickr.err

export PYTHONPATH=$(builtin cd ..; pwd)

CODE_DIR=/home/sxk199/projects/multimodal-gender-bias/src/bias/extrinsic_bias_amp
BASE_DIR="/projects/nlp/data/data/multimodal-gender-bias/outputs/Retrieval_Flickr"
INPUT_DIR="/projects/nlp/data/data/multimodal-gender-bias/data/flickr"

task_name=Retrieval_Flickr
echo "Task eval: ${task_name}"

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate multimodal

cd $CODE_DIR
#Image retrieval
python flickr_dba.py ${BASE_DIR} ${INPUT_DIR} IR
#Text retrieval
python flickr_dba.py ${BASE_DIR} ${INPUT_DIR} TR
conda deactivate