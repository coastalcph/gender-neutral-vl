#!/bin/bash
#SBATCH --job-name=gqa_dba
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --output=eval_bias_gqa.out
#SBATCH --error=eval_bias_gqa.err

export PYTHONPATH=$(builtin cd ..; pwd)

CODE_DIR=/home/sxk199/projects/multimodal-gender-bias/src/bias/extrinsic_bias_amp
BASE_DIR="/projects/nlp/data/data/multimodal-gender-bias/outputs/GQA"
INPUT_DIR="/projects/nlp/data/data/multimodal-gender-bias/data/gqa"

task_name=GQA
echo "Task eval: ${task_name}"

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate multimodal

cd $CODE_DIR
sweep=0 
python gqa_dba.py ${BASE_DIR} ${INPUT_DIR} ${sweep}
sweep=1  # Results for appendix after 6 run
python gqa_dba.py ${BASE_DIR} ${INPUT_DIR}/sweeps ${sweep}

conda deactivate