#!/bin/bash
#SBATCH --job-name=gqa_dba
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --output=eval_bias_gqa.out
#SBATCH --error=eval_bias_gqa.err

export PYTHONPATH=$(builtin cd ..; pwd)

. ../../main.config

results_to_eval=${OUTS_DIR}/GQA
input_dir=${DATA_DIR}/gqa

task_name=GQA
echo "Task eval: ${task_name}"

. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate genvlm

cd $CODE_DIR
sweep=0 
python bias/extrinsic_bias_amp/gqa_dba.py ${results_to_eval} ${input_dir} ${sweep}
sweep=1  # Results for appendix after 6 run
python bias/extrinsic_bias_amp/gqa_dba.py ${results_to_eval} ${input_dir}/sweeps ${sweep}

conda deactivate