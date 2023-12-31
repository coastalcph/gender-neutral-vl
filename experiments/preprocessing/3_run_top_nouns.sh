#!/bin/bash
#SBATCH --job-name=task_prep
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

export PYTHONPATH=$(builtin cd ..; pwd)

. ../../main.config

OUT_FILES=("${DATA_DIR}/pretrain/coco/caption_train_gender.pkl" \
    "${DATA_DIR}/pretrain/cc3m/caption_train_gender.pkl" \
    "${DATA_DIR}/vqav2/v2_mscoco_train2014_qa_gender.pkl" \
    "${DATA_DIR}/gqa/train_qa_gender.pkl" \
    "${DATA_DIR}/flickr/train_ann_gender.pkl")


. /etc/profile.d/modules.sh
# module load anaconda3/5.3.1
# module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate genvlm

cd $CODE_DIR

# Generate files to evaluate intrinsic (coco, cc3m) 
# and extrinsic bias (vqav2, gqa, flickr)
i=0
for name in "coco" "cc3m" "vqa" "gqa" "flickr"; do
    echo "${name} - preprocessing"
    case $name in
        "coco")
            input=${DATA_DIR}/volta/mscoco/annotations/caption_train.json
            python preprocessing/prepare_imgid2gender_pretrain.py ${name} ${input} ${OUT_FILES[$i]}
            ;;
        "cc3m")
            input=${DATA_DIR}/volta/conceptual_captions/annotations/caption_train.json
            python preprocessing/prepare_imgid2gender_pretrain.py ${name} ${input} ${OUT_FILES[$i]} 
            ;;
        "vqa"|"gqa")
            DATA_DIR=${DATA_DIR}/volta/${name}
            python preprocessing/prepare_${name}.py ${DATA_DIR} ${OUT_FILES[$i]}
            ;;
        "flickr")
            DATA_DIR=${DATA_DIR}/volta/flickr30k
            python preprocessing/prepare_${name}.py ${DATA_DIR} ${OUT_FILES[$i]}
            ;;
        *)
            continue
            ;;
    esac

    echo "Computing ranking of NOUN tokens by frequency in the training split"
    python preprocessing/top_objects/get_nouns_ranking_in_train.py ${name} ${BASE_FILES[$i]} ${CODE_DIR}/top_objects

    ((i++))

done

conda deactivate