import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


base_dir = "/projects/nlp/data/data/multimodal-gender-bias"

RESULTS = {
    "lxmert_180k": f"{base_dir}/outputs/VQA/lxmert_180k/val_result.json",
    "lxmert_180k_neutral": f"{base_dir}/outputs/VQA/lxmert_180k_neutral/val_result.json",
    "lxmert_cc3m": f"{base_dir}/outputs/VQA/lxmert_cc3m/val_result.json",
    "lxmert_cc3m_neutral": f"{base_dir}/outputs/VQA/lxmert_cc3m_neutral/val_result.json",
    "albef_4m": f"{base_dir}/outputs/VQA/albef_4m/vqa_result.json",
    "albef_4m_coco_neutral": f"{base_dir}/outputs/VQA/albef_4m_coco_neutral/vqa_result.json",
    "albef_4m_cc3m_neutral": f"{base_dir}/outputs/VQA/albef_4m_cc3m_neutral/vqa_result.json",
    "albef_14m": f"{base_dir}/outputs/VQA/albef_14m/vqa_result.json",
    "albef_14m_coco_neutral": f"{base_dir}/outputs/VQA/albef_14m_coco_neutral/vqa_result.json",
    "albef_14m_cc3m_neutral": f"{base_dir}/outputs/VQA/albef_14m_cc3m_neutral/vqa_result.json",
    "blip": f"{base_dir}/outputs/VQA/blip/vqa_result.json",
    "blip_coco_neutral": f"{base_dir}/outputs/VQA/blip_coco_neutral/vqa_result.json",
}


def compute_score_with_logits(df, df_val_target, ans2label, no_tqdm=False):
    score = []
    errors = 0
    no_label = []
    predictions = {k: v for k, v in zip(df["question_id"].values, df["answer"].values)}
    for qid, pred in tqdm(predictions.items(), total=len(predictions), disable=no_tqdm):
        # One-hot prediction
        pred_id = torch.Tensor([ans2label[pred]]).to(torch.int64)
        one_hot = torch.zeros(1, len(ans2label))
        one_hot.scatter_(1, pred_id.view(-1, 1), 1)
        # One-hot * scores target
        scores_one_hot_target = torch.zeros(len(ans2label), 1)
        entry = df_val_target.loc[df_val_target["question_id"] == qid]
        if len(entry["labels"]) > 0:  # Due to a format error from lxmert files
            labels = (
                torch.Tensor(entry["labels"].values[0]).to(torch.int64).unsqueeze(0)
            )
            scores = torch.Tensor([sc for sc in entry["scores"].values[0]]).unsqueeze(1)
            scores_one_hot_target.scatter_(0, labels.view(-1, 1), scores)
        else:
            no_label.append(qid)

        pair_score = (one_hot.T * scores_one_hot_target).sum()
        score.append(pair_score.item())
        if pair_score == 0:
            errors += 1

    print(f"Accuracy: {100 * np.mean(score):.2f}")
    print(f"Errors: {errors} (out of {len(predictions)})")


def acc_by_gender(qdf_, df, df_val_target, ans2label):
    # Accuracy by gender
    gender = "Male"
    qdf_gender = qdf_.loc[(qdf_["q_gender"] == gender)]
    df_gender = df.loc[df["question_id"].isin(qdf_gender["question_id"])]
    df_val_target_gender = df_val_target.loc[
        df_val_target["question_id"].isin(qdf_gender["question_id"])
    ]
    print(gender)
    compute_score_with_logits(df_gender, df_val_target_gender, ans2label, no_tqdm=True)

    gender = "Female"
    qdf_gender = qdf_.loc[(qdf_["q_gender"] == gender)]
    df_gender = df.loc[df["question_id"].isin(qdf_gender["question_id"])]
    df_val_target_gender = df_val_target.loc[
        df_val_target["question_id"].isin(qdf_gender["question_id"])
    ]
    print(gender)
    compute_score_with_logits(df_gender, df_val_target_gender, ans2label, no_tqdm=True)

    gender = "Neutral"
    qdf_gender = qdf_.loc[(qdf_["q_gender"] == gender)]
    df_gender = df.loc[df["question_id"].isin(qdf_gender["question_id"])]
    df_val_target_gender = df_val_target.loc[
        df_val_target["question_id"].isin(qdf_gender["question_id"])
    ]
    print(gender)
    compute_score_with_logits(df_gender, df_val_target_gender, ans2label, no_tqdm=True)
    print("")


# LOAD DF WITH GENDER FROM QUESTIONS & DATA
qdf_ = pd.read_pickle(
    f"{base_dir}/data/vqav2/v2_mscoco_val2014_qa_gender_and_lemma.pkl"
)
df_val_target = pd.DataFrame(
    pd.read_pickle(f"{base_dir}/data/volta/vqa/cache/val_target.pkl")
)
ans2label = json.load(
    open(f"{base_dir}/data/volta/vqa/annotations/trainval_ans2label.json")
)

for modelname, path in RESULTS.items():
    print(modelname)
    df_original = pd.read_json(path)
    if modelname.startswith("albef") or modelname.startswith("blip"):
        # For Blip and Albef, we need to distinguish between validation and test instances.
        # TEST
        df_original_test = df_original[
            ~df_original.question_id.isin(df_val_target.question_id)
        ]
        df_original_test.to_json(path.replace(".json", "_test.json"), orient="records")
        # VALIDATION
        df_original = df_original[
            df_original.question_id.isin(df_val_target.question_id)
        ]
    compute_score_with_logits(df_original, df_val_target, ans2label)
    # Accuracy by gender
    acc_by_gender(qdf_, df_original, df_val_target, ans2label)
