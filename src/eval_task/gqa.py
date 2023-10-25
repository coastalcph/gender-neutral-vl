import pandas as pd
from sklearn.metrics import accuracy_score

base_dir = "/projects/nlp/data/data/multimodal-gender-bias"

RESULTS = {
    "lxmert_180k": f"{base_dir}/outputs/GQA/lxmert_180k/val_result.json",
    "lxmert_180k_neutral": f"{base_dir}/outputs/GQA/lxmert_180k_neutral/val_result.json",
    "lxmert_cc3m": f"{base_dir}/outputs/GQA/lxmert_cc3m/val_result.json",
    "lxmert_cc3m_neutral": f"{base_dir}/outputs/GQA/lxmert_cc3m_neutral/val_result.json",
    "albef_4m": f"{base_dir}/outputs/GQA/albef_4m/vqa_result.json",
    "albef_4m_coco_neutral": f"{base_dir}/outputs/GQA/albef_4m_coco_neutral/vqa_result.json",
    "albef_4m_cc3m_neutral": f"{base_dir}/outputs/GQA/albef_4m_cc3m_neutral/vqa_result.json",
    "albef_14m": f"{base_dir}/outputs/GQA/albef_14m/vqa_result.json",
    "albef_14m_coco_neutral": f"{base_dir}/outputs/GQA/albef_14m_coco_neutral/vqa_result.json",
    "albef_14m_cc3m_neutral": f"{base_dir}/outputs/GQA/albef_14m_cc3m_neutral/vqa_result.json",
    "blip": f"{base_dir}/outputs/GQA/blip/vqa_result.json",
    "blip_coco_neutral": f"{base_dir}/outputs/GQA/blip_coco_neutral/vqa_result.json",
}

# LOAD DF WITH GENDER FROM QUESTIONS
gqadf = pd.read_pickle(f"{base_dir}/data/gqa/valid_qa_gender_and_lemma.pkl")
gqadf["question_id"] = gqadf["question_id"]


def evaluate(df):
    score = 0.0
    target = df["target"]
    prediction = df["prediction"]
    for ans, label in zip(prediction, target):
        if ans == label:
            score += 1.0
    return score / len(df)


def acc_by_gender(df):
    # Accuracy by gender
    gender = "Male"
    items = df.loc[df["gender"] == gender, ["target", "prediction"]]
    print(
        gender
        + ": {:.2f}".format(100 * accuracy_score(items["prediction"], items["target"]))
    )

    gender = "Female"
    items = df.loc[df["gender"] == gender, ["target", "prediction"]]
    print(
        gender
        + ": {:.2f}".format(100 * accuracy_score(items["prediction"], items["target"]))
    )

    gender = "Neutral"
    items = df.loc[df["gender"] == gender, ["target", "prediction"]]
    print(
        gender
        + ": {:.2f}".format(100 * accuracy_score(items["prediction"], items["target"]))
    )
    print("")


for modelname, path in RESULTS.items():
    print(modelname)
    pred_df = pd.read_json(path, dtype=str)
    if modelname.startswith("albef") or modelname.startswith("blip"):
        # For Blip and Albef, we need to distinguish between validation and test instances.
        # Take LXMERT as ground-truth for splitting into VAL and TEST
        df = pd.read_json(
            f"{base_dir}/outputs/GQA/lxmert_180k/results/test_result.json",
            dtype=str,
        )

        # TEST
        df_original_test = pred_df[pred_df.questionId.isin(df.questionId)]
        # df_original_test["questionId"] = df_original_test["questionId"].apply(str)
        df_original_test.to_json(path.replace(".json", "_test.json"), orient="records")

        # VALIDATION
        pred_df = pred_df[pred_df.questionId.isin(gqadf.question_id)]

        # add target & gender
        pred_df.sort_values("questionId", inplace=True)
        gqadf.sort_values("question_id", inplace=True)
        pred_df["gender"] = gqadf["q_gender"].values
        pred_df["target"] = gqadf["label_"].values

    print(
        "accuracy: {:.2f}".format(
            100 * accuracy_score(pred_df["prediction"].values, pred_df["target"].values)
        )
    )
    # Accuracy by gender
    acc_by_gender(pred_df)
