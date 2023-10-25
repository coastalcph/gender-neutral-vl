import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

base_dir = "/projects/nlp/data/data/multimodal-gender-bias"

# Comment/uncomment the following variables according to the split
# you want to evaluate and whether we're averaging across sweeps (multiple runs)
split = "val"
# split = "test"
# sweep = True
sweep = False


RESULTS = {
    "lxmert_180k": f"{base_dir}/outputs/nlvr2/lxmert_180k/{split}_result.json",
    "lxmert_180k_neutral": f"{base_dir}/outputs/nlvr2/lxmert_180k_neutral/{split}_result.json",
    # "lxmert_180k_sanity_check": f"{base_dir}/outputs/nlvr2/lxmert_180k_sanity_check/{split}_result.json",
    "lxmert_3m": f"{base_dir}/outputs/nlvr2/lxmert_3m/{split}_result.json",
    "lxmert_3m_neutral": f"{base_dir}/outputs/nlvr2/lxmert_3m_neutral/{split}_result.json",
    # "lxmert_3m_small": f"{base_dir}/outputs/nlvr2/lxmert_3m_small/{split}_result.json",
    "albef_4m": f"{base_dir}/outputs/nlvr2/albef_4m/nlvr2_{split}_epoch9.json",
    "albef_4m_coco_neutral": f"{base_dir}/outputs/nlvr2/albef_4m_coco_neutral/nlvr2_{split}_epoch9.json",
    "albef_4m_cc3m_neutral": f"{base_dir}/outputs/nlvr2/albef_4m_cc3m_neutral/nlvr2_{split}_epoch9.json",
    "albef_14m": f"{base_dir}/outputs/nlvr2/albef_14m/nlvr2_{split}_epoch9.json",
    "albef_14m_coco_neutral": f"{base_dir}/outputs/nlvr2/albef_14m_coco_neutral/nlvr2_{split}_epoch9.json",
    "albef_14m_cc3m_neutral": f"{base_dir}/outputs/nlvr2/albef_14m_cc3m_neutral/nlvr2_{split}_epoch9.json",
    "blip": f"{base_dir}/outputs/nlvr2/blip/nlvr2_{split}_epoch9.json",
    "blip_coco_neutral": f"{base_dir}/outputs/nlvr2/blip_coco_neutral/nlvr2_{split}_epoch9.json",
}

RESULTS_SWEEP = {
    "lxmert_180k": f"{base_dir}/sweeps/lxmert_180k/nlvr2/results_SEED/{split}_result.json",
    "lxmert_180k_neutral": f"{base_dir}/sweeps/lxmert_180k_neutral/nlvr2/results_SEED/{split}_result.json",
    "lxmert_3m": f"{base_dir}/sweeps/lxmert_3m/nlvr2/results_SEED/{split}_result.json",
    "lxmert_3m_neutral": f"{base_dir}/sweeps/lxmert_3m_neutral/nlvr2/results_SEED/{split}_result.json",
}


def acc_by_gender(df):
    # Accuracy by gender
    gender = "Male"
    items = df.loc[df["gender"] == gender, ["label", "prediction"]]
    print(
        gender
        + ": {:.2f}".format(100 * accuracy_score(items["prediction"], items["label"]))
    )

    gender = "Female"
    items = df.loc[df["gender"] == gender, ["label", "prediction"]]
    print(
        gender
        + ": {:.2f}".format(100 * accuracy_score(items["prediction"], items["label"]))
    )

    gender = "Neutral"
    items = df.loc[df["gender"] == gender, ["label", "prediction"]]
    print(
        gender
        + ": {:.2f}".format(100 * accuracy_score(items["prediction"], items["label"]))
    )
    print("")


# LOAD DF WITH GENDER FROM SENTENCES
data_dict = json.load(
    open(f"{base_dir}/data/volta/nlvr2/annotations/dev_gender.jsonl", "r")
)
df = pd.DataFrame.from_dict(data_dict)
gender_counter = df["gender"].value_counts()


if not sweep:
    for modelname, path in RESULTS.items():
        print(modelname)
        data = json.load(open(path, "r"))
        y_pred = [item["prediction"] for item in data]
        y_label = [item["label"] for item in data]
        print("{:.2f}".format(100 * accuracy_score(y_pred, y_label)))
        if split == "val":
            pred_df = pd.DataFrame.from_dict(data)
            # Add gender information
            pred_df["gender"] = ""
            pred_df["sentence"] = [
                df.loc[df["id"] == id_, "sentence"].values[0]
                for id_ in pred_df.identifier
            ]
            sents = pred_df["sentence"].values
            for s in sents:
                gender = df.loc[df["sentence"] == s, "gender"].values[0]
                pred_df.loc[
                    pred_df.loc[pred_df["sentence"] == s].index, "gender"
                ] = gender
            ## Accuracy by gender
            acc_by_gender(pred_df)
            print()
else:
    for modelname, path in RESULTS_SWEEP.items():
        print(modelname)
        acc = []
        data = json.load(open(RESULTS[modelname], "r"))
        y_pred = [item["prediction"] for item in data]
        y_label = [item["label"] for item in data]
        acc.append(accuracy_score(y_pred, y_label))
        for s in [0, 23, 42, 56, 92]:
            data = json.load(open(path.replace("SEED", str(s)), "r"))
            y_pred = [item["prediction"] for item in data]
            y_label = [item["label"] for item in data]
            acc.append(accuracy_score(y_pred, y_label))
        print(acc)
        print("NLVR2 Mean (test): {:.2f}".format(100 * np.mean(acc)))
        print("NLVR2 Std (test): {:.2f}".format(100 * np.std(acc)))
        print()
