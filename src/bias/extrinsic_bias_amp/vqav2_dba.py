import sys
import json
import numpy as np
import pandas as pd
from typing import List
from bias.tools.directional_biasamp import biasamp_attribute_to_task as dbamp_at

base_dir = sys.argv[1]
input_dir = sys.argv[2]
sweep = bool(int(sys.argv[3]))

RESULTS = {
    "lxmert_180k": f"{base_dir}/lxmert_180k/val_result.json",
    "lxmert_180k_neutral": f"{base_dir}/lxmert_180k_neutral/val_result.json",
    "lxmert_cc3m": f"{base_dir}/lxmert_cc3m/val_result.json",
    "lxmert_cc3m_neutral": f"{base_dir}/lxmert_cc3m_neutral/val_result.json",
    "albef_4m": f"{base_dir}/albef_4m/vqa_result.json",
    "albef_4m_coco_neutral": f"{base_dir}/albef_4m_coco_neutral/vqa_result.json",
    "albef_4m_cc3m_neutral": f"{base_dir}/albef_4m_cc3m_neutral/vqa_result.json",
    "albef_14m": f"{base_dir}/albef_14m/vqa_result.json",
    "albef_14m_coco_neutral": f"{base_dir}/albef_14m_coco_neutral/vqa_result.json",
    "albef_14m_cc3m_neutral": f"{base_dir}/albef_14m_cc3m_neutral/vqa_result.json",
    "blip": f"{base_dir}/blip/vqa_result.json",
    "blip_coco_neutral": f"{base_dir}/blip_coco_neutral/vqa_result.json",
}


def preprocess_vqa_df_to_dbamp_at_input(
    gender_counter,
    top_objects: List,
    qdf: pd.DataFrame,
    pred_df: pd.DataFrame = None,
    attr_label_names=["Male", "Female", "Neutral"],
):
    """
    return: task_labels: np.ndarray
            attribute_labels: np.ndarray
            task_preds: np.ndarray
    """
    # Initialize variables
    G2I = {k: n for n, k in enumerate(attr_label_names)}
    n = sum([gender_counter[k] for k in attr_label_names])
    task_labels = np.zeros((n, len(top_objects)))
    attribute_labels = np.zeros((n, len(attr_label_names)))
    task_preds = np.zeros((n, len(top_objects)))
    my_counter_labels_gender = np.zeros(
        (len(top_objects), len(attr_label_names)), dtype=int
    )

    sub_qdf = qdf.loc[qdf["q_gender"] != ""]
    if len(attr_label_names) == 2:
        sub_qdf = sub_qdf.loc[sub_qdf["q_gender"] != "Neutral"]

    if pred_df is not None:
        pred_df = pred_df.loc[pred_df["question_id"].isin(list(sub_qdf["question_id"]))]

    # Fill-in TASK values
    for obj_idx, obj in enumerate(top_objects):
        # normalize spaces
        obj = obj.replace(" ", "")
        if pred_df is None:  # Only qdf provided to preprocess ground truth
            # IN THE DATA - only need to compute it once (same ground truth)
            q = np.where(sub_qdf["multiple_choice_answer"].str.match(obj))
            for q_idx in q[0]:
                task_labels[q_idx][obj_idx] = 1
                my_counter_labels_gender[obj_idx][
                    G2I[sub_qdf.iloc[q_idx]["q_gender"]]
                ] += 1
        # PREDICTIONS
        if pred_df is not None:
            q = np.where(pred_df["answer"].str.match(obj))
            for q_idx in q[0]:
                task_preds[q_idx][obj_idx] = 1

    # Fill-in ATTRIBUTE values - only need to compute it once (same ground truth)
    if qdf is not None:
        for attr_idx, attr in enumerate(attr_label_names):
            q = np.where(sub_qdf["q_gender"].str.match(attr))
            for q_idx in q[0]:
                attribute_labels[q_idx][attr_idx] = 1

    return task_labels, attribute_labels, task_preds, my_counter_labels_gender


# Define minimum frequency (in train split)
TH = 50

# Gender in Questions
qdf = pd.read_pickle(f"{input_dir}/v2_mscoco_val2014_qa_gender.pkl")
qdf["question_id"] = qdf["question_id"].apply(str)
qdf = qdf.loc[qdf["q_gender"] != ""]

labels = json.load(
    open(
        "../../preprocessing/top_objects/labels_in_gender_text_vqav2_train.json",
        "rb",
    )
)
_top_objects = [k for k, v in labels.items() if v > TH]
print(f"{len(_top_objects)} labels > {TH}")

# Remove numbers and yes/no
top_objects = []
for obj in _top_objects:
    try:
        n = int(obj)
    except ValueError:
        if obj not in ["yes", "no"]:
            top_objects.append(obj)
print(f"{len(top_objects)} labels > {TH} (excluding numbers and 'yes/no')")

# Remove questions whose answer is not amongst top objects
qdf = qdf[qdf.multiple_choice_answer_lemma.isin(top_objects)]
question_ids = sorted(qdf.question_id.tolist())
assert len(qdf) == len(question_ids)

gender_counter = qdf["q_gender"].value_counts()
print(
    f"Total: {len(qdf)} questions-answer pairs with gender info and target 'objects' with FREQ > {TH}"
)
print(f"Gender distrib. after cleaning out of FREQ >= {TH} objects")
print(gender_counter)

attribute_label_names = ["Male", "Female", "Neutral"]

dbadf = pd.DataFrame(
    columns=attribute_label_names + ["Objects", "N", "nMale", "nFemale", "nNeutral"]
)

if not sweep:
    for modelname, path_ in RESULTS.items():
        print(modelname)
        model = pd.read_json(path_, dtype=str)

        # VALIDATION
        model = model[model.question_id.isin(qdf.question_id)]

        if "gender" not in model.columns:
            # Add gender to predictions
            qdf.sort_values("question_id", ascending=True, inplace=True)
            model.sort_values("question_id", ascending=True, inplace=True)
            try:
                model["img_id"] = qdf["image_id"].values
            except ValueError as err:
                print(f"(!) {err} . Drop duplicates (rnd)")
                model.drop_duplicates(inplace=True)
                model["img_id"] = qdf["image_id"].values
            # gender
            model["gender"] = ""
            for qid, img_id, gender in qdf.loc[
                qdf["q_gender"] != "", ["question_id", "image_id", "q_gender"]
            ].values:
                model.loc[
                    (model["question_id"] == qid) & (model["img_id"] == img_id),
                    "gender",
                ] = gender
        qdf.sort_values("question_id", ascending=True, inplace=True)
        model.sort_values("question_id", ascending=True, inplace=True)

        # DBA
        (
            task_labels,
            attribute_labels,
            _,
            my_counter_labels_gender,
        ) = preprocess_vqa_df_to_dbamp_at_input(
            gender_counter, top_objects, qdf=qdf, attr_label_names=attribute_label_names
        )

        _, _, task_model_preds, _ = preprocess_vqa_df_to_dbamp_at_input(
            gender_counter,
            top_objects,
            qdf,
            model,
            attr_label_names=attribute_label_names,
        )

        dba_gender_to_obj_lxm, dba_distr = dbamp_at(
            task_labels,
            attribute_labels,
            task_model_preds,
            None,
            None,
            [top_objects, attribute_label_names],
        )
        dbadf.loc[modelname] = [
            dba_distr[0],
            dba_distr[1],
            dba_distr[2],
            top_objects,
            np.sum(my_counter_labels_gender, axis=1),
            my_counter_labels_gender[:, 0],
            my_counter_labels_gender[:, 1],
            my_counter_labels_gender[:, 2],
        ]
        print("\n" + modelname.upper(), format(dba_gender_to_obj_lxm, "f"))
        print(np.sum(dba_distr, axis=1))
        print()
else:
    # Sweeps only for variants of LXMERT
    for modelname in [
        "lxmert_180k",
        "lxmert_180k_neutral",
        "lxmert_cc3m",
        "lxmert_cc3m_neutral",
    ]:
        results = []
        for seed in [0, 23, 42, 56, 92]:
            try:
                model = pd.read_json(
                    f"{base_dir}/results_{seed}/val_result.json",
                    dtype=str,
                )
            except:
                continue

            print(modelname, seed)

            # VALIDATION
            model = model[model.question_id.isin(qdf.question_id)]

            if "gender" not in model.columns:
                # Add gender to predictions
                qdf.sort_values("question_id", ascending=True, inplace=True)
                model.sort_values("question_id", ascending=True, inplace=True)
                try:
                    model["img_id"] = qdf["image_id"].values
                except ValueError as err:
                    print(f"(!) {err} . Drop duplicates (rnd)")
                    model.drop_duplicates(inplace=True)
                    model["img_id"] = qdf["image_id"].values
                # gender
                model["gender"] = ""
                for qid, img_id, gender in qdf.loc[
                    qdf["q_gender"] != "", ["question_id", "image_id", "q_gender"]
                ].values:
                    model.loc[
                        (model["question_id"] == qid) & (model["img_id"] == img_id),
                        "gender",
                    ] = gender
            qdf.sort_values("question_id", ascending=True, inplace=True)
            model.sort_values("question_id", ascending=True, inplace=True)

            # DBA
            (
                task_labels,
                attribute_labels,
                _,
                my_counter_labels_gender,
            ) = preprocess_vqa_df_to_dbamp_at_input(
                gender_counter,
                top_objects,
                qdf=qdf,
                attr_label_names=attribute_label_names,
            )

            _, _, task_model_preds, _ = preprocess_vqa_df_to_dbamp_at_input(
                gender_counter,
                top_objects,
                qdf,
                model,
                attr_label_names=attribute_label_names,
            )

            dba_gender_to_obj_lxm, dba_distr = dbamp_at(
                task_labels,
                attribute_labels,
                task_model_preds,
                None,
                None,
                [top_objects, attribute_label_names],
                False,
            )

            results.append(np.sum(dba_distr, axis=1))

        # Latex
        print(f"{modelname} & {np.mean(results)}\pm{np.std(results)}")
        print(f" & {np.mean(results, axis=0)[0]}\pm{np.std(results, axis=0)[0]}")
        print(f" & {np.mean(results, axis=0)[1]}\pm{np.std(results, axis=0)[1]}")
        print(f" & {np.mean(results, axis=0)[2]}\pm{np.std(results, axis=0)[2]}")
