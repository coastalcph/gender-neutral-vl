import os
import sys
import json
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from typing import List
from bias.tools.directional_biasamp import biasamp_attribute_to_task as dbamp_at
from bias.tools.utils import get_seed_words, infer_gender

base_dir = sys.argv[1]
input_dir = sys.argv[2]
mode = sys.argv[3]

assert mode in ["IR", "TR"]

RESULTS = {
    "lxmert_180k": f"{base_dir}/lxmert_180k/",
    "lxmert_180k_neutral": f"{base_dir}/lxmert_180k_neutral/",
    "lxmert_cc3m": f"{base_dir}/lxmert_cc3m/",
    "lxmert_cc3m_neutral": f"{base_dir}/lxmert_cc3m_neutral/",
    "albef_4m": f"{base_dir}/albef_4m/result/",
    "albef_4m_coco_neutral": f"{base_dir}/albef_4m_coco_neutral/result/",
    "albef_4m_cc3m_neutral": f"{base_dir}/albef_4m_cc3m_neutral/result/",
    "albef_14m": f"{base_dir}/albef_14m/result/",
    "albef_14m_coco_neutral": f"{base_dir}/albef_14m_coco_neutral/result/",
    "albef_14m_cc3m_neutral": f"{base_dir}/albef_14m_cc3m_neutral/result/",
    "blip": f"{base_dir}/blip/result/",
    "blip_coco_neutral": f"{base_dir}/blip_coco_neutral/result/",
}


def assign_gender(m, subdf, nlp, img_retrieval=True):
    m["gender"] = ""
    m["gender_ans"] = ""
    m["top1"] = m["answer"].apply(lambda x: x[0])
    m["text"] = ""
    m["gender_query"] = ""
    # Load one of the albef text_val.json files for reference;
    # these 'text_val.json' are the same in all models
    cap_id_2_cap_txt = json.load(
        open(os.path.join(RESULTS["albef_14m"], "text_val.json"))
    )
    cap_id_2_cap_txt_gender = []
    for text in cap_id_2_cap_txt:
        cap_id_2_cap_txt_gender.append(infer_gender(text, nlp))

    # 1. Assign "ground truth" gender
    if img_retrieval:
        for row, target in enumerate(m.target.values):
            target_row_gender = subdf.loc[subdf["image_old_idx"] == target]
            m.loc[row, "gender"] = (
                target_row_gender.gender.values[0]
                if not target_row_gender.empty
                else ""
            )
            m.loc[row, "image_idx"] = (
                target_row_gender.id.values[0] if not target_row_gender.empty else ""
            )

    else:
        m.rename(columns={"image_idx": "image_old_idx"})
        # Target is 5 captions associated with the query image.
        # Thus, the target gender is the same as the query image
        for row, img_idx in enumerate(m.image_idx.values):
            target_row_gender = subdf.loc[subdf["image_old_idx"] == img_idx]
            m.loc[row, "gender"] = (
                target_row_gender.gender.values[0]
                if not target_row_gender.empty
                else ""
            )
            m.loc[row, "image_idx"] = (
                target_row_gender.id.values[0] if not target_row_gender.empty else ""
            )

    # 2. Assign gender from prediction
    for row, top1 in enumerate(m.top1.values):
        if img_retrieval:
            m.loc[row, "gender_ans"] = m.loc[m["target"] == top1, "gender"].values[0]
        else:
            m.loc[row, "gender_ans"] = cap_id_2_cap_txt_gender[top1]

    # 3. Assign gender of query
    if img_retrieval:
        for row, idx in enumerate(m.caption_idx.values):
            text = cap_id_2_cap_txt[idx]
            m.loc[row, "text"] = text
            m.loc[row, "gender_query"] = cap_id_2_cap_txt_gender[idx]
    else:
        for row, idx in enumerate(m.image_idx.values):
            query_row_gender = subdf.loc[subdf["id"] == idx]
            m.loc[row, "gender_query"] = (
                query_row_gender.gender.values[0] if not query_row_gender.empty else ""
            )
    return m


def preprocess_flickr_df_to_dbamp_at_input(
    gender_counter,
    top_objects: List,
    df30k: pd.DataFrame,
    pred_df: pd.DataFrame = None,
    attr_label_names=["Male", "Female", "Neutral"],
):
    """
    return: task_labels: np.ndarray
            attribute_labels: np.ndarray
            task_preds: np.ndarray
    """
    # Initialize variables
    n_captions = 5
    G2I = {k: n for n, k in enumerate(attr_label_names)}
    n = sum([gender_counter[k] for k in attr_label_names]) * n_captions
    task_labels = np.zeros((n, len(top_objects)))
    attribute_labels = np.zeros((n, len(attr_label_names)))
    task_preds = np.zeros((n, len(top_objects)))
    my_counter_labels_gender = np.zeros(
        (len(top_objects), len(attr_label_names)), dtype=int
    )

    # sub_qdf = df30k.loc[df30k["gender"] != ""]
    sub_qdf = df30k.copy()
    if len(attr_label_names) == 2:
        sub_qdf = sub_qdf.loc[sub_qdf["gender"] != "Neutral"]
    if pred_df is not None:
        pred_df = pred_df.loc[pred_df["gender"] != ""]
        pred_df = pred_df.loc[pred_df["image_idx"].isin(list(sub_qdf["id"]))]
        print(len(pred_df))

    # Fill-in TASK values
    if pred_df is None:  # Only qdf provided to preprocess ground truth
        for obj_idx, obj in enumerate(top_objects):
            # IN THE DATA - only need to compute it once (same ground truth)
            # for q_idx, captions in enumerate(sub_qdf['sentences_lemma'].values):
            for q_idx, captions in enumerate(sub_qdf["sentences"].values):
                position_id = np.where(
                    [obj in cap.split() for cap in captions]
                )  # this way, 'phone' not in 'phone-booth'
                for cap_id in position_id[0]:
                    task_labels[n_captions * q_idx + cap_id][obj_idx] = 1
                    my_counter_labels_gender[obj_idx][
                        G2I[sub_qdf.iloc[q_idx]["gender"]]
                    ] += 1

    # Fill-in ATTRIBUTE values - only need to compute it once (same ground truth)
    # And also PREDICTIONS
    # In the case of Image Retrieval, we will evaluate whether the gender associated
    # to the image retrieved is the same/diff as the gender from the target caption
    # (taken from its correspondent image)
    for attr_idx, attr in enumerate(attr_label_names):
        # attributes
        q = np.where(sub_qdf["gender"].str.match(attr))
        for q_idx in q[0]:
            attribute_labels[q_idx][attr_idx] = 1
        # image retrieved (predictions)
        if pred_df is not None:
            q = np.where(pred_df["gender_ans"].str.match(attr))
            for q_idx in q[0]:
                task_preds[q_idx][attr_idx] = 1
    if pred_df is not None:
        # If gender_ans is Neutral, add 1 to the same attr_idx from the caption
        q = np.where(pred_df["gender_ans"].str.match("Neutral"))
        for q_idx in q[0]:
            _attr_idx = G2I[pred_df.iloc[q_idx].gender]
            task_preds[q_idx][_attr_idx] = 1

    return task_labels, attribute_labels, task_preds, my_counter_labels_gender


# Define minimum frequency (in train split)
TH = 50

# Gender in captions
df30k = pd.read_pickle(f"{input_dir}/valid_ann_gender.pkl")
df30k.id = pd.to_numeric(df30k.id)

df30k.drop(df30k[df30k.gender == "X"].index, inplace=True)
df30k = df30k.loc[df30k["gender"] != ""]

subdf = df30k.loc[df30k["gender"] != "", ["id", "gender"]]
subdf.reset_index(inplace=True)
subdf.rename(columns={"index": "image_old_idx"}, inplace=True)

labels = json.load(
    open(
        "../../preprocessing/top_objects/labels_in_gender_text_flickr_train.json",
        "r",
    )
)
_top_objects = [k.lower() for k, v in labels.items() if v > TH]
print(f"{len(_top_objects)} labels > {TH}")

# Remove person-related nouns
f, m, neutral = get_seed_words()
f.extend(m)
f.extend(neutral)
top_objects = []
for obj in _top_objects:
    if obj not in f:
        top_objects.append(obj)

print(f"{len(top_objects)} labels > {TH} (exclud. person type nouns)")

# Remove questions whose answer is not amongst top objects
image_ids = sorted(df30k.id.tolist())
gender_counter = df30k["gender"].value_counts()
print(
    f"Total: {len(df30k)} questions-answer pairs with gender info and target 'objects' with FREQ>{TH}"
)
print(f"Gender distrib. after cleaning out of FREQ >= {TH} objects")
print(gender_counter)

attribute_label_names = ["Male", "Female"]
nlp = spacy.load("en_core_web_sm")

if mode == "IR":
    for modelname, path_ in RESULTS.items():
        # LXMERT (from Volta) stores different format compared to ALBEF and BLIP
        if modelname.startswith("lxmert"):
            model = pd.read_json(os.path.join(path_, "val_result_IR.json"))
            try:
                model["target"] = [int(item) for item in model["target"]]
            except TypeError:
                model["target"] = [int(item[0]) for item in model["target"]]
        else:
            preds = np.load(os.path.join(path_, f"score_val_t2i.npy"))
            model = pd.DataFrame(columns=["caption_idx", "answer", "target"])
            model["caption_idx"] = list(range(0, preds.shape[0]))
            model["target"] = [i // 5 for i in range(preds.shape[0])]
            model["answer"] = [np.argsort(s)[::-1] for s in preds]

        if "gender" not in model.columns:
            model = assign_gender(model, subdf, nlp)
        if path_.endswith(".json"):
            model.to_json(
                path_.replace(".json", "_gender.json"), orient="records", indent=4
            )

        # DBA
        (
            task_labels,
            attribute_labels,
            _,
            my_counter_labels_gender,
        ) = preprocess_flickr_df_to_dbamp_at_input(
            gender_counter,
            top_objects,
            df30k=df30k,
            attr_label_names=attribute_label_names,
        )

        _, _, task_model_preds, _ = preprocess_flickr_df_to_dbamp_at_input(
            gender_counter,
            top_objects,
            df30k,
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
        print("\n" + modelname.upper(), format(dba_gender_to_obj_lxm, "f"))
        print(np.sum(dba_distr, axis=1))
        print()

        # QUERY WITH NEUTRAL TEXT
        neutralid = np.where(model.gender_query.str.match("Neutral"))[0]
        gender_image_prediction = model.gender_ans[neutralid]
        gender_image_prediction = Counter(gender_image_prediction)
        print(f"When query the model with neutral caption, the images retrieved:")
        print(
            f"{modelname}=[{100 * (gender_image_prediction['Male'] / len(neutralid))}, "
            f"{100 * (gender_image_prediction['Female'] / len(neutralid))}, "
            f"{100 * (gender_image_prediction['Neutral'] / len(neutralid))}]"
        )

elif mode == "TR":
    for modelname, path_ in RESULTS.items():
        if modelname.startswith("lxmert"):
            model = pd.DataFrame(
                pd.read_pickle(os.path.join(path_, "val_result_TR.pkl"))
            )
        else:
            preds = np.load(os.path.join(path_, f"score_val_i2t.npy"))
            preds_in_list = [
                {
                    "image_idx": i,
                    "target": list(range(5 * i, 5 * i + 5)),
                    "answer": np.argsort(item)[::-1],
                }
                for i, item in enumerate(preds)
            ]
            model = pd.DataFrame(
                columns=["image_idx", "answer", "target"], data=preds_in_list
            )

        if "gender" not in model.columns:
            model = assign_gender(model, subdf, nlp, False)
        if path_.endswith(".json"):
            model.to_json(
                path_.replace(".json", "_gender.json"), orient="records", indent=4
            )

        # DBA
        (
            task_labels,
            attribute_labels,
            _,
            my_counter_labels_gender,
        ) = preprocess_flickr_df_to_dbamp_at_input(
            gender_counter,
            top_objects,
            df30k=df30k,
            attr_label_names=attribute_label_names,
        )

        _, _, task_model_preds, _ = preprocess_flickr_df_to_dbamp_at_input(
            gender_counter,
            top_objects,
            df30k,
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
        print("\n" + modelname.upper(), format(dba_gender_to_obj_lxm, "f"))
        print(np.sum(dba_distr, axis=1))
        print()

        # NEUTRAL CAPTIONS
        neutralid = np.where(model.gender_query.str.match("Neutral"))[0]
        gender_image_prediction = model.gender_ans[neutralid]
        gender_image_prediction = Counter(gender_image_prediction)
        print(f"When query the model with 'neutral image', the captions retrieved:")
        print(
            f"{modelname}=[{100 * (gender_image_prediction['Male'] / len(neutralid))}, "
            f"{100 * (gender_image_prediction['Female'] / len(neutralid))}, "
            f"{100 * (gender_image_prediction['Neutral'] / len(neutralid))}]"
        )

else:
    print(f"ERROR! Argument {mode} not recognised")
