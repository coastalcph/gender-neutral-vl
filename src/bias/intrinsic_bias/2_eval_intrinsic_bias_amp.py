import os
import argparse
import json
import numpy as np
import pandas as pd
from tools.utils import get_seed_words
from tools.directional_biasamp import biasamp_task_to_attribute as dbamp_ta
from typing import List

G2I = {"Male": 0, "Female": 1, "Neutral": 2}


def biasamp_in_pretraining(args: argparse.ArgumentParser) -> [float, List[float]]:
    """Compute Bias Amplification in pretraining as explained in Section 4.2

    Args:
        args (argparse.ArgumentParser): parser with input arguments.

    Returns:
        [float, List[float]]: return the overall bias amplification,
        and the bias amplification split by demographic group
    """

    def _preprocess_df_to_dbamp_ta_input(
        gender_counter,
        top_objects: List,
        df_: pd.DataFrame,
        attr_label_names=["Male", "Female", "Neutral"],
    ):
        """
        return: task_labels: np.ndarray
                attribute_labels: np.ndarray
                task_preds: np.ndarray
        """
        # Initialize variables
        n = sum(
            [
                gender_counter["Male"],
                gender_counter["Female"],
                gender_counter["Neutral"],
            ]
        )
        task_labels = np.zeros((n, len(top_objects)))
        attribute_labels = np.zeros((n, 3))
        attribute_preds = np.zeros((n, 3))
        # my_counter_labels_gender = np.zeros((len(top_objects), 3), dtype=int)

        # Fill-in TASK values
        for obj_idx, obj in enumerate(top_objects):
            q = np.where(df_["tokens"].str.contains(obj))
            for q_idx in q[0]:
                task_labels[q_idx][obj_idx] = 1
                # my_counter_labels_gender[obj_idx][G2I[df_.iloc[q_idx]["gender"]]] += 1

        # Fill-in ATTRIBUTE values - only need to compute it once (same ground truth)
        for attr_idx, attr in enumerate(attr_label_names):
            q = np.where(df_["gender"] == attr)
            for q_idx in q[0]:
                attribute_labels[q_idx][attr_idx] = 1
            q_pred = np.where(df_["gender_pred"] == attr)
            for q_idx in q_pred[0]:
                attribute_preds[q_idx][attr_idx] = 1
        return task_labels, attribute_labels, attribute_preds

    # Gender mappings
    f, m, n = get_seed_words()
    gender_tokens = {"Female": f, "Male": m, "Neutral": n}
    gender_tokens["all"] = set(f).union(set(m)).union(set(n))

    annotations_df = pd.read_csv(args.annotations, sep="\t")

    # Load list with objects ranked by frequency of appearance
    all_objects = json.load(
        open(
            f"preprocessing/1_top_objects/labels_in_gender_text_{args.dataset}_train.json",
            "r",
        )
    )
    top_objects = []
    K = 100
    for k in all_objects.keys():
        if k not in gender_tokens["all"]:
            top_objects.append(k)
            if len(top_objects) == K:
                break
    print(f"Top-{K} nouns in {args.dataset}: {top_objects}")

    annotations_df["gender_pred"].replace("", np.nan, inplace=True)
    annotations_df.dropna(subset=["gender_pred"], inplace=True)
    gender_counter = annotations_df["gender"].value_counts()

    # Match input format to dbamp_ta()
    (
        task_labels,
        attribute_labels,
        attribute_preds,
    ) = _preprocess_df_to_dbamp_ta_input(gender_counter, top_objects, annotations_df)

    # Compute bias amplification
    dba_gender_to_obj, values = dbamp_ta(
        task_labels,
        attribute_labels,
        attribute_preds,
        None,
        None,
        [top_objects, list(G2I.keys())],
    )
    dba_gender_to_obj_values = np.nanmean(values, axis=1)
    return dba_gender_to_obj, dba_gender_to_obj_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
        required=True,
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Full path to the file to process (*predictions_tokens.json)",
        required=True,
    )
    parser.add_argument(
        "--annotations",
        type=str,
        help="Annotations file with mapped captions, "
        "output from running 1_eval_intrinsic_bias.py (mapped_captions_val_preds.csv)",
        required=True,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "cc3m"],
        help="Name of the dataset to be used",
    )
    args = parser.parse_args()

    # Compute bias amplification in pretraining (BiasAmp(T -> A))
    dba_, dba_values_gender = biasamp_in_pretraining(args)
    print(f"DBA (on {args.dataset.upper()}): {dba_:.4f}")  # Table 3
    print(f"DBA (on {args.dataset.upper()}) per gender: {dba_values_gender}")  # Table 6
