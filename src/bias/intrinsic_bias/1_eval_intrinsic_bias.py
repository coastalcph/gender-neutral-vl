import os
import argparse
import json
import pandas as pd
from tools.utils import (
    get_seed_words,
    compute_classification_report,
    mark_gender_word,
    plot_confusion_matrix,
)
from datetime import datetime


def evaluate_error_preds(args: argparse.ArgumentParser) -> None:
    """Compute intrinsic bias metrics as explained in Section 4.1.

    Args:
        args (argparse.ArgumentParser): parser with input arguments

    Raises:
        Exception: _description_
    """
    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print("Output directory:", args.output_dir)

    # Gender mappings
    f, m, n = get_seed_words()
    gender_tokens = {"Female": f, "Male": m, "Neutral": n}
    gender_tokens["all"] = set(f).union(set(m)).union(set(n))
    # Load prediction and annotation files
    predictions = json.load(open(args.modelname, "r", encoding="utf8"))
    annotations_df = pd.read_csv(args.annotations, sep="\t")
    # Discard images with no gender assigned: do not count towards evaluation
    annotations_df.drop(
        annotations_df[annotations_df.gender == "X"].index, inplace=True
    )
    print(f"{len(annotations_df)} captions left for evaluation.")
    # The error rate is the number of man/woman misclassifications,
    # while gender neutral terms are not considered errors
    # (read footnote 10 in the paper)
    gender_predictions_for_captions = {
        "Female": {"Female": 0, "Male": 0, "Neutral": 0},
        "Male": {"Female": 0, "Male": 0, "Neutral": 0},
        "Neutral": {"Female": 0, "Male": 0, "Neutral": 0},
    }

    # Initialize variables
    annotations_df["preds"] = [[None]] * len(annotations_df)
    annotations_df["gender_pred"] = [""] * len(annotations_df)
    # The following variables are for debugging purposes
    oov = []
    total_sentences = 0
    sentences_without_masks = 0
    sentences_with_non_agreement_bt_gender_tokens = []

    # Loop over model's predictions (loaded from json file)
    for img_id, word_preds in predictions.items():
        img_data = annotations_df.loc[annotations_df["id"] == int(img_id)]
        if len(img_data) == 0:
            continue
        try:
            assert len(word_preds) == len(img_data)
        except AssertionError:
            # If 'Captions: 0' is because the image doesn't have a gender associated with it
            print(
                f"AssertionError in img {img_id}. Captions: {len(img_data)}. Predictions: {len(word_preds)}"
            )
            continue

        assert len(set(img_data["gender"])) == 1
        img_gender = list(set(img_data["gender"]))[0]

        # For a given image (img_id), loop over its captions
        for id_caption, masks in word_preds.items():
            total_sentences += 1
            if masks:
                annotations_df.at[
                    annotations_df.loc[annotations_df["id"] == int(img_id)].index[
                        int(id_caption)
                    ],
                    "preds",
                ] = masks
                pred_genders = []
                for w in masks:
                    if w.lower() not in gender_tokens["all"]:
                        oov.append(w)
                    else:
                        pred_genders.append(mark_gender_word(w.lower(), gender_tokens))
                pred_genders = list(set(pred_genders))

                if len(pred_genders) == 0:
                    continue  # OOV
                elif len(pred_genders) == 1 or len(pred_genders) == 2:
                    if len(pred_genders) == 1:
                        pred_gender_ = pred_genders[0]
                    elif "Female" in pred_genders and "Neutral" in pred_genders:
                        pred_gender_ = "Female"
                    elif "Male" in pred_genders and "Neutral" in pred_genders:
                        pred_gender_ = "Male"
                    else:
                        # Here usually due to wrong co-reference resolution e.g. ('woman', 'his')
                        pred_gender_ = pred_genders[0]
                        sentences_with_non_agreement_bt_gender_tokens.append(img_id)
                    gender_predictions_for_captions[img_gender][pred_gender_] += 1
                    annotations_df.at[
                        annotations_df.loc[annotations_df["id"] == int(img_id)].index[
                            int(id_caption)
                        ],
                        "gender_pred",
                    ] = pred_gender_
                else:
                    print(
                        f" >>> Caption {id_caption} for image {img_id} has too many genders detected!: {masks}"
                    )
                    sentences_with_non_agreement_bt_gender_tokens.append(img_id)
            else:
                sentences_without_masks += 1

    report = compute_classification_report(annotations_df)

    print("Total sentences: ", total_sentences)
    print(f"... from which, {sentences_without_masks} sentences without masks")
    print(
        f"{len(sentences_with_non_agreement_bt_gender_tokens)} "
        f"images with no gender agreement between top1 predicted tokens"
    )
    print("----------------")
    print(gender_predictions_for_captions)

    # Save the complete dataframe into a new a CSV file
    # This file will be used in 2_plot_intrinsic_bias.py
    af_extended = os.path.basename(os.path.normpath(args.annotations))
    annotations_df.to_csv(
        os.path.join(args.output_dir, af_extended.replace(".csv", "_preds.csv")),
        index=False,
        sep="\t",
    )
    print(
        f'File {os.path.join(args.output_dir, af_extended.replace(".csv", "_preds.csv"))} saved!'
    )

    # Save results in a txt file
    with open(
        os.path.join(args.output_dir, args.output_file),
        "w",
        encoding="utf8",
    ) as f:
        f.write(str(datetime.today()) + "\n\n")
        f.write(f"Total sentences: {total_sentences}\n")
        f.write(f"... from which, {sentences_without_masks} sentences without masks\n")
        f.write(
            f"{len(sentences_with_non_agreement_bt_gender_tokens)} "
            f"images with no gender agreement between top1 predicted tokens\n"
        )
        # f.writelines(sentences_with_non_agreement_bt_gender_tokens)
        f.write("Confusion matrix:\n")
        f.writelines(
            [
                f"{k} ({sum(v_dict.values())}): {str(v_dict)}\n"
                for k, v_dict in gender_predictions_for_captions.items()
            ]
        )
        f.write("Classification report:\n")
        f.write(json.dumps(report))
        oov = set(oov)
        f.write(f"\n{len(oov)} OOV:\n")
        f.writelines([i + "\n" for i in sorted(list(oov))])
    print(f"File {os.path.join(args.output_dir, args.output_file)} saved!")

    # Visualize confusion matrix (not reported in the paper)
    plot_confusion_matrix(gender_predictions_for_captions, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
        required=True,
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output filename",
        default="evaluate_error_preds_summary.txt",
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
        help="Annotations file with mapped captions (mapped_captions_val.csv)",
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

    # Compute intrinsic bias: Precision, Recall, F1-score
    evaluate_error_preds(args)
