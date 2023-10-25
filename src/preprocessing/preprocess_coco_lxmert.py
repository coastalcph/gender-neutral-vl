import os
import enum
import json
import pandas as pd
import spacy
from args import parse_args
from bias.tools.utils import infer_image_gender, compute_classification_report
from dataclasses import dataclass
from typing import List
from tqdm import tqdm


@dataclass
class LxmertDatasetItem:
    id: int
    file_name: str
    gender: str
    karpathy_split: str
    caption: str
    tokens: List[str]
    sentid: int


@dataclass
class DatasetItem:
    id: int
    file_name: str
    skin: List[str]
    ratio: List[float]
    bb_skin: str
    bb_gender: str
    split: str
    caption: str
    tokens: List[str]
    sentid: int


def main(args):
    def _filename_to_id(filename: str) -> int:
        filename_ = filename[len("COCO_val2014_") :]
        id_ = int(filename_)
        return id_

    # Set output file path
    output_dir = os.path.join(args.output, "lxmert")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"caption_{args.split}.csv")
    print(f"Output file: {output_file}")

    # Load NLP pipeline
    # No need to load a tokenizer because captions in Karpathy splits
    # are already tokenized. We will take them from there.
    nlp = spacy.load("en_core_web_sm")

    # Source data: https://github.com/airsplay/lxmert#pre-training
    if args.split == "val":
        # Read images used in LXMERT pretraining (VALIDATION: minival)
        data_df = pd.read_json(os.path.join(args.input_dir, "minival.json"))
        print("LXMERT minival split loaded!")
    elif args.split == "train":
        # Read images used in LXMERT pretraining (TRAINING: train + nominival)
        train_df = pd.read_json(os.path.join(args.input_dir, "mscoco_train.json"))
        print("LXMERT train split loaded!")
        # Read images used in LXMERT pretraining (TRAINING: train + nominival)
        nominival_df = pd.read_json(
            os.path.join(args.input_dir, "mscoco_nominival.json")
        )
        print("LXMERT nominival split loaded!")
        data_df = pd.concat([train_df, nominival_df])
    else:
        raise Exception(f"Split {args.split} unknown!")

    # Load demographics from Princeton's dataset,
    # to compare to gender extracted from captions
    demog_df = _load_demographics(args.princeton_demo)
    print("Demographics loaded!")
    # Load the Karpathy splits because LXMERT was trained
    # with part of the Karpathy validation split (minival split)
    # so we can remove it later from evaluation
    kp_captions_df = _load_karpathy_splits(args.coco_karpathy)
    image_ids_minival = list(set(data_df["img_id"]))
    print("MSCOCO dataset (Karpathy splits) loaded!")

    # Initialize variable
    my_dataset_with_captions_and_demographics = []
    # Following are for debugging:
    gender_inferred = {
        "Female": {"Female": 0, "Male": 0, "Neutral": 0},
        "Male": {"Female": 0, "Male": 0, "Neutral": 0},
        "Neutral": {"Female": 0, "Male": 0, "Neutral": 0},
    }
    gender_inferred_imgs = {
        "Female": {"Female": 0, "Male": 0, "Neutral": 0},
        "Male": {"Female": 0, "Male": 0, "Neutral": 0},
        "Neutral": {"Female": 0, "Male": 0, "Neutral": 0},
    }

    # gender_inferred meaning: the KEY is the "true gender" coming from the annotations,
    # while the key genders inside the dictionaries are inferred.
    for img_id in tqdm(
        image_ids_minival,
        desc=f"Building up LXMERT {args.split} set with annotations ...",
    ):
        id_ = _filename_to_id(img_id)
        k_row = kp_captions_df.loc[
            kp_captions_df["img_id"] == (img_id + ".jpg")
        ].values.tolist()
        row_df = demog_df.loc[demog_df["id"] == id_]
        # Infer gender from captions
        captions_ = [row[2] for row in k_row]
        gender_inf = infer_image_gender(captions_, nlp)
        if gender_inf not in ["Female", "Male", "Neutral"]:
            continue  # Not gender or "X"
        else:
            gender_inferred[row_df["bb_gender"].values[0]][gender_inf] += 1
            gender_inferred_imgs[row_df["bb_gender"].values[0]][gender_inf].append(
                img_id
            )

        # Add captions and corresponding split in Karpathy
        # so we can remove it later from evaluation because LXMERT
        # was trained with part of the Karpathy validation split
        for caption_split_pair in k_row:
            file_name, split_, caption, tokens, sentid = caption_split_pair
            image_caption = LxmertDatasetItem(
                id_, file_name, gender_inf, split_, caption, tokens, sentid
            )
            my_dataset_with_captions_and_demographics.append(image_caption)

    output_df = pd.DataFrame(my_dataset_with_captions_and_demographics)
    output_df.to_csv(output_file, index=False, sep="\t")
    print(f"{output_file} saved!")

    # Print some stats
    print(f"Stats:")
    print(f"{len(set(output_df['id']))} unique images.")
    print(output_df.info())
    print(output_df["gender"].value_counts())
    print(f"{len(set(output_df.loc[output_df['gender'] == 'Male', 'id']))} Male images")
    print(
        f"{len(set(output_df.loc[output_df['gender'] == 'Female', 'id']))} Female images"
    )
    print(
        f"{len(set(output_df.loc[output_df['gender'] == 'Neutral', 'id']))} Neutral images"
    )
    if args.split == "val":
        print(
            f"{len(set(output_df.loc[output_df['gender'] == 'X', 'id']))} images X discarded"
        )
        output_df.drop(output_df[output_df.gender == "X"].index, inplace=True)
        output_df.to_csv(
            output_file.replace(".csv", "_person.csv"), index=False, sep="\t"
        )
        print(f'{output_file.replace(".csv", "_person.csv")} saved!')

    # Comparison between gender from captions and inferred and F1-scores
    (
        f1_score_micro,
        f1_score_binary,
        f1_score_micro_Neutral_ok,
    ) = compute_classification_report(gender_inferred)

    json.dump(
        gender_inferred_imgs,
        open(
            os.path.join(output_dir, f"annotated_vs_inferred_gender_imgid.json"), "w+"
        ),
        indent=True,
    )
    with open(
        os.path.join(output_dir, f"annotated_vs_inferred_gender_imgid.txt"), "w+"
    ) as f:
        f.write(f"{len(set(output_df['id']))} unique images.")
        f.write(
            f"F1-score for comparing gender from annotations VS gender from captions:"
        )
        f.write("F1 (micro)")
        f.write(str(f1_score_micro))
        f.write("F1 (micro, Neutral as OK)")
        f.write(str(f1_score_micro_Neutral_ok))
        f.write("F1 (binary, Neutral as OK)")
        f.write(str(f1_score_binary))


def _load_demographics(annotation_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(annotation_path, "images_val2014.csv"))
    return df


def _load_karpathy_splits(karpathy_basedir: str) -> pd.DataFrame:
    with open(os.path.join(karpathy_basedir, "dataset_coco.json")) as fp:
        captions = json.load(fp)["images"]
    coco = []
    for img in captions:
        for sent in img["sentences"]:
            coco.append(
                {
                    "img_id": img["filename"],
                    "split": img["split"],
                    "caption": sent["raw"],
                    "tokens": sent["tokens"],
                    "sentid": sent["sentid"],
                }
            )
    return pd.DataFrame(coco)


if __name__ == "__main__":
    args = parse_args()
    main(args)
