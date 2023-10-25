import os
import json
import spacy
import pandas as pd

from args import parse_args
from bias.tools.utils import infer_image_gender
from dataclasses import dataclass
from transformers import BertTokenizer
from typing import List
from tqdm import tqdm


@dataclass
class DatasetItem:
    id: int
    gender: str
    split: str
    caption: str
    tokens: List[str]


def main(args):
    # Set output file path
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f"preprocess_caption_{args.split}.csv")
    print(f"Output file: {output_file}")

    # Load tokenizer and NLP pipeline
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    nlp = spacy.load("en_core_web_sm")

    # Load data
    if args.split == "train":
        filepath = os.path.join(args.input_dir, "caption_train.json")
    elif args.split == "val":
        filepath = os.path.join(args.input_dir, "caption_valid.json")
    else:
        raise Exception(f"Split {args.split} unknown!")

    kp_captions = _load_cc_captions(filepath)
    print(f"{args.datafile} loaded!")

    # Map captions and demographic groups (gender)
    my_dataset_with_captions = []
    for img_id, caption in tqdm(
        kp_captions.items(),
        desc="Building up my custom dataset with captions and annotations ...",
    ):
        token_ids = tokenizer(caption)
        tokens = tokenizer.convert_ids_to_tokens(token_ids.input_ids[1:-1])
        gender = infer_image_gender([caption], nlp, th=1)
        image_caption = DatasetItem(img_id, gender, args.split, caption, tokens)
        my_dataset_with_captions.append(image_caption)

    output_df = pd.DataFrame(my_dataset_with_captions)

    # Save raw output
    output_df.to_csv(output_file, index=False, sep="\t")
    print(f"{output_file} saved!")

    print(f"Stats:")
    print(f"{len(set(output_df['id']))} unique images.")
    print(output_df.info())
    print(output_df["gender"].value_counts())

    # Save clean output (discard images with no clear gender)
    output_df.drop(output_df[output_df.gender == "X"].index, inplace=True)
    output_df.to_csv(output_file.replace(".csv", "_clean.csv"), index=False, sep="\t")
    print(f'{output_file.replace(".csv", "_clean.csv")} saved!')


def _load_cc_captions(caption_path: str) -> pd.DataFrame:
    with open(caption_path) as fp:
        captions = json.load(fp)
    return captions


if __name__ == "__main__":
    args = parse_args()
    main(args)
