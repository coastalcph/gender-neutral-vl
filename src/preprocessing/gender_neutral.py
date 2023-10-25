import os
import sys
import spacy
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from bias.tools.utils import get_mappings, get_mappings_gender_dict


class GenderNeutralData:
    """Create files to continue pretraining on gender-neutral data.
    This step is needed for COCO and CC3M datasets.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.FEMALE_TOKENS, self.MALE_TOKENS, self.NEUTRAL_TOKENS = get_mappings()
        self.ALL_TOKENS = (
            set(self.FEMALE_TOKENS)
            .union(set(self.MALE_TOKENS))
            .union(set(self.NEUTRAL_TOKENS))
        )
        self.FEMALE_MAPPINGS, self.MALE_MAPPINGS = get_mappings_gender_dict()

    @staticmethod
    def _filename_to_id(filename: str, sp) -> str:
        filename_ = filename[len(f"COCO_{sp}2014_") :]
        id_ = int(filename_)
        return str(id_)

    def _simple_mapper(self, input_strings, img_genders, filename):
        docs = self.nlp.pipe(input_strings)
        all_mapped = []
        all_masked = []
        i = 0
        for doc in tqdm(
            docs, total=len(input_strings), desc=f"Processing captions in {filename}..."
        ):
            mapped = ""
            masked = ""
            nsubj_masked = False
            for token in doc:
                raw_token = token.text.lower()
                token_lemma = token.lemma_.lower()
                if (
                    (
                        token.dep_ == "nsubj"
                        and (
                            token.pos_ == "NOUN"
                            or token.pos_ == "PRON"
                            or token.pos_ == "PROPN"
                        )
                    )
                    or (
                        token.dep_ == "nsubjpass"
                        and (
                            token.pos_ == "NOUN"
                            or token.pos_ == "PRON"
                            or token.pos_ == "PROPN"
                        )
                    )
                    or (
                        token.dep_ == "ROOT"
                        and (
                            token.pos_ == "NOUN"
                            or token.pos_ == "PRON"
                            or token.pos_ == "PROPN"
                        )
                    )
                    or (token.dep_ == "attr" and token.pos_ == "NOUN")
                    or (token.dep_ == "poss" and nsubj_masked)
                ):
                    if token_lemma in self.MALE_TOKENS:
                        mapped += (
                            self.MALE_MAPPINGS[raw_token]
                            if raw_token in self.MALE_MAPPINGS
                            else self.MALE_MAPPINGS[token_lemma]
                        )
                        masked += "[MASK]"
                        nsubj_masked = True
                    elif token_lemma in self.FEMALE_TOKENS:
                        mapped += (
                            self.FEMALE_MAPPINGS[raw_token]
                            if raw_token in self.FEMALE_MAPPINGS
                            else self.FEMALE_MAPPINGS[token_lemma]
                        )
                        masked += "[MASK]"
                        nsubj_masked = True
                    else:
                        mapped += raw_token
                        masked += "[MASK]"
                        nsubj_masked = True
                else:
                    mapped += raw_token
                    masked += raw_token
                mapped += token.whitespace_
                masked += token.whitespace_
            masked, mapped = self._review_spacy_errors(
                masked, mapped, img_genders[i], nsubj_masked
            )
            i += 1
            all_masked.append(masked)
            all_mapped.append(mapped)
        return {"masked": all_masked, "mapped": all_mapped}

    def _review_spacy_errors(self, mask_sent, map_sent, gender, nsubj_masked):
        # Correct for sentences where SpaCy fails to detect a man/woman/neutral subject in the sentence
        if not nsubj_masked and (gender == "Female"):
            for word in self.FEMALE_TOKENS:
                if word in mask_sent.split():
                    mask_sent = mask_sent.replace(word, "[MASK]")
                    map_sent = map_sent.replace(word, f"{self.FEMALE_MAPPINGS[word]}")
                    nsubj_masked = True
            if not nsubj_masked:
                for word in self.NEUTRAL_TOKENS:
                    if word in mask_sent.split():
                        mask_sent = mask_sent.replace(word, "[MASK]")
        elif not nsubj_masked and (gender == "Male"):
            for word in self.MALE_TOKENS:
                if word in mask_sent.split():
                    mask_sent = mask_sent.replace(word, "[MASK]")
                    map_sent = map_sent.replace(word, f"{self.MALE_MAPPINGS[word]}")
                    nsubj_masked = True
            if not nsubj_masked:
                for word in self.NEUTRAL_TOKENS:
                    if word in mask_sent.split():
                        mask_sent = mask_sent.replace(word, "[MASK]")
        elif not nsubj_masked and gender == "Unsure":
            for word in self.NEUTRAL_TOKENS:
                if word in mask_sent.split():
                    mask_sent = mask_sent.replace(word, "[MASK]")
            for word in self.FEMALE_TOKENS:
                if word in mask_sent.split():
                    mask_sent = mask_sent.replace(word, "[MASK]")
                    map_sent = map_sent.replace(word, f"{self.FEMALE_MAPPINGS[word]}")
            for word in self.MALE_TOKENS:
                if word in mask_sent.split():
                    mask_sent = mask_sent.replace(word, "[MASK]")
                    map_sent = map_sent.replace(word, f"{self.MALE_MAPPINGS[word]}")

        return mask_sent, map_sent

    def map_files(
        self, dataset: str, split_: str, input_dir: str, output_dir: str
    ) -> None:
        # Define output files
        output_caption_file = f"{output_dir}/caption_{split_}.json"
        output_masked_file = f"{output_dir}/masked_captions_{split_}.json"
        print(
            f"Output caption file (for measuring intrinsic bias): {output_caption_file}"
        )
        print(f"Output masked file (for MLM): {output_masked_file}")

        files_to_handle = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(f"{split_}.csv")
        ]

        for file in files_to_handle:
            print(f"Processing file {file}...")
            captions = pd.read_csv(file, sep="\t")

            gender = captions["gender"].values

            normalized = self._simple_mapper(captions["caption"].values, gender, file)
            normalized = pd.DataFrame(normalized)
            normalized.index = captions.index

            captions["masked"] = normalized["masked"]
            captions["mapped"] = normalized["mapped"]
            # captions["targets"] = normalized["targets"]

            # Save json file in the right format to continue pretraining with gender neutral data
            if dataset == "cc3m":
                # 1 caption per image
                dic = pd.Series(captions.mapped.values, index=captions.id).to_dict()
            else:
                dic = {}
                img_ids = list(set(captions["id"]))
                for id_ in img_ids:
                    dic[id_] = list(
                        captions.loc[captions["id"] == id_, "mapped"].values
                    )
            json.dump(dic, open(output_caption_file, "w+"))
            print(f"{output_caption_file} saved!")

            # Format
            id_masked_caps = defaultdict(list)
            for _, row in captions.iterrows():
                img_id = row.values[0]
                caps = row.values[-2]
                id_masked_caps[str(img_id)].append(caps)
            with open(output_masked_file, "w+") as f:
                json.dump(id_masked_caps, f, indent=2)
            print(f"{output_masked_file} saved!")


if __name__ == "__main__":
    dataset = sys.argv[1]
    split_ = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]

    # Validate inputs
    assert dataset in ["cc3m", "coco"]
    assert split_ in ["train", "val"]
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    GN = GenderNeutralData()
    GN.map_files(dataset, split_, input_dir, output_dir)
