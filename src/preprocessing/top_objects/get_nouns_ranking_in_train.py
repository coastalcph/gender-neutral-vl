import os
import sys
import json
import pandas as pd
import spacy
from collections import Counter, defaultdict, OrderedDict


def main(dataset, base_file_with_genders, output_dir):
    # Define output
    output_file = os.path.join(
        output_dir, f"labels_in_gender_text_{dataset}_train.json"
    )

    # NLP pipeline
    nlp = spacy.load("en_core_web_sm")

    if dataset == "flickr":
        data = pd.read_pickle(base_file_with_genders)
        data = data[data.gender.isin(["Male", "Female", "Neutral"])]
        sentences = [item for captions in data["sentences"].values for item in captions]
    elif dataset == "vqav2":
        qdf = pd.read_pickle(base_file_with_genders)
        qdf = qdf.loc[qdf["q_gender"] != ""]
        answers = qdf.multiple_choice_answer.tolist()
    elif dataset == "gqa":
        qdf = pd.read_pickle(base_file_with_genders)
        qdf = qdf.loc[qdf["q_gender"] != ""]
        answers = [k for item in qdf.label.tolist() for k, v in item.items()]
    elif dataset in ["coco", "cc3m"]:
        data = pd.read_pickle(base_file_with_genders)
        data = data[data.gender.isin(["Male", "Female", "Neutral"])]
        if dataset == "coco":
            sentences = [
                item for captions in data["captions"].values for item in captions
            ]
        else:
            sentences = data["captions"].values
    else:
        print("Not supported")
        sys.exit(0)

    if dataset in ["flickr", "coco", "cc3m"]:
        # Process captions (sentences)
        docs = nlp.pipe(list(set(sentences)))
        # Lemmatize
        answers_lemma_all = []
        for doc in docs:
            for token in doc:
                # We are interested only in NOUN tokens
                if token.pos_ == "NOUN":
                    ans_lemma = token.lemma_
                    answers_lemma_all.append(ans_lemma)

        freq = Counter(list(answers_lemma_all))
        freq_dict = defaultdict(int)
        for x in OrderedDict(freq.most_common()).items():
            freq_dict[x[0]] = x[1]
    else:  # vqav2, gqa
        # Process answers (not sentences)
        docs = nlp.pipe(list(set(answers)))
        # Lemmatize
        answers_lemma = {}
        for doc in docs:
            ans = doc.text
            ans_lemma = ""
            for token in doc:
                ans_lemma += token.lemma_ + " "
            ans_lemma = ans_lemma.rstrip()
            answers_lemma[ans] = ans_lemma

        freq = Counter(answers)
        freq_dict = defaultdict(int)
        for x in OrderedDict(freq.most_common()).items():
            freq_dict[answers_lemma[x[0]]] += x[1]

    # Save dictionary in a json file
    json.dump(
        freq_dict,
        open(output_file, "w"),
        indent=2,
    )
    print(f"{len(freq_dict)} items written in {output_file}")


if __name__ == "__main__":
    dataset = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]

    main(dataset, input_file, output_dir)
