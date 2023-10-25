import sys
import json
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
from bias.tools.utils import infer_gender

root = sys.argv[1]
output_file = sys.argv[2]

nlp = spacy.load("en_core_web_sm")

# LOAD QA
gqa_path = f"{root}/annotations/train.json"
gqadf = pd.DataFrame(json.load(open(gqa_path)))
images_val = list(set(gqadf["img_id"].values))

# GENERATE DF WITH GENDER FROM QUESTIONS
questions = set(gqadf["sent"].tolist())
for q in tqdm(questions, leave=False):
    gender = infer_gender(q, nlp)
    indexes = gqadf.loc[gqadf["sent"] == q].index
    for idx in indexes:
        gqadf.at[idx, "q_gender"] = gender

gqadf.to_pickle(output_file)

print(f"Final gender distribution in GQA:")
print(gqadf["q_gender"].value_counts())

"""
10234 unique images in GQA val. 132062 entries.
Final gender distribution if GQA validation split:
Male      8265
Female    4860
Neutral    4442
"""
