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

# LOAD QUESTIONS
q_path = f"{root}/questions/v2_OpenEnded_mscoco_train2014_questions.json"
qdf = pd.DataFrame(json.load(open(q_path))["questions"])
qdf.sort_values("question_id", ascending=True, inplace=True)
imgid_vqa = list(set(qdf["image_id"]))

# LOAD ANSWERS
ans_path = f"{root}/annotations/v2_mscoco_train2014_annotations.json"
ansdf = pd.DataFrame(json.load(open(ans_path))["annotations"])
ansdf.sort_values("question_id", ascending=True, inplace=True)
assert len(qdf) == len(ansdf)

qdf["question_type"] = ansdf["question_type"]
qdf["multiple_choice_answer"] = ansdf["multiple_choice_answer"]

# GENERATE DF WITH GENDER FROM QUESTIONS
qdf_ = qdf.copy()
qdf_["q_gender"] = np.nan
questions = set(qdf_["question"].tolist())

for q in tqdm(questions, leave=False):
    gender = infer_gender(q, nlp)
    indexes = qdf_.loc[qdf_["question"] == q].index
    for idx in indexes:
        qdf_.at[idx, "q_gender"] = gender

qdf_.to_pickle(output_file)

print(f"Final gender distribution in VQAv2:")
print(qdf_["q_gender"].value_counts())

"""
48 objects
Final gender distribution if VQAv2 validation split:
          166307
Male       20000 (9.33%)
Neutral     18549 (8.65%)
Female      9498 (4.43%)
"""
