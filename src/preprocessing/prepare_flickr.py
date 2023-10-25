import sys
import spacy
import pandas as pd
from bias.tools.utils import infer_image_gender

root = sys.argv[1]
output_file = sys.argv[2]

nlp = spacy.load("en_core_web_sm")

# LOAD RETRIEVAL DATA
path = f"{root}/annotations/train_ann.jsonl"
df = pd.read_json(path, lines=True)
images_val = list(set(df["id"].values))

# INFER GENDER FROM CAPTIONS
df["gender"] = ""
questions = df["sentences"]
for i, sents in enumerate(questions):
    df.at[i, "gender"] = infer_image_gender(sents, nlp, th=3)

mq = df.loc[df["gender"] == "Male"]
# print(f"{len(set(mq['sentences']))} unique male questions")
fq = df.loc[df["gender"] == "Female"]
# print(f"{len(set(fq['sentences']))} unique female questions")
uq = df.loc[df["gender"] == "Unsure"]
# print(f"{len(set(uq['sentences']))} unique unsure questions")
print(
    "{:.1f}% questions refer to people".format(
        100 * (len(mq) + len(fq) + len(uq)) / len(df)
    )
)

df.to_pickle(output_file)

df.drop(df[df.gender == "X"].index, inplace=True)

print(f"Final gender distribution in Flickr30K:")
print(df["gender"].value_counts())

"""
1014 unique images in flickr30k val (5070 captions approx)
87.6% questions refer to people
Final gender distribution in Flickr30K validation split:
Male      345
Neutral    336
Female    207
"""
