import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Optional


def get_seed_words() -> List[List[str], List[str], List[str]]:
    """Retrieve the list of seed words for each group"""
    mappings = pd.read_csv("Mappings.csv", sep=";")

    f = list(set(mappings["Female"]))
    m = list(set(mappings["Male"]))
    n = list(set(mappings["Neutral"]))
    return f, m, n


def get_mappings_gender_dict() -> List[Dict[str, str], Dict[str, str]]:
    """Retrieve the word mappings between a word from
    the female or male subsets to its gender-neutral equivalent"""
    mappings = pd.read_csv("Mappings.csv", sep=";")

    f = list(mappings["Female"])
    m = list(mappings["Male"])
    n = list(mappings["Neutral"])
    f_mappings = {k: v for k, v in zip(f, n)}
    m_mappings = {k: v for k, v in zip(m, n)}
    return f_mappings, m_mappings


def mark_gender_word(word: str, gender_dict: Dict[str, Any]) -> str:
    """Given a word, return its gender group"""
    if word in gender_dict["Female"]:
        g = "Female"
    elif word in gender_dict["Male"]:
        g = "Male"
    else:
        g = "Neutral"
    return g


def infer_image_gender(captions: List[str], nlp: Any, th: Optional[int] = 3) -> str:
    """Images are labelled as Male if the majority of its captions include a word
       from a set of male-related tokens (e.g., BOY), and no caption includes a word
       from the set of female-related tokens (e.g., GIRL); and vice-versa for Female.
       Images are labelled as Neutral if most of the subjects are listed as gender-neutral
       (e.g., PERSON), or if there is no majority gender mention in the texts.
       Finally, images are discarded from the analysis when the text mentions
       both male and female entities, or there are no people mentioned.
    Args:
        captions (List[str]): List of captions associated with an image.
        nlp (Any): NLP pipeline (SpaCy) to process the text in 'captions'.
        th (Optional[int], optional): Threshold that defines what we
        consider majority of captions. Defaults to 3.

    Returns:
        str: inferred gender group
    """
    genders = []
    # Loop over each caption
    for caption in captions:
        # Get gender per caption
        genders.append(infer_gender(caption, nlp))

    if len(set(genders)) == 1:
        if "Female" in genders:
            return "Female"
        elif "Male" in genders:
            return "Male"
        elif "Neutral" in genders:
            return "Neutral"
    elif genders.count("Female") >= th:
        return "Female"
    elif genders.count("Male") >= th:
        return "Male"
    elif genders.count("Neutral") >= th:
        return "Neutral"

    elif (genders.count("Female") < th and "Male" not in genders) or (
        genders.count("Male") < th and "Female" not in genders
    ):
        return "Neutral"
    elif "Male" in genders and "Female" in genders:
        return "X"  # This case will be later discarded

    return "X"  # This case will be later discarded


def infer_gender(caption: str, nlp: Any) -> str:
    """Get the gender mentioned in a given text (caption).
    We build upon a seed word list for each gender group considered.

    Args:
        caption (str): Sentence describing an image
        nlp (Any): NLP pipeline (SpaCy) to process the text in 'caption'

    Returns:
        str: inferred gender group associated with the text in 'caption'
    """
    # Get seed words classified by gender group
    female_tokens, male_tokens, neutral_tokens = get_seed_words()

    genders = []

    # Process text in caption
    doc = nlp(caption)
    # Loop over each token
    for token in doc:
        raw_token = token.text.lower()
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
            or ((token.dep_ == "attr" or token.dep_ == "conj") and token.pos_ == "NOUN")
            or (token.dep_ == "poss")
        ):
            if raw_token in female_tokens:
                genders.append("Female")
            elif raw_token in male_tokens:
                genders.append("Male")
            elif raw_token in neutral_tokens:
                genders.append("Neutral")

    if len(set(genders)) == 1:
        if "Female" in genders:
            return "Female"
        elif "Male" in genders:
            return "Male"
        else:
            return "Neutral"
    elif "Female" in genders and "Male" not in genders:
        return "Female"
    elif "Male" in genders and "Female" not in genders:
        return "Male"
    return ""  # No gender detected in this caption


def compute_classification_report(df: pd.DataFrame) -> Dict[str, str]:
    """Build a text report showing the main classification metrics.
    Call method from sklearn.metrics

    Args:
        df (pd.DataFrame): Pandas DataFrame with gender
          associated with each image.
          The column 'gender' shows the gender associated
          with an image computed with infer_image_gender().
          The column 'gender_pred' shows the gender associated
          with the word predicted by a model in an MLM task
          (see Section 4.1. in the paper)

    Returns:
        Dict[str, str]: Dictionary with the precision,
        recall, F1 score for each class.
    """
    # Predicting a gender-neutral term shows that the model understands
    # the depicted visual concept at the generic level (footnote 10).
    # Thus, we consider this prediction equally valid to having predicted
    # a word from the associated gender group in each case.
    df.loc[df["gender_pred"] == "Neutral", "gender_pred"] = df["gender"]
    # Make the report
    report = classification_report(
        df["gender"],
        df["gender_pred"],
        labels=["Female", "Male", "Neutral"],
        output_dict=True,
    )
    return report


def answer2lemma(df, field_name, nlp):
    df[field_name + "_lemma"] = ""
    df.sort_index(inplace=True)
    docs = nlp.pipe(df[field_name].tolist())
    for i, doc in enumerate(tqdm(docs, total=len(df))):
        for token in doc:
            df.loc[i, field_name + "_lemma"] += token.lemma_
    return df


def plot_confusion_matrix(
    gender_predictions_for_captions: Dict[str, Any],
    out_dir: str = None,
) -> None:
    """Depict a confusion matrix from
    the predictions in 'gender_predictions_for_captions'

    Args:
        gender_predictions_for_captions (Dict[str, Any]): The dictionary keys
          represent the true label.
          Values in each case represent labels predictions.
        out_dir (str): output directory to store the confusion matrix as an image.
          Defaults to None.
    """
    classes = list(gender_predictions_for_captions.keys())

    y_true = []
    y_pred = []
    for k, v in gender_predictions_for_captions.items():
        y_true += [classes.index(k)] * sum(v.values())
        for k_confusion, times in v.items():
            y_pred += [classes.index(k_confusion)] * times
        assert len(y_pred) == len(y_true)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(
        xlabel="Pred",
        ylabel="True",
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion matrix",
    )
    plt.yticks(rotation=0)
    if out_dir:
        plt.savefig(
            os.path.join(out_dir, "gender_confusion_matrix.png"), bbox_inches="tight"
        )
    else:
        plt.show()
