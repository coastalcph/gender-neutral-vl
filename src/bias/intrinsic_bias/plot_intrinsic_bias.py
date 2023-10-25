import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "/projects/nlp/data/data/multimodal-gender-bias/outputs"


def print_gender_distr(model, df):
    m = len(df.query("gender_pred=='M'").index.values)
    f = len(df.query("gender_pred=='F'").index.values)
    n = len(df.query("gender_pred=='N'").index.values)
    total = m + f + n
    print(
        f"{model}: {100*m/total}% Male, {100*f/total}% Female, {100*n/total}% Neutral"
    )


data_path = {
    "coco": f"{BASE_DIR}/eval_on_coco/#MODEL/mapped_captions_valid_preds.csv",
    "cc3m": f"{BASE_DIR}/eval_on_cc/#MODEL/mapped_captions_val_preds.csv",
}
models = [
    "lxmert_180k",
    "lxmert_180k_neutral",
    "lxmert_3m",
    "lxmert_3m_neutral",
]

df_results = pd.DataFrame(
    columns=["Precision", "Recall", "F1", "support", "label", "dataset", "model"]
)
labels = ["M", "F", "N"]

for dataset, file_ in data_path.items():
    for modelname in models:
        df = pd.read_csv(file_.replace("#MODEL", modelname), sep="\t")
        df = df.dropna(subset=["gender_pred"])
        df["gender"] = df["gender"].apply(lambda x: x.replace("Male", "M"))
        df["gender_pred"] = df["gender_pred"].apply(lambda x: x.replace("Male", "M"))
        df["gender"] = df["gender"].apply(lambda x: x.replace("Female", "F"))
        df["gender_pred"] = df["gender_pred"].apply(lambda x: x.replace("Female", "F"))
        df["gender"] = df["gender"].apply(lambda x: x.replace("Neutral", "N"))
        df["gender_pred"] = df["gender_pred"].apply(lambda x: x.replace("Neutral", "N"))

        print_gender_distr(modelname, df)

        mask = df.query("gender_pred=='N'").index.values
        df.loc[mask, "gender_pred"] = df.loc[mask, "gender"]

        prec, rec, f1, support = precision_recall_fscore_support(
            df["gender"], df["gender_pred"], average=None, labels=labels
        )

        for ilabel, label in enumerate(labels):
            df_results.loc[modelname + "_" + label + "_" + dataset] = [
                prec[ilabel],
                rec[ilabel],
                f1[ilabel],
                support[ilabel],
                label,
                dataset,
                modelname,
            ]


fig, axs = plt.subplots(len(models), 3, sharey=False, sharex=False)
df = df_results.query('label in ["M", "F", "N"] and dataset in ["coco", "cc"]')
columns = ["Precision", "Recall", "F1"]
for irow, modelname in enumerate(models):
    for icol, col in enumerate(columns):
        axs[irow, icol].xaxis.tick_top()
        axs[0, icol].set_title(col)
        df_tmp = (
            df.query("model==@modelname")[[col, "dataset", "label"]]
            .pivot_table(index="label", columns="dataset")
            .round(2)
            .reindex(columns=["coco", "cc"], level="dataset")
            .reindex(["M", "F", "N"])
        )

        im = sns.heatmap(
            df_tmp,
            annot=True,
            xticklabels=True,
            ax=axs[irow, icol],
            vmin=0.6,
            vmax=1,
            cbar=None,
        )
        if icol > 0:
            axs[irow, icol].set_ylabel("")
        else:
            if modelname == "lxmert_180k":
                axs[irow, 0].set_ylabel("$LXMERT_{180K}$", fontsize=10, rotation=95)
            elif modelname == "lxmert_180k_neutral":
                axs[irow, 0].set_ylabel("$LXMERT_{180K}^{N}$", fontsize=10, rotation=95)
            elif modelname == "lxmert_3m":
                axs[irow, 0].set_ylabel("$LXMERT_{3M}$", fontsize=10, rotation=95)
            else:
                axs[irow, 0].set_ylabel("$LXMERT_{3M}^{N}$", fontsize=10, rotation=95)
        axs[irow, icol].set_xlabel("")
        if irow == 0:
            axs[irow, icol].set_xticklabels(["COCO", "CC3M"], rotation=0)
        else:
            axs[irow, icol].set_xticks([])

        if icol > 0:
            axs[irow, icol].set_yticks([])

fig.subplots_adjust(bottom=0.1)
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
mappable = im.get_children()[0]
plt.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(f"intrinsic_bias.png", dpi=300, bbox_inches="tight")
