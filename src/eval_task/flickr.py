import sys
import os
import pandas as pd
import numpy as np
from typing import List

ROOT = "/Users/sxk199/PhD/experiments-output/multimodal/retrieval-flickr30k"
sweep = True
split = "test" #sys.argv[1]

RESULTS_IR = {
    "lxmert_original": f"{ROOT}/lxmert_original/lxmert_original-{split}_result.json",
    "lxmert_original_ckpt": f"{ROOT}/lxmert_original_ckpt/lxmert_original_ckpt-{split}_result.json",
    "lxmert_cc_ctrl": f"{ROOT}/lxmert_cc_ctrl/lxmert_cc_ctrl-{split}_result.json",
    "lxmert_cc_ctrl_ckpt": f"{ROOT}/lxmert_cc_ctrl_ckpt/lxmert_cc_ctrl_ckpt-{split}_result.json",
    "lxmert_cc_ctrl_small": f"{ROOT}/lxmert_cc_ctrl_small/lxmert_cc_ctrl_small-test_result.json",
    "lxmert_original_sanity_check": f"{ROOT}/lxmert_original_sanity_check/lxmert_original_sanity_check-test_result.json",
    "albef_4m": f"{ROOT}/albef_4m/",
    "albef_4m_coco_neutral": f"{ROOT}/albef_4m_coco_neutral/",
    "albef_4m_cc3m_neutral": f"{ROOT}/albef_4m_cc3m_neutral/",
    "albef_14m": f"{ROOT}/albef_14m/",
    "albef_14m_coco_neutral": f"{ROOT}/albef_14m_coco_neutral/",
    "albef_14m_cc3m_neutral": f"{ROOT}/albef_14m_cc3m_neutral/",
    "blip": f"{ROOT}/blip/",
    "blip_coco_neutral": f"{ROOT}/blip_coco_neutral/",
}
RESULTS_TR = {
    "lxmert_original": f"{ROOT}/lxmert_original/lxmert_original-{split}_result.pkl",
    "lxmert_original_ckpt": f"{ROOT}/lxmert_original_ckpt/lxmert_original_ckpt-{split}_result.pkl",
    "lxmert_cc_ctrl": f"{ROOT}/lxmert_cc_ctrl/lxmert_cc_ctrl-{split}_result.pkl",
    "lxmert_cc_ctrl_ckpt": f"{ROOT}/lxmert_cc_ctrl_ckpt/lxmert_cc_ctrl_ckpt-{split}_result.pkl",
    "lxmert_cc_ctrl_small": f"{ROOT}/lxmert_cc_ctrl_small/lxmert_cc_ctrl_small-test_result.pkl",
    "lxmert_original_sanity_check": f"{ROOT}/lxmert_original_sanity_check/lxmert_original_sanity_check-test_TR_result.pkl",
    "albef_4m": f"{ROOT}/albef_4m/",
    "albef_4m_coco_neutral": f"{ROOT}/albef_4m_coco_neutral/",
    "albef_4m_cc3m_neutral": f"{ROOT}/albef_4m_cc3m_neutral/",
    "albef_14m": f"{ROOT}/albef_14m/",
    "albef_14m_coco_neutral": f"{ROOT}/albef_14m_coco_neutral/",
    "albef_14m_cc3m_neutral": f"{ROOT}/albef_14m_cc3m_neutral/",
    "blip": f"{ROOT}/blip/",
    "blip_coco_neutral": f"{ROOT}/blip_coco_neutral/",
}

RESULTS_SWEEP_IR = {
    # "lxmert_original": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_original/flickr30k/results_SEED/lxmert_original-{split}{split}_result.json",
    # "lxmert_original_ckpt": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_original_ckpt/flickr30k/results_SEED/lxmert_original_ckpt-{split}{split}_result.json",
    "lxmert_cc_ctrl": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_cc_ctrl/flickr30k/results_SEED/lxmert_cc_ctrl-{split}{split}_result.json",
    # "lxmert_cc_ctrl_ckpt": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_cc_ctrl_ckpt/flickr30k/results_SEED/lxmert_cc_ctrl_ckpt-{split}{split}_result.json"
}

RESULTS_SWEEP_TR = {
    # "lxmert_original": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_original/flickr30k/results_SEED/lxmert_original-{split}{split}_TR_result.pkl",
    # "lxmert_original_ckpt": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_original_ckpt/flickr30k/results_SEED/lxmert_original_ckpt-{split}{split}_TR_result.pkl",
    "lxmert_cc_ctrl": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_cc_ctrl/flickr30k/results_SEED/lxmert_cc_ctrl-{split}{split}_TR_result.pkl",
    # "lxmert_cc_ctrl_ckpt": f"/Users/sxk199/mnt/nlp/data/multimodal-gender-bias/sweeps/lxmert_cc_ctrl_ckpt/flickr30k/results_SEED/lxmert_cc_ctrl_ckpt-{split}{split}_TR_result.pkl"
}

def get_mappings():
    try:
        mappings = pd.read_csv("/Users/sxk199/PhD/code/multimodal-gender-bias/src/bias/Mappings.csv", sep=";")
    except FileNotFoundError:
        mappings = pd.read_csv("/home/sxk199/code/multimodal-gender-bias/src/bias/Mappings.csv", sep=";")

    f = list(set(mappings["Female"]))
    m = list(set(mappings["Male"]))
    n = list(set(mappings["Neutral"]))
    return f, m, n


def recall_ir(df: pd.DataFrame, val_samples: int, valid_idxs: List = None, title: bool = True):
    ansall = df['answer'].tolist()
    tgtall = df['target'].tolist()
    idxs = df['caption_idx'].tolist()
    rank_vector = np.empty(val_samples)
    rank_vector[:] = np.nan
    for ans, tgt, idx in zip(ansall, tgtall, idxs):
        if (valid_idxs is None) or (valid_idxs is not None and idx in valid_idxs):
            rank = np.where((
                                    np.asarray(ans)
                                    == np.asarray(tgt)
                            ) == 1
                            )[0][0]
            rank_vector[idx] = rank
        else:
            rank_vector[idx] = np.nan
    rank_vector = rank_vector[~np.isnan(rank_vector)]
    r1 = 100.0 * np.sum(rank_vector < 1) / len(rank_vector)
    r5 = 100.0 * np.sum(rank_vector < 5) / len(rank_vector)
    r10 = 100.0 * np.sum(rank_vector < 10) / len(rank_vector)
    medr = np.floor(np.median(rank_vector) + 1)
    meanr = np.mean(rank_vector) + 1
    if title:
        print("**************** Image Retrieval *****************")
    #     print(
    #         "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
    #         % (r1, r5, r10, medr, meanr)
    #     )
    print(
        "Final r1:%.3f" % (r1)
    )
    return r1, r5, r10, medr, meanr


def recall_tr(results: List, val_samples: int, valid_idxs: List = None, title: bool = True):
    rank_vector = np.empty(val_samples)
    rank_vector[:] = np.nan
    for item in results:
        if (valid_idxs is None) or (valid_idxs is not None and item['image_idx'] in valid_idxs):
            ranks = []
            tgt_captions = item['target']
            sorted_scores = item['answer']
            for tgt_caption in tgt_captions:
                ranks.append(np.where((np.asarray(sorted_scores) == np.asarray(tgt_caption)) == 1)[0][0])
            rank_vector[item['image_idx']] = min(ranks)

        else:
            rank_vector[item['image_idx']] = np.nan

    rank_vector = rank_vector[~np.isnan(rank_vector)]
    r1 = 100.0 * np.sum(rank_vector < 1) / len(rank_vector)
    r5 = 100.0 * np.sum(rank_vector < 5) / len(rank_vector)
    r10 = 100.0 * np.sum(rank_vector < 10) / len(rank_vector)
    medr = np.floor(np.median(rank_vector) + 1)
    meanr = np.mean(rank_vector) + 1
    if title:
        print("**************** Text Retrieval ******************")
    #     print(
    #         "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
    #         % (r1, r5, r10, medr, meanr)
    #     )
    print(
        "Final r1:%.3f" % (r1)
    )
    return r1, r5, r10, medr, meanr


def recall_ir_by_gender(df_ans: pd.DataFrame, df: pd.DataFrame):
    targets_flat = df_ans['target'].tolist()

    gender = "Male"
    df_gender_idx = df.loc[(df["gender"] == gender)].index
    target_idxs = []
    for idx in df_gender_idx:
        target_idx = np.where(np.asarray(targets_flat) == int(idx))
        if target_idx:
            target_idxs.extend(target_idx[0].tolist())
    df_cc_ = df_ans.iloc[target_idxs]
    print("gender:", gender, "samples:", len(df_cc_))
    recall_ir(df_cc_, len(df_ans), target_idxs, title=False)

    gender = "Female"
    df_gender_idx = df.loc[(df["gender"] == gender)].index
    target_idxs = []
    for idx in df_gender_idx:
        target_idx = np.where(np.asarray(targets_flat) == int(idx))
        if target_idx:
            target_idxs.extend(target_idx[0].tolist())
    df_cc_ = df_ans.iloc[target_idxs]
    print("gender:", gender, "samples:", len(df_cc_))
    recall_ir(df_cc_, len(df_ans), target_idxs, title=False)

    gender = "Unsure"
    df_gender_idx = df.loc[(df["gender"] == gender)].index
    target_idxs = []
    for idx in df_gender_idx:
        target_idx = np.where(np.asarray(targets_flat) == int(idx))
        if target_idx:
            target_idxs.extend(target_idx[0].tolist())
    df_cc_ = df_ans.iloc[target_idxs]
    print("gender:", gender, "samples:", len(df_cc_))
    recall_ir(df_cc_, len(df_ans), target_idxs, title=False)


def recall_tr_by_gender(df_ans_list: List, df: pd.DataFrame):
    gender = "Male"
    df_gender_idx = df.loc[(df["gender"] == gender)].index
    df_ans_list_ = [item for item in df_ans_list if item["image_idx"] in df_gender_idx]
    print("gender:", gender, "samples:", len(df_ans_list_))
    recall_tr(df_ans_list_, len(df_ans_list), df_gender_idx, title=False)

    gender = "Female"
    df_gender_idx = df.loc[(df["gender"] == gender)].index
    df_ans_list_ = [item for item in df_ans_list if item["image_idx"] in df_gender_idx]
    print("gender:", gender, "samples:", len(df_ans_list_))
    recall_tr(df_ans_list_, len(df_ans_list), df_gender_idx, title=False)

    gender = "Unsure"
    df_gender_idx = df.loc[(df["gender"] == gender)].index
    df_ans_list_ = [item for item in df_ans_list if item["image_idx"] in df_gender_idx]
    print("gender:", gender, "samples:", len(df_ans_list_))
    recall_tr(df_ans_list_, len(df_ans_list), df_gender_idx, title=False)




# LOAD GENDER LIST
FEMALE_TOKENS, MALE_TOKENS, NEUTRAL_TOKENS = get_mappings()

with open("/Users/sxk199/PhD/code/multimodal-gender-bias/src/bias/reducingbias/objs_flickr30k") as f:
    top_objects = f.readlines()
    top_objects = [obj.rstrip() for obj in top_objects]

# LOAD DF WITH GENDER FROM CAPTIONS
df30k = pd.read_pickle("/Users/sxk199/PhD/data/flickr30k/valid_ann_gender.pkl")

map_id2image_idx = dict(zip(df30k.index, df30k.id))
# map_image_idx2id = dict(zip(df30k.id, df30k.index))

map_captionid_2_text = {}
counter = 0
for id_, image_idx in map_id2image_idx.items():
    sentences = df30k.loc[df30k['id'] == image_idx, 'sentences'].values[0]
    for caption in sentences:
        map_captionid_2_text[counter] = caption
        counter += 1

df30k.drop(df30k[df30k.gender == "X"].index, inplace=True)
df30k.drop(df30k[df30k.gender == "Empty"].index, inplace=True)
gender_counter = df30k["gender"].value_counts()

if not sweep:
    for modelname, path in RESULTS_IR.items():
        print(modelname)
        if modelname.startswith("lxmert"):
            print(modelname)
            df_cc = pd.read_json(path)
            recall_ir(df_cc, len(df_cc))
            if split == "val":
                recall_ir_by_gender(df_cc, df30k)
            print("")
            df_cctr_list = pd.read_pickle(RESULTS_TR[modelname])
            recall_tr(df_cctr_list, len(df_cctr_list))
            if split == "val":
                recall_tr_by_gender(df_cctr_list, df30k)
            print("")
        else:
            preds = np.load(os.path.join(path, f"score_{split}_t2i.npy"))
            df = pd.DataFrame(columns=["caption_idx", "answer", "target"])
            df["caption_idx"] = list(range(0, preds.shape[0]))
            df["target"] = [i//5 for i in range(preds.shape[0])]
            df["answer"] = [np.argsort(s)[::-1] for s in preds]
            recall_ir(df, len(df))
            if split == "val":
                recall_ir_by_gender(df, df30k)
            print("")

            preds = np.load(os.path.join(path, f"score_{split}_i2t.npy"))
            preds_in_list = [
                {
                    "image_idx": i,
                    "target": list(range(5*i, 5*i+5)),
                    "answer": np.argsort(item)[::-1]
                }
                for i, item in enumerate(preds)
            ]
            recall_tr(preds_in_list, len(preds_in_list))
            if split == "val":
                recall_tr_by_gender(preds_in_list, df30k)
else:
    for modelname, path in RESULTS_SWEEP_IR.items():
        print(modelname)
        results_ir = []
        results_tr = []
        if modelname == "lxmert_original":
            results_ir.append(53.0)
            results_tr.append(61.1)
        elif modelname == "lxmert_cc_ctrl":
            results_ir.append(54.4)
            results_tr.append(59.5)
        elif modelname == "lxmert_original_ckpt":
            results_ir.append(53.9)
            results_tr.append(66.2)
        elif modelname == "lxmert_cc_ctrl_ckpt":
            results_ir.append(50.2)
            results_tr.append(57.4)

        for s in [0, 23, 42, 56, 92]:
            if os.path.exists(path.replace("SEED", str(s))):
                df_cc = pd.read_json(path.replace("SEED", str(s)))
                r1, _, _, _, _ = recall_ir(df_cc, len(df_cc), title=False)
                results_ir.append(r1)
                df_cctr_list = pd.read_pickle(RESULTS_SWEEP_TR[modelname].replace("SEED", str(s)))
                r1, _, _, _, _ = recall_tr(df_cctr_list, len(df_cctr_list), title=False)
                results_tr.append(r1)
            else:
                print(f"Skipping {path.replace('SEED', str(s))}")
        print("F30k IR Mean (test): {:.2f}".format(np.mean(results_ir)))
        print("F30k IR Std (test): {:.2f}".format(np.std(results_ir)))
        print("IR", results_ir)
        print("F30k TR Mean (test): {:.2f}".format(np.mean(results_tr)))
        print("F30k TR Std (test): {:.2f}".format(np.std(results_tr)))
        print("TR", results_tr)
        print()
    """
    lxmert_original
    Final r1:53.620
    Final r1:60.700
    Final r1:52.280
    Final r1:59.700
    Final r1:54.600
    Final r1:66.300
    Final r1:48.380
    Final r1:58.700
    Final r1:55.240
    Final r1:65.700
    F30k IR Mean (test): 52.82
    F30k IR Std (test): 2.44
    F30k TR Mean (test): 62.22
    F30k TR Std (test): 3.16
    
    lxmert_original_ckpt
    Final r1:53.900
    Final r1:63.600
    Final r1:52.480
    Final r1:60.600
    Final r1:54.820
    Final r1:65.000
    Final r1:49.460
    Final r1:56.400
    Final r1:54.880
    Final r1:63.500
    F30k IR Mean (test): 52.97
    F30k IR Std (test): 2.24
    F30k TR Mean (test): 62.02
    F30k TR Std (test): 3.12
    
    lxmert_cc_ctrl

    
    lxmert_cc_ctrl_ckpt
    Final r1:50.180
    Final r1:57.400
    Final r1:47.080
    Final r1:55.000
    Final r1:44.460
    Final r1:51.700
    Final r1:46.240
    Final r1:54.600
    Final r1:46.480
    Final r1:54.900
    Final r1:47.040
    Final r1:54.900
    F30k IR Mean (test): 46.91
    F30k IR Std (test): 1.70
    F30k TR Mean (test): 54.75
    F30k TR Std (test): 1.66



    """