# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Modified: 05 October 2022, Laura Cabello
import os
import sys
import h5py
# import pymp
import json
import jsonlines
import _pickle as cPickle
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm


def get_neighbors(image_list, image_feature, kdt, dataset):
    total_image = len(image_list)
    batch_size = 1
    p_length = int(total_image / batch_size)
    # hard_pool = pymp.shared.array((total_image, 100))
    # with pymp.Parallel(40) as p:
    #   for index in p.range(0, p_length):
    hard_pool = np.zeros((total_image, 100))
    for index in range(0, total_image):
        if dataset == "mscoco":
            ind = kdt.query(image_feature[index: index + 1][()].T, k=100, return_distance=False)
        else:
            ind = kdt.query(image_feature[index: index + 1], k=100, return_distance=False)
        hard_pool[index] = ind
        # print("finish worker", index)
    return hard_pool


def process_flickr30k(inputImg, inputJson, cache_root):
    with h5py.File(inputImg, "r") as features_h5:
        _image_ids = list(features_h5["image_ids"])

    train_image_list = []
    with jsonlines.open(inputJson) as reader:
        # Build an index which maps image id with a list of caption annotations.
        for annotation in reader:
            train_image_list.append(int(annotation["img_path"].split(".")[0]))

    num_train = len(train_image_list)
    print(len(train_image_list))

    train_image_feature = pymp.shared.array([num_train, 2048])

    with pymp.Parallel(40) as p:
        with h5py.File(inputImg, "r", libver="latest", swmr=True) as features_h5:
            for i in p.range(0, num_train):
                image_id = train_image_list[i]
                index = _image_ids.index(image_id)
                num_boxes = int(features_h5["num_boxes"][index])
                feature = features_h5["features"][index]
                train_image_feature[i] = feature[:num_boxes].sum(0) / num_boxes
                print("finish worker", i)

    kdt = BallTree(train_image_feature[:, :], metric="euclidean")
    print("finish create the ball tree")

    train_hard_pool = get_neighbors(train_image_list, train_image_feature, kdt, dataset="f30k")

    # save the pool info into
    cache_file = os.path.join(cache_root, "hard_negative.pkl")
    save_file = {}
    save_file["train_hard_pool"] = train_hard_pool
    save_file["train_image_list"] = train_image_list
    cPickle.dump(save_file, open(cache_file, "wb"))


def process_mscoco(images_dir, cache_imgs, outputdir):
    # print("loading entries from %s" % cache_imgs)
    # entries = cPickle.load(open(cache_imgs, "rb"))
    # train_image_list = [item['image_id'] for item in entries]
    splits_ids_path = "/image/nlp-datasets/laura/data/mscoco/datasets/multimodal-gender-bias/lxmert/gender-mappings/dataset_masked_splits.json"
    train_image_list = json.load(open(splits_ids_path, "r"))["train_images"]
    num_train = len(train_image_list)
    print(len(train_image_list), "images in training")

    features_h5 = h5py.File(os.path.join(images_dir, "image_features.hdf5"), "r", libver="latest", swmr=True)
    train_boxes = h5py.File(os.path.join(images_dir, "boxes_train_and_val.h5"), "r")

    train_image_feature = np.zeros((num_train, 2048))
    i=0
    for image_id in tqdm(train_image_list):
        feature = features_h5[str(image_id)][()]
        num_boxes = len(train_boxes[str(image_id)][()])
        train_image_feature[i] = feature[:num_boxes].sum(0) / num_boxes
        i+=1
        
    kdt = BallTree(train_image_feature.reshape(-1, 1), metric="euclidean")
    print("finish create the ball tree")
    print("Getting neighbours...")
    train_hard_pool = get_neighbors(train_image_list, train_image_feature, kdt, dataset="mscoco")

    # save the pool info into
    cache_file = os.path.join(outputdir, "hard_negative.pkl")
    save_file = {}
    save_file["train_hard_pool"] = train_hard_pool
    save_file["train_image_list"] = train_image_list
    cPickle.dump(save_file, open(cache_file, "wb"))


if __name__ == "__main__":
    if sys.argv[1] == "f30k":
        inputImg = "data/flick30k/flickr30k.h5"
        inputJson = "data/flick30k/all_data_final_train_2014.jsonline"
        cache_root = "data/flick30k"
        process_flickr30k(inputImg, inputJson, cache_root)
    elif sys.argv[1] == "mscoco":
        images_dir = "/image/nlp-datasets/laura/data/mscoco/datasets/multimodal-gender-bias/images/"
        cache_imgs = "/image/nlp-datasets/laura/data/mscoco/datasets/multimodal-gender-bias/lxmert/cache/RetrievalCOCO_train_bert-base-uncased_30.pkl"
        output = "/image/nlp-datasets/laura/data/mscoco/datasets/multimodal-gender-bias/lxmert/"
        process_mscoco(images_dir, cache_imgs, output)
    else:
        raise Exception(f"Option {sys.argv[1]} not supported")