"""
python h5_to_td-lmdb_lxmert.py
 --h5_tr /projects/nlp/data/emanuele/data/mscoco/features/train2014_boxes36.h5
 --h5_val /projects/nlp/data/emanuele/data/mscoco/features/val2014_boxes36.h5
 --lmdb_tr /projects/nlp/data/data/multimodal-gender-bias/data/volta/mscoco/features/from_train2014/train2014_boxes36_lxmert.lmdb
 --lmdb_val /projects/nlp/data/data/multimodal-gender-bias/data/volta/mscoco/features/from_train2014/val2014_boxes36_lxmert.lmdb
"""
import h5py
import argparse
import json
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, splits, split, shuffle=False, num_imgs=None):
        self.corpus_path = corpus_path
        self.splits = json.load(open(splits, "r"))[split]
        self.shuffle = shuffle

        if num_imgs is None:
            with h5py.File(corpus_path, 'r') as f:
                num_imgs = len(f)
        self.num_imgs = num_imgs

    def __len__(self):
        return self.num_imgs

    def __iter__(self):
        with h5py.File(self.corpus_path, 'r') as f:
            for i, img_id in enumerate(f.keys()):
                if i == 0:
                    keys = list(f[img_id].keys())

                img_id_in_lxmert = str(int(img_id.split("2014_")[-1]))
                if img_id_in_lxmert in self.splits:
                    item = {}
                    for k in keys:
                        item[k] = f[f'{img_id}/{k}'][()]
                    item['img_id'] = img_id_in_lxmert
                    yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_tr', type=str)
    parser.add_argument('--h5_val', type=str)
    parser.add_argument('--lmdb_tr', type=str)
    parser.add_argument('--lmdb_val', type=str)
    parser.add_argument('--num_imgs', type=int, default=None)
    parser.add_argument('--splits', type=str,
                        default='/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/annotations/gender-mappings/dataset_masked_splits.json')
    args = parser.parse_args()

    target_fname_tr = args.lmdb_tr
    target_fname_val = args.lmdb_val
    img_distribution = args.splits

    #############
    ### TRAIN ###
    #############
    # Train in train
    ds_tr = PretrainData(args.h5_tr, img_distribution, "train_images", num_imgs=args.num_imgs)
    ds_tr = PrefetchDataZMQ(ds_tr, nr_proc=1)
    LMDBSerializer.save(ds_tr, target_fname_tr)
    # LXMERT validation in H5 train
    ds_tr = PretrainData(args.h5_tr, img_distribution, "val_images", num_imgs=args.num_imgs)
    ds_tr = PrefetchDataZMQ(ds_tr, nr_proc=1)
    LMDBSerializer.save(ds_tr, target_fname_val.replace(".", "_2."))

    #############
    ### VALID ###
    #############
    # # Validation in validation
    # ds_val = PretrainData(args.h5_val, img_distribution, "val_images", num_imgs=args.num_imgs)
    # ds_val = PrefetchDataZMQ(ds_val, nr_proc=1)
    # LMDBSerializer.save(ds_val, target_fname_val)
    # # LXMERT train in H5 val
    # ds_val = PretrainData(args.h5_val, img_distribution, "train_images", num_imgs=args.num_imgs)
    # ds_val = PrefetchDataZMQ(ds_val, nr_proc=1)
    # LMDBSerializer.save(ds_val, target_fname_tr.replace(".", "_2."))



