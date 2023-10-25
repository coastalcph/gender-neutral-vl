"""
python h5_to_lxmert_splits.py
 --h5_tr /projects/nlp/data/emanuele/data/mscoco/features/train2014_boxes36.h5
 --h5_val /projects/nlp/data/emanuele/data/mscoco/features/val2014_boxes36.h5
"""
import h5py
import argparse
import json
from tqdm import tqdm
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


def write_h5(list_of_items, output_fname):
    with h5py.File(output_fname, 'w') as f:
        for datum in list_of_items:

            img_id = datum['img_id']
            num_boxes = 36
            for i in range(5):
                grp = f.create_group(img_id+'_'+str(i))
                grp['attr_conf'] = datum['attr_conf']
                grp['attr_id'] = datum['attr_id']
                grp['boxes'] = datum['boxes']
                grp['features'] = datum['features'].reshape(num_boxes, 2048)
                grp['img_h'] = datum['img_h']
                grp['img_w'] = datum['img_w']
                grp['obj_conf'] = datum['obj_conf']
                grp['obj_id'] = datum['obj_id']
                grp['caption_id'] = i


def main(corpus_path_tr, corpus_path_val, splits):
    splits = json.load(open(splits, "r"))
    imgs_to_train = []
    imgs_to_val = []
    imgs_to_test = []

    with h5py.File(corpus_path_tr, 'r') as f:
        for i, img_id in enumerate(tqdm(f.keys())):
            if i == 0:
                keys = list(f[img_id].keys())

            img_id_in_lxmert = str(int(img_id.split("2014_")[-1]))
            item = {}
            for k in keys:
                item[k] = f[f'{img_id}/{k}'][()]
            item['img_id'] = img_id_in_lxmert

            if img_id_in_lxmert in splits["train_images"]:
                imgs_to_train.append(item)
            elif img_id_in_lxmert in splits["val_images"]:
                imgs_to_val.append(item)
            else:
                imgs_to_test.append(item)

    with h5py.File(corpus_path_val, 'r') as f:
        for i, img_id in enumerate(tqdm(f.keys())):
            if i == 0:
                keys = list(f[img_id].keys())

            img_id_in_lxmert = str(int(img_id.split("2014_")[-1]))

            item = {}
            for k in keys:
                item[k] = f[f'{img_id}/{k}'][()]
            item['img_id'] = img_id_in_lxmert

            if img_id_in_lxmert in splits["train_images"]:
                imgs_to_train.append(item)
            elif img_id_in_lxmert in splits["val_images"]:
                imgs_to_val.append(item)
            else:
                imgs_to_test.append(item)

    return imgs_to_train, imgs_to_val, imgs_to_test


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, shuffle=False, num_imgs=None):
        self.corpus_path = corpus_path
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

                item = {}
                for k in keys:
                    item[k] = f[f'{img_id}/{k}'][()]
                item['img_id'] = img_id

                yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_tr', type=str)
    parser.add_argument('--h5_val', type=str)
    parser.add_argument('--splits', type=str,
                        default='/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/annotations/gender-mappings/dataset_masked_splits.json')
    args = parser.parse_args()

    source_train = args.h5_tr
    source_val = args.h5_val
    h5_train, h5_val, h5_test = main(source_train, source_val, args.splits)

    print(f"Saving {len(h5_train)} instances in train")
    write_h5(h5_train,
             "/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/h5_features/train2014_boxes36_lxmert.h5")
    print(f"Saving {len(h5_val)} instances in validation")
    write_h5(h5_val,
             "/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/h5_features/val2014_boxes36_lxmert.h5")
    print(f"Saving {len(h5_test)} instances in test")
    write_h5(h5_test,
             "/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/h5_features/test2014_boxes36_lxmert.h5")

    # TO LMDB
    print(">>> Now to LMDB")
    ds = PretrainData("/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/h5_features/train2014_boxes36_lxmert.h5",
                      )
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, "/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/train_feat.lmdb")

    ds = PretrainData("/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/h5_features/val2014_boxes36_lxmert.h5",
                      )
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, "/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/val_feat.lmdb")

    # ds = PretrainData("/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/h5_features/test2014_boxes36_lxmert.h5",
    #                   )
    # ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    # LMDBSerializer.save(ds1, "/projects/nlp_mgr-AUDIT/data/dataset/sxk199/mscoco/resnet101_faster_rcnn_genome_imgfeats/test_feat.lmdb")