# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import logging
import h5py
import numpy as np
import tensorpack.dataflow as td
import torch
import torch.distributed as dist
import msgpack_numpy
msgpack_numpy.patch()

from toolkit.data.datasets import CaptionDataset
from toolkit.utils import (
    BOXES_TRAIN_AND_VAL,
    WIDHTS_HEIGHTS,
    BU_FEATURES_FILENAME,
)
# from toolkit.utils import add_global_img_feature

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
            (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
            (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
            np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
            - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
            + 1
    )
    iw[iw < 0] = 0

    ih = (
            np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
            - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
            + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self,
        image_feat=None,
        image_cls=None,
        obj_labels=None,
        obj_confs=None,
        attr_labels=None,
        attr_confs=None,
        image_attrs=None,
        caption=None,
        is_next=None,
        lm_labels=None,
        image_loc=None,
        num_boxes=None,
        overlaps=None,
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_cls = image_cls
        self.obj_labels = obj_labels    # (label, conf)
        self.obj_confs = obj_confs
        self.attr_labels = attr_labels  # (label, conf)
        self.attr_confs = attr_confs
        self.image_attrs = image_attrs
        self.num_boxes = num_boxes
        self.overlaps = overlaps


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_cls=None,
        obj_labels=None,
        obj_confs=None,
        attr_labels=None,
        attr_confs=None,
        image_attrs=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
        masked_label=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_cls = image_cls
        self.obj_labels = obj_labels
        self.obj_confs = obj_confs
        self.attr_labels = attr_labels
        self.attr_confs = attr_confs
        self.image_attrs = image_attrs
        self.image_mask = image_mask
        self.masked_label = masked_label


class MscocoDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """
    CAPTION_LEN = 20

    def __init__(self, dataset_splits_dir, image_features_dir, dataset, split, enc_tokenizer, volta_config, max_seq_len):
        super().__init__(dataset_splits_dir, image_features_dir, dataset, split)

        self.captions_per_image = 5

        self.enc_tokenizer = enc_tokenizer
        self.max_seq_len = max_seq_len
        self.train_boxes = h5py.File(os.path.join(image_features_dir, BOXES_TRAIN_AND_VAL), "r")
        self.widths_and_heights = h5py.File(os.path.join(image_features_dir, WIDHTS_HEIGHTS), "r")
        self.conf = json.load(open(volta_config, "r"))

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image
        image = self.get_image_features(coco_id)
        num_boxes = image.shape[0]
        bounding_boxes = self.train_boxes[coco_id][()]  # (36,4)
        width_and_heights = self.widths_and_heights[coco_id][()]
        w = width_and_heights[0]
        h = width_and_heights[1]

        #calculate the IOU here.
        # overlaps = iou(bounding_boxes, bounding_boxes)
        # image, bounding_boxes, num_boxes = add_global_img_feature(image, torch.tensor(bounding_boxes), self.conf, w,
        #                                                h)  # ve se funcionou...

        caption = self.captions_text[coco_id][caption_index]
        # caption = "A [MASK] is playing tennis with [MASK] [MASK]."
        input_cap = self.enc_tokenizer(caption,
                                       padding="max_length",
                                       max_length=self.max_seq_len,
                                       add_special_tokens=True,
                                       truncation=True,
                                       return_tensors="pt")
        # indices_mask  = (input_cap.input_ids == self.enc_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        input_tokens = input_cap.input_ids[0]
        attention_mask = input_cap.attention_mask[0]

        return input_tokens, attention_mask, image, bounding_boxes, \
               coco_id, caption_index, w, h, \
               caption, num_boxes

    def __len__(self):
        return len(self.split) * self.captions_per_image


class MsCocoLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
        self,
        split,
        annotations_path,
        features_path,
        # img_id_split,
        dataset_splits_dir,
        volta_config_path,
        tokenizer,
        seq_len,
        batch_size=512,
        num_epochs=1,
        p_neutral_cap=0.15,
        num_workers=25,
        objective=0,
        num_locs=5,
        add_global_imgfeat="first"
    ):
        self.captions_per_image = 5
        # self.img_id_split = img_id_split[f"{split}_images"]
        # self.image_features = h5py.File(os.path.join(features_path, "image_features.hdf5"), "r")

        caption_path = os.path.join(annotations_path, f"caption_{split}.json")
        neutral_caption_path = os.path.join(annotations_path, f"gender-neutral/caption_{split}.json")

        self.dataset = MscocoDataset(dataset_splits_dir,
                                     features_path,
                                     "mscoco",
                                     split,
                                     tokenizer,
                                     volta_config_path,
                                     seq_len)
        self.num_dataset = len(self.dataset) #* self.captions_per_image

        self.preprocess_function = BertPreprocessBatch(
            caption_path,
            neutral_caption_path,
            p_neutral_cap,
            tokenizer,
            seq_len,
            36,
            len(self.dataset),
            batch_size,
            num_epochs,
            objective=objective,
            num_locs=num_locs,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_global_imgfeat = add_global_imgfeat
        self.num_locs = num_locs

        self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)

    def __iter__(self):
        for batch_data in self.data_loader:
            text_input_ids, text_attention_mask, image_features, \
            bounding_boxes, coco_id, caption_index, \
            image_w, image_h, caption_text, num_boxes = batch_data

            batch_size = text_input_ids.shape[0]

            input_ids = np.empty(text_input_ids.shape)
            input_masks = np.empty((batch_size, text_input_ids.shape[1],))
            segment_ids = np.empty((batch_size, text_input_ids.shape[1],))
            lm_label_ids = np.empty((batch_size, text_input_ids.shape[1],))
            is_next = np.empty((batch_size,))
            image_feats = np.empty(image_features.shape)
            image_locs = np.empty(bounding_boxes.shape)
            image_cls = []
            obj_labels = np.empty((batch_size,))
            obj_confs = np.empty((batch_size,))
            attr_labels = np.empty((batch_size,))
            attr_confs = np.empty((batch_size,))
            image_attrs = []
            image_labels = np.empty((batch_size, num_boxes[0],))
            image_masks = np.empty((batch_size, num_boxes[0],))
            image_ids = np.empty((batch_size,), dtype="S10")
            masked_label = np.empty((batch_size, 1, num_boxes[0]))

            for b in range(batch_size):
                batch_data_function = [image_features[b],
                                       None, None, None, None, None, None,
                                       bounding_boxes[b],
                                       num_boxes[b],
                                       image_w[b],
                                       image_h[b],
                                       coco_id[b],
                                       caption_text[b]]
                # print(">>>", b, coco_id[b], caption_text[b])
                # batch_data_function = [image_features,
                #                        [], [], [], [], [], [],
                #                        bounding_boxes,
                #                        num_boxes,
                #                        image_w,
                #                        image_h,
                #                        coco_id,
                #                        caption_text]

                input_id, input_mask, segment_id, lm_label_id, is_next_, image_feat, image_loc, \
                    image_cls_, obj_label, obj_conf, attr_label, attr_conf, image_attr, \
                image_label, image_mask, masked_label_, image_id = \
                    self.preprocess_function(batch_data_function)

                input_ids[b] = input_id
                input_masks[b] = input_mask
                segment_ids[b] = segment_id
                lm_label_ids[b] = lm_label_id
                is_next[b] = is_next_
                image_feats[b] = image_feat
                image_locs[b] = image_loc
                image_cls.append(image_cls_)
                obj_labels[b] = obj_label
                obj_confs[b] = obj_conf
                attr_labels[b] = attr_label
                attr_confs[b] = attr_conf
                image_attrs.append(image_attr)
                image_labels[b] = image_label
                image_masks[b] = image_mask
                image_ids[b] = image_id
                masked_label[b] = masked_label_

            if self.add_global_imgfeat == "first":
                sum_count = np.sum(masked_label == 0, axis=2, keepdims=True)
                sum_count[sum_count == 0] = 1
                g_image_feat = np.sum(image_feats, axis=2) / sum_count
                # image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
                image_feats = np.concatenate([g_image_feat.transpose((0,2,1)), image_feats], axis=2)
                image_feats = np.array(image_feats, dtype=np.float32)  # (batch_size, 36, 2049)

                g_loc = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_locs = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_locs], axis=1) # (37,4)
                image_locs = np.array(image_locs, dtype=np.float32)

                g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_masks = np.concatenate([g_image_mask, image_masks], axis=1)
            # elif self.add_global_imgfeat == "last":
            #     sum_count = np.sum(np.where(masked_label == 0), axis=1, keepdims=True)
            #     sum_count[sum_count == 0] = 1
            #     g_image_feat = np.sum(image_feats, axis=1) / sum_count
            #     image_feats = np.concatenate([image_feats, g_image_feat.T], axis=1)
            #     image_feats = np.array(image_feats, dtype=np.float32)
            #
            #     g_loc = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
            #     g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
            #     image_locs = np.concatenate([image_locs, g_image_loc], axis=1)
            #
            #     image_locs = np.array(image_locs, dtype=np.float32)
            #     g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            #     image_masks = np.concatenate([np.expand_dims(image_masks, axis=1), g_image_mask], axis=1)

            batch = (
                torch.tensor(input_ids, dtype=torch.long),  #0
                torch.tensor(input_masks, dtype=torch.long),  #1
                torch.tensor(segment_ids, dtype=torch.long),  #2
                torch.tensor(lm_label_ids, dtype=torch.long),  #3
                torch.tensor(is_next, dtype=torch.long),  #4
                torch.tensor(image_feats, dtype=torch.float32),  #5
                torch.tensor(image_locs, dtype=torch.float32),  #6
                torch.tensor(image_cls, dtype=torch.float32),  #7
                torch.tensor(obj_labels, dtype=torch.float32),  #8
                torch.tensor(obj_confs, dtype=torch.float32),  #9
                torch.tensor(attr_labels, dtype=torch.float32),  #10
                torch.tensor(attr_confs, dtype=torch.float32),  #11
                torch.tensor(image_attrs, dtype=torch.float32),  #12
                torch.tensor(image_labels, dtype=torch.long),  #13
                torch.tensor(image_masks, dtype=torch.long),  #14
            )

            # yield tuple([torch.tensor(data) for data in batch] + [image_ids])
            yield tuple([data for data in batch] + [image_ids])

    def __len__(self):
        return len(self.dataset)


class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
            neutral_caption_path,
            p_neutral_caption,
            tokenizer,
            seq_len,
            region_len,
            data_size,
            batch_size,
            num_epochs,
            visualization=False,
            objective=0,
            num_locs=5
    ):

        self.max_seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, "r")).values())
        neutral_captions_dict_ = json.load(open(neutral_caption_path, "r"))
        self.neutral_captions_dict = {}
        # clean keys... sometimes is '0978' -> '978'
        for k, v in neutral_captions_dict_.items():
            self.neutral_captions_dict[str(int(k))] = v
        self.p_neutral_caption = p_neutral_caption
        self.visualization = visualization
        self.objective = objective
        self.num_locs = num_locs

        self.step = 0
        self.num_train_steps = int(data_size/batch_size) * num_epochs

    def __call__(self, data):
        image_feature_wp, image_cls_wp, obj_labels, obj_confs, attr_labels, attr_confs, attr_scores, \
            image_location_wp, num_boxes, image_w, image_h, image_id, caption = data

        image_feature_wp = image_feature_wp.squeeze(dim=0)
        image_location_wp = image_location_wp.squeeze(dim=0)

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_cls = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_attrs = np.zeros((self.region_len, 401), dtype=np.float32)
        image_location = np.zeros((self.region_len, self.num_locs), dtype=np.float32)

        # calculate the IOU here.
        overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        # image_cls[:num_boxes] = image_cls_wp
        # image_attrs[:num_boxes] = attr_scores
        image_location[:num_boxes, :4] = image_location_wp
        # obj_labels = obj_labels[:num_boxes]
        # obj_confs = obj_confs[:num_boxes]
        # attr_labels = attr_labels[:num_boxes]
        # attr_confs = attr_confs[:num_boxes]

        if self.num_locs >= 5:
            image_location[:, -1] = (
                (image_location[:, 3] - image_location[:, 1])
                * (image_location[:, 2] - image_location[:, 0])
                / (float(image_w) * float(image_h))
            )

        # The following is not needed for the MSCOCO vectors
        # loaded since they are already normalized
        # # Normalize the box locations (to 0 ~ 1)
        # image_location[:, 0] = image_location[:, 0] / float(image_w)
        # image_location[:, 1] = image_location[:, 1] / float(image_h)
        # image_location[:, 2] = image_location[:, 2] / float(image_w)
        # image_location[:, 3] = image_location[:, 3] / float(image_h)
        #
        # if self.num_locs > 5:
        #     image_location[:, 4] = image_location[:, 2] - image_location[:, 0]
        #     image_location[:, 5] = image_location[:, 3] - image_location[:, 1]

        caption, label = self.random_cap(caption, image_id)
        tokens_caption = self.tokenizer.encode(caption,
                                               padding="max_length",
                                               max_length=self.max_seq_len,
                                               truncation=True,
                                               add_special_tokens=False)
        # Update probability for selecting a neutral caption
        self.step += 1
        self.p_neutral_caption = self.p_neutral_caption + \
                                 (1 - self.p_neutral_caption) * \
                                 (self.step / self.num_train_steps)

        cur_example = InputExample(
            image_feat=image_feature,
            image_cls=image_cls,
            obj_labels=obj_labels,
            obj_confs=obj_confs,
            attr_labels=attr_labels,
            attr_confs=attr_confs,
            image_attrs=image_attrs,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.max_seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_cls,
            cur_features.obj_labels,
            cur_features.obj_confs,
            cur_features.attr_labels,
            cur_features.attr_confs,
            cur_features.image_attrs,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.masked_label,
            image_id,
        )
        return cur_tensors

    def random_cap(self, caption, image_id):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0

        if self.objective != 2 and random.random() > 0.5:
            caption = self.get_random_caption()
            label = 1
        else:
            if random.random() <= self.p_neutral_caption:
                caption = self.neutral_captions_dict[image_id][
                    random.randint(0, len(self.neutral_captions_dict[image_id])-1)
                ]
            label = 0
            if isinstance(caption,tuple):
                caption = caption[0]

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, len(self.captions) - 1)
        if random.random() <= self.p_neutral_caption:
            caption = list(self.neutral_captions_dict.values())[rand_doc_idx]
        else:
            caption = self.captions[rand_doc_idx]
        # Select 1 caption out of the 5 captions/image
        caption = caption[random.randint(0, len(caption)-1)]

        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_cls = example.image_cls
        num_boxes = int(example.num_boxes)
        overlaps = example.overlaps

        self._truncate_seq_pair(tokens, max_seq_length - 2)

        tokens, tokens_label = self.random_word(tokens, tokenizer)
        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, num_boxes, overlaps
        )

        # concatenate lm labels and account for CLS and SEP: [CLS] tokens [SEP]
        lm_label_ids = [-1] + tokens_label + [-1]
        tokens = tokenizer.build_inputs_with_special_tokens(tokens)
        segment_ids = [0] * len(tokens)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        image_mask = [1] * num_boxes
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_cls=image_cls,
            obj_labels=example.obj_labels,
            obj_confs=example.obj_confs,
            attr_labels=example.attr_labels,
            attr_confs=example.attr_confs,
            image_attrs=example.image_attrs,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            masked_label=masked_label,
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break
            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, overlaps):
        """
        """
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 90% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label, masked_label
