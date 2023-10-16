import json
import os
import random
from collections import defaultdict

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
from models.tokenization_bert import BertTokenizer


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = pre_caption(ann["caption"], self.max_words)

        return image, caption, self.img_ids[ann["image_id"]]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, index


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if type(ann["caption"]) == list:
            caption = pre_caption(random.choice(ann["caption"]), self.max_words)
        else:
            caption = pre_caption(ann["caption"], self.max_words)

        image = Image.open(ann["image"]).convert("RGB")
        image = self.transform(image)

        return image, caption


class pretrain_neutral_dataset(Dataset):
    def __init__(
        self,
        ann_file,
        ann_neutral_file,
        transform,
        max_words=30,
        p_neutral_caption=0.15,
        batch_size=256,
        num_epochs=1,
    ):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        self.ann_neutral = []
        for f in ann_neutral_file:
            self.ann_neutral += json.load(open(f, "r"))

        self.transform = transform
        self.max_words = max_words

        self.p_neutral_caption = p_neutral_caption
        self.step = 0
        self.num_train_steps = int(len(self.ann) / batch_size) * num_epochs

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # Update probability for selecting a neutral caption
        self.step += 1
        self.p_neutral_caption = self.p_neutral_caption + (
            1 - self.p_neutral_caption
        ) * (self.step / self.num_train_steps)

        ann = self.ann[index]
        rnd = random.random()
        if rnd <= self.p_neutral_caption:
            caption = self.ann_neutral[index]["caption"]
        else:
            caption = ann["caption"]

        if type(caption) == list:
            caption = pre_caption(random.choice(caption), self.max_words)
        else:
            caption = pre_caption(caption, self.max_words)

        image = Image.open(ann["image"]).convert("RGB")
        image = self.transform(image)

        return image, caption
