import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os, glob


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform):
        self.ann_pretrain = []
        for f in ann_file:
            print("loading " + f)
            ann = json.load(open(f, "r"))
            self.ann_pretrain += ann

        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path, "*.json"))

            print("loading " + self.laion_files[0])
            with open(self.laion_files[0], "r") as f:
                self.ann_laion = json.load(f)

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain

        self.transform = transform

    def reload_laion(self, epoch):
        n = epoch % len(self.laion_files)
        print("loading " + self.laion_files[n])
        with open(self.laion_files[n], "r") as f:
            self.ann_laion = json.load(f)

        self.annotation = self.ann_pretrain + self.ann_laion

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image = Image.open(ann["image"]).convert("RGB")
        image = self.transform(image)
        caption = pre_caption(ann["caption"], 30)

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
        self.ann_pretrain = []
        for f in ann_file:
            print("loading " + f)
            ann = json.load(open(f, "r"))
            self.ann_pretrain += ann

        self.ann_neutral = []
        for f in ann_neutral_file:
            self.ann_neutral += json.load(open(f, "r"))

        self.transform = transform
        self.max_words = max_words

        self.p_neutral_caption = p_neutral_caption
        self.step = 0
        self.num_train_steps = int(len(self.ann_pretrain) / batch_size) * num_epochs

    def __len__(self):
        return len(self.ann_pretrain)

    def __getitem__(self, index):
        # Update probability for selecting a neutral caption
        self.step += 1
        self.p_neutral_caption = self.p_neutral_caption + (
            1 - self.p_neutral_caption
        ) * (self.step / self.num_train_steps)

        ann = self.ann_pretrain[index]
        rnd = random.random()
        if rnd <= self.p_neutral_caption:
            caption = self.ann_neutral[index]["caption"]
        else:
            caption = ann["caption"]

        image = Image.open(ann["image"]).convert("RGB")
        image = self.transform(image)
        caption = pre_caption(ann["caption"], self.max_words)

        return image, caption
