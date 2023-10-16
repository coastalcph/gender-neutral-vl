import os
import re
import json
import tempfile
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from functools import partial
from typing import List
import cv2
from PIL import Image
import numpy as np
from cog import BasePredictor, Path, Input
from collections import defaultdict

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision import transforms

from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertForMaskedLM
from models.tokenization_bert import BertTokenizer


class Predictor(BasePredictor):
    def setup(self):
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert/bert-base-uncased")

        bert_config_path = "configs/config_bert.json"
        self.model = VL_Transformer_ITM(
            text_encoder="bert/bert-base-uncased", config_bert=bert_config_path
        )

        checkpoint = torch.load("refcoco.pth", map_location="cpu")
        msg = self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        self.block_num = 8
        self.model.text_encoder.base_model.base_model.encoder.layer[
            self.block_num
        ].crossattention.self.save_attention = True

        self.model.cuda()

    def predict(
        self,
        image: Path = Input(description="Input image."),
        caption: str = Input(
            description="Caption for the image. Grad-CAM visualization will be generated "
            "for each word in the cation."
        ),
    ) -> Path:
        image_pil = Image.open(str(image)).convert("RGB")
        img = self.transform(image_pil).unsqueeze(0)

        text = pre_caption(caption)
        text_input = self.tokenizer(text, return_tensors="pt")

        img = img.cuda()
        text_input = text_input.to(img.device)

        # Compute GradCAM
        output = self.model(img, text_input)
        loss = output[:, 1].sum()

        self.model.zero_grad()
        loss.backward()

        with torch.no_grad():
            mask = text_input.attention_mask.view(
                text_input.attention_mask.size(0), 1, -1, 1, 1
            )

            grads = self.model.text_encoder.base_model.base_model.encoder.layer[
                self.block_num
            ].crossattention.self.get_attn_gradients()
            cams = self.model.text_encoder.base_model.base_model.encoder.layer[
                self.block_num
            ].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(img.size(0), 12, -1, 24, 24) * mask
            grads = (
                grads[:, :, :, 1:].clamp(0).reshape(img.size(0), 12, -1, 24, 24) * mask
            )

            gradcam = cams * grads
            gradcam = gradcam[0].mean(0).cpu().detach()

        num_image = len(text_input.input_ids[0])
        fig, ax = plt.subplots(num_image, 1, figsize=(20, 8 * num_image))

        rgb_image = cv2.imread(str(image))[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        ax[0].imshow(rgb_image)
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel("Image")

        for i, token_id in enumerate(text_input.input_ids[0][1:]):
            word = self.tokenizer.decode([token_id])
            gradcam_image = getAttMap(rgb_image, gradcam[i + 1])
            ax[i + 1].imshow(gradcam_image)
            ax[i + 1].set_yticks([])
            ax[i + 1].set_xticks([])
            ax[i + 1].set_xlabel(word)

        out_path = Path(tempfile.mkdtemp()) / "output.png"
        fig.savefig(str(out_path))
        return out_path


class VL_Transformer_ITM(nn.Module):
    def __init__(self, text_encoder=None, config_bert=""):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False
        )

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output


class PredictorMLM:
    def __init__(self, args):
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.model = VL_Transformer_MLM(
            text_encoder="bert-base-uncased",
            config_bert=args.config,
            tokenizer=self.tokenizer,
        )

        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        msg = self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.model.cuda()

    def predict(
        self,
        image: Path = Input(description="Input image."),
        caption: str = Input(
            description="Caption for the image. Grad-CAM visualization will be generated "
            "for each word in the cation."
        ),
    ) -> List[str]:
        image_pil = Image.open(str(image)).convert("RGB")
        img = self.transform(image_pil).unsqueeze(0)

        text = pre_caption(caption)
        text = text.replace("[mask]", "[MASK]")
        text_input = self.tokenizer(text, return_tensors="pt")

        img = img.cuda()
        text_input = text_input.to(img.device)

        output = self.model(img, text_input)

        is_masked_token = torch.eq(
            text_input.input_ids, self.tokenizer.mask_token_id
        ).to(torch.long)
        current_masks_tokens = []
        _, indices = torch.topk(
            is_masked_token.squeeze(),
            k=torch.sum(is_masked_token.squeeze()).item(),
            sorted=True,
        )
        for mask_position in indices.tolist():
            id_of_prediction = torch.argmax(output[:, mask_position, :])
            token_of_prediction = self.tokenizer.convert_ids_to_tokens(
                [id_of_prediction]
            )
            current_masks_tokens.extend(token_of_prediction)

        return current_masks_tokens


class VL_Transformer_MLM(nn.Module):
    def __init__(self, text_encoder=None, config_bert="", tokenizer=None):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)
        self.tokenizer = tokenizer
        self.visual_encoder = VisionTransformer(
            img_size=384,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.text_encoder = BertForMaskedLM.from_pretrained(
            text_encoder, config=bert_config
        )

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            return_logits=True,
        )

        return output


def pre_caption(caption, max_words=30):
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])
    return caption


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap**0.7).reshape(attMap.shape + (1,)) * img
            + (attMap**0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap


def _get_img_id(dataset, fname, split):
    if dataset == "mscoco":
        if split == "test":
            fname = fname[len("COCO_test2014_") :]
        else:
            fname = fname[len("COCO_val2014_") :]
    return str(int(fname.split(".")[0]))


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script for evaluating models at inference time"
    )
    parser.add_argument("--output-path", type=str, help="Folder where to store outputs")
    parser.add_argument("--timestamp", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mscoco", "cc"],
        help="Name of the dataset to be used",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "valid", "test"]
    )
    parser.add_argument(
        "--config", type=str, help="Directory with the encoder configuration files"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="Directory with volta checkpoints"
    )
    parser.add_argument("--ablation", action="store_true")

    parser.add_argument(
        "--features_path",
        type=str,
        help="Folder where the preprocessed image data is located",
    )
    parser.add_argument(
        "--dataset-splits-dir",
        type=str,
        help="Pickled file containing the dataset splits",
    )
    parser.add_argument(
        "--annotations_path",
        default="datasets/conceptual_caption/annotations",
        type=str,
        help="The corpus annotations directory.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    return parser


if __name__ == "__main__":
    MASKED_CAPTIONS_FILENAME = "masked_captions_SPLIT.json"
    DATASET_MASKED_SPLITS_FILENAME = "dataset_masked_splits.json"
    TRAIN_SPLIT = "train_images"
    VALID_SPLIT = "val_images"
    TEST_SPLIT = "test_images"

    parser = create_parser()
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logging.info(args)

    # Output directories
    if args.timestamp:
        timestamp = "_{:%d%h_%H%M}".format(datetime.today())
        save_path = args.output_path + timestamp
    else:
        save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)

    predictor = PredictorMLM(args)
    predictions_tokens = defaultdict(dict)

    # load captions with text for retrieval
    with open(
        os.path.join(
            args.dataset_splits_dir,
            MASKED_CAPTIONS_FILENAME.replace("SPLIT", args.split),
        )
    ) as f:
        captions_text_ = json.load(f)
        captions_text = {}
        # clean keys... sometimes is '0978' -> '978'
        for k, v in captions_text_.items():
            captions_text[str(int(k))] = v

    # Load images
    a = 0
    for root, _, imgfiles in os.walk(args.features_path):
        # Map image to caption
        for fname in imgfiles:
            image_ = os.path.join(root, fname)
            img_id = _get_img_id(args.dataset, fname, args.split)

            if img_id in captions_text:
                captions = captions_text[img_id]
                for caption_idx, cap in enumerate(captions):
                    current_masks_tokens = predictor.predict(image_, cap)
                    predictions_tokens[img_id][caption_idx] = current_masks_tokens

    # Saving predictions_tokens
    predictions_tokens_file = os.path.join(
        save_path, args.split + "predictions_tokens.json"
    )
    with open(predictions_tokens_file, "w+") as f:
        json.dump(predictions_tokens, f, indent=2)
    print(
        f"File {predictions_tokens_file} saved with {len(predictions_tokens)} entries written."
    )
