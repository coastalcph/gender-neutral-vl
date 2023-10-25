# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import yaml
import random
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from io import open
from tqdm import tqdm
from easydict import EasyDict as edict
from train_utils import parse_retrieval_task

from volta.volta.config import BertConfig, M3PConfig
from volta.volta.encoders import BertForVLPreTraining, BertForVLTasks, M3PForVLTasks
from volta.volta.task_utils import LoadDatasetEval

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    args = parse_retrieval_task()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        # print("LOCAL_RANK", local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(
        f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}"
    )

    # Load config
    if args.is_m3p:
        config = M3PConfig.from_json_file(args.config_file)
    else:
        config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id

    # Output dirs
    save_path = os.path.join(args.output_dir, args.save_name)
    # if default_gpu and not os.path.exists(savePath):
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"> Output directory: {save_path}")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(
        args, config, task_cfg, args.task
    )

    # Model
    if args.zero_shot:
        config.visual_target_weights = {}  # [0, 0, 0, 0, 0, 0, 0]
        model = BertForVLPreTraining.from_pretrained(
            args.from_pretrained, config=config
        )
    else:
        if args.is_m3p:
            model = M3PForVLTasks.from_pretrained(
                args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task]
            )
        else:
            model = BertForVLTasks.from_pretrained(
                args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task]
            )

    # Move to GPU(s)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model)
        raise ValueError("Please run with a single GPU")

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters)
        print("  Batch size: ", batch_size)

    # Evaluate
    model.eval()
    results = []
    others = []
    score_matrix = np.zeros((dset_val.num_entries, dset_val.num_images))
    target_matrix = np.zeros((dset_val.num_entries, dset_val.num_images))
    rank_vector = np.ones(dset_val.num_entries) * dset_val.num_images
    count = 0
    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        (
            features,
            spatials,
            image_mask,
            question,
            input_mask,
            segment_ids,
            target,
            caption_idx,
            image_idx,
        ) = batch

        features = features.squeeze(0)
        spatials = spatials.squeeze(0)
        image_mask = image_mask.squeeze(0)
        question = question.repeat(features.size(0), 1)
        segment_ids = segment_ids.repeat(features.size(0), 1)
        input_mask = input_mask.repeat(features.size(0), 1)

        target = target.view(-1).float().cpu().numpy()
        with torch.no_grad():
            if args.zero_shot:
                _, _, vil_logit, _, _ = model(
                    question, features, spatials, segment_ids, input_mask, image_mask
                )

                score_matrix[
                    caption_idx,
                    image_idx
                    * dset_val.max_num_images : image_idx
                    * dset_val.max_num_images
                    + len(target),
                ] = (
                    torch.softmax(vil_logit, dim=1)[:, 0].view(-1).cpu().numpy()
                )
                target_matrix[
                    caption_idx,
                    image_idx
                    * dset_val.max_num_images : image_idx
                    * dset_val.max_num_images
                    + len(target),
                ] = target

            else:
                vil_logit, _, _, _ = model(
                    question,
                    features,
                    spatials,
                    task,
                    segment_ids,
                    input_mask,
                    image_mask,
                )

                score_matrix[
                    caption_idx,
                    image_idx
                    * dset_val.max_num_images : image_idx
                    * dset_val.max_num_images
                    + len(target),
                ] = (
                    vil_logit.view(-1).cpu().numpy()
                )
                target_matrix[
                    caption_idx,
                    image_idx
                    * dset_val.max_num_images : image_idx
                    * dset_val.max_num_images
                    + len(target),
                ] = target

            if image_idx.item() == (args.num_subiters - 1):
                rank = np.where(
                    (
                        np.argsort(-score_matrix[caption_idx])
                        == np.where(target_matrix[caption_idx] == 1)[0][0]
                    )
                    == 1
                )[0][0]
                rank_vector[caption_idx] = rank

                cur_rank_vector = rank_vector[: caption_idx + 1]
                r1 = 100.0 * np.sum(cur_rank_vector < 1) / len(cur_rank_vector)
                r5 = 100.0 * np.sum(cur_rank_vector < 5) / len(cur_rank_vector)
                r10 = 100.0 * np.sum(cur_rank_vector < 10) / len(cur_rank_vector)

                medr = np.floor(np.median(cur_rank_vector) + 1)
                meanr = np.mean(cur_rank_vector) + 1
                print(
                    "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
                    % (count, r1, r5, r10, medr, meanr)
                )
                results.append(
                    {
                        "image_idx": image_idx.cpu().item(),
                        "caption_idx": caption_idx.cpu().item(),
                        "answer": np.argsort(-score_matrix[caption_idx]).tolist(),
                        "target": np.where(target_matrix[caption_idx] == 1)[0].tolist()[
                            0
                        ],
                    }
                )

        count += 1

    r1 = 100.0 * np.sum(rank_vector < 1) / len(rank_vector)
    r5 = 100.0 * np.sum(rank_vector < 5) / len(rank_vector)
    r10 = 100.0 * np.sum(rank_vector < 10) / len(rank_vector)

    medr = np.floor(np.median(rank_vector) + 1)
    meanr = np.mean(rank_vector) + 1

    print("************************************************")
    print("****************Image Retrieval*****************")
    print("************************************************")
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************")

    try:
        json.dump(results, open(save_path + "result.json", "w"), indent=4)
        json.dump(others, open(save_path + "others.json", "w"), indent=4)
    except TypeError:
        pickle.dump(results, open(save_path + "result.pkl", "wb"))

    # Text Retrieval
    results = []
    rank_vector = np.zeros(dset_val.num_images)
    for image_idx in range(dset_val.num_images):
        ranks = []
        tgt_captions = np.where(target_matrix[:, image_idx] == 1)[0]
        sorted_scores = np.argsort(-score_matrix[:, image_idx])
        for tgt_caption in tgt_captions:
            ranks.append(np.where((sorted_scores == tgt_caption) == 1)[0][0])
        rank_vector[image_idx] = min(ranks)
        results.append(
            {
                "image_idx": image_idx,
                "answer": sorted_scores.tolist(),
                "target": tgt_captions.tolist(),
            }
        )

    r1 = 100.0 * np.sum(rank_vector < 1) / len(rank_vector)
    r5 = 100.0 * np.sum(rank_vector < 5) / len(rank_vector)
    r10 = 100.0 * np.sum(rank_vector < 10) / len(rank_vector)

    medr = np.floor(np.median(rank_vector) + 1)
    meanr = np.mean(rank_vector) + 1

    print("************************************************")
    print("****************Text Retrieval******************")
    print("************************************************")
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************")

    pickle.dump(results, open(save_path + "TR_result.pkl", "wb"))


if __name__ == "__main__":
    main()
