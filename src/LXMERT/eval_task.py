# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Source: https://github.com/e-bug/volta/blob/main/eval_task.py

import os
import sys
import json
import yaml
import random
import logging
from io import open
import numpy as np
from tqdm import tqdm
from datetime import datetime
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.distributed as dist
from train_utils import parse_task_args
from volta.volta.config import BertConfig, M3PConfig
from volta.volta.encoders import BertForVLTasks, M3PForVLTasks
from volta.volta.train_utils import tbLogger
from volta.volta.task_utils import LoadDatasetEval, LoadLoss, EvaluatingModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    args = parse_task_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
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
    task_name = task_cfg[task]["name"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    save_path = os.path.join(args.output_dir, args.save_name)
    if default_gpu and not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.info(f"> Output directory: {save_path}")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(
        args, config, task_cfg, args.task
    )

    # Logging
    tb_logger = tbLogger(
        args.logdir,
        save_path,
        [task_name],
        [task],
        task2num_iters,
        1,
        save_logger=False,
        txt_name="eval.txt",
    )

    # Model
    if "roberta" in config.bert_model:
        config.model = "roberta"
    if args.is_m3p:
        model = M3PForVLTasks.from_pretrained(
            args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task]
        )
    else:
        model = BertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task]
        )

    # Optimization details
    criterion = LoadLoss(args, task_cfg, args.task)

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

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters[task])
        print("  Batch size: ", batch_size)

    # Evaluate
    model.eval()
    results = []
    others = []
    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
        loss, score, batch_size, results, others = EvaluatingModel(
            config,
            task_cfg,
            device,
            task,
            batch,
            model,
            dl_val,
            criterion,
            results,
            others,
        )

        tb_logger.step_val(0, float(loss), float(score), task, batch_size, "val")
        sys.stdout.write("%d/%d\r" % (i, len(dl_val)))
        sys.stdout.flush()

    # save the result or evaluate the result.
    ave_score, ave_loss = tb_logger.showLossVal(task)
    if task == "TASK12":
        from collections import defaultdict

        sent2corrects = defaultdict(list)
        for e in results:
            sent2corrects[e["sentence"]].append(e["prediction"] == e["label"])
        s = 0
        for l in sent2corrects.values():
            s += sum(l) == len(l)
        consistency = float(s) / len(sent2corrects) * 100
        logger.info(f"Consistency: {consistency}")

    json_path = os.path.join(save_path, task_cfg[task]["val_split"])
    json.dump(results, open(json_path + "_result.json", "w"))
    json.dump(others, open(json_path + "_others.json", "w"))


if __name__ == "__main__":
    main()
