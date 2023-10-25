# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Script heavily based on https://github.com/e-bug/volta/blob/main/train_concap.py

import os
import sys
import json
import random
import logging
import torch
import torch.distributed as dist
import numpy as np

from io import open
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from pytorch_transformers.optimization import (
    AdamW,
    WarmupLinearSchedule,
)
from train_utils import parse_args, init_wandb, log_wandb
from volta.volta.config import BertConfig
from volta.volta.encoders import BertForVLPreTraining
from volta.volta.datasets import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    CocoLoaderTrain,
    CocoLoaderVal,
)
from volta.volta.train_utils import (
    freeze_layers,
    tbLogger,
    summary_parameters,
    save,
    resume,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(args):
    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(
            backend="nccl"
        )  # Init distributed backend for sychronizing nodes/GPUs
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
    config = BertConfig.from_json_file(args.config_file)

    # Output dirs
    save_path = args.output_dir
    logdir = args.logdir
    if args.timestamp:
        timestamp = "_{:%d%h_%H%M}".format(datetime.today())
        save_path += timestamp
        logdir += timestamp
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {save_path}")
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    cache = 5000
    args.train_batch_size = args.train_batch_size // args.grad_acc_steps
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        args.train_batch_size = args.train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained(
        config.bert_model, do_lower_case=config.do_lower_case
    )
    if args.task == "Pretrain_CC3M_neutral":
        train_dataset = ConceptCapLoaderTrain(
            args.annotations_path,
            args.features_path,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            num_epochs=args.num_train_epochs,
            p_neutral_cap=args.p_neutral_cap,
            num_workers=args.num_workers,
            local_rank=args.local_rank,
            objective=args.objective,
            cache=cache,
            add_global_imgfeat=config.add_global_imgfeat,
            num_locs=config.num_locs,
        )
        valid_dataset = ConceptCapLoaderVal(
            args.annotations_path,
            args.features_path,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            num_epochs=args.num_train_epochs,
            p_neutral_cap=args.p_neutral_cap,
            num_workers=2,
            objective=args.objective,
            add_global_imgfeat=config.add_global_imgfeat,
            num_locs=config.num_locs,
        )
        task_names = ["Conceptual_Caption"]
    elif args.task == "Pretrain_COCO_neutral":
        train_dataset = CocoLoaderTrain(
            args.annotations_path,
            args.features_path,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            num_epochs=args.num_train_epochs,
            p_neutral_cap=args.p_neutral_cap,
            num_workers=args.num_workers,
            local_rank=args.local_rank,
            objective=args.objective,
            cache=cache,
            add_global_imgfeat=config.add_global_imgfeat,
            num_locs=config.num_locs,
        )
        valid_dataset = CocoLoaderVal(
            args.annotations_path,
            args.features_path,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            num_epochs=args.num_train_epochs,
            p_neutral_cap=args.p_neutral_cap,
            num_workers=2,
            objective=args.objective,
            add_global_imgfeat=config.add_global_imgfeat,
            num_locs=config.num_locs,
        )
        task_names = ["MS_COCO"]
    else:
        raise Exception(f"Task '{args.task}' not supported.")

    # Task details
    task_ids = ["TASK0"]
    task2num_iters = {"TASK0": train_dataset.num_dataset / args.train_batch_size}

    # Logging
    if default_gpu:
        tb_logger = tbLogger(
            logdir, save_path, task_names, task_ids, task2num_iters, args.grad_acc_steps
        )
    else:
        tb_logger = None

    # Wandb
    if args.use_wandb:
        init_wandb(args)

    # Model
    model = BertForVLPreTraining(config)
    checkpoint = torch.load(args.from_pretrained, map_location="cpu")
    model.load_state_dict(checkpoint)

    # Optimization details
    freeze_layers(model)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_weight_name = json.load(
        open(
            "volta/config/bert-base-uncased_weight_name.json",
            "r",
        )
    )
    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": args.weight_decay}
                    ]
        if default_gpu:
            print(
                len(list(model.named_parameters())), len(optimizer_grouped_parameters)
            )
    optimizer = AdamW(
        optimizer_grouped_parameters, eps=args.adam_epsilon, betas=args.adam_betas
    )
    if args.small:
        num_train_optimization_steps = 2310
    else:
        num_train_optimization_steps = (
            int(train_dataset.num_dataset / args.train_batch_size / args.grad_acc_steps)
            * args.num_train_epochs
        )
    warmup_steps = (
        args.warmup_steps or args.warmup_proportion * num_train_optimization_steps
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps
    )
    if args.last_epoch:
        scheduler.last_epoch = args.last_epoch
        scheduler.last_step = num_train_optimization_steps * args.last_epoch

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, _ = resume(
        args.resume_file, model, optimizer, scheduler, tb_logger
    )

    # Move to GPU(s)
    model.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Save initial model
    save(
        save_path,
        logger,
        -1,
        model,
        optimizer,
        scheduler,
        global_step,
        tb_logger,
        default_gpu,
        -1,
    )

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_dataset.num_dataset)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

    # Train
    for epoch_id in range(start_epoch, int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(
            tqdm(train_dataset, desc=f"Epoch {epoch_id}", leave=False)
        ):
            iter_id = start_iter_id + step + (epoch_id * len(train_dataset))
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_match,
                image_feat,
                image_loc,
                image_cls,
                obj_labels,
                obj_confs,
                attr_labels,
                attr_confs,
                image_attrs,
                image_label,
                image_mask,
            ) = batch

            if args.objective == 1:
                # Ignore labels (setting them to -1) for mismatched caption-image pairs
                image_label = image_label * (is_match == 0).long().unsqueeze(1)
                image_label[image_label == 0] = -1
                lm_label_ids = lm_label_ids * (is_match == 0).long().unsqueeze(1)
                lm_label_ids[lm_label_ids == 0] = -1

            train_masked_loss_t, train_masked_loss_v, train_pair_match_loss = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                lm_label_ids,
                image_label,
                image_cls,
                obj_labels,
                obj_confs,
                attr_labels,
                attr_confs,
                image_attrs,
                is_match,
            )
            if args.objective == 2:
                train_pair_match_loss = train_pair_match_loss * 0

            loss = train_masked_loss_t + train_masked_loss_v + train_pair_match_loss
            if n_gpu > 1:
                loss = loss.mean()
                train_masked_loss_t = train_masked_loss_t.mean()
                train_masked_loss_v = train_masked_loss_v.mean()
                train_pair_match_loss = train_pair_match_loss.mean()

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train_CC(
                        epoch_id,
                        iter_id,
                        float(train_masked_loss_t),
                        float(train_masked_loss_v),
                        float(train_pair_match_loss),
                        optimizer.param_groups[0]["lr"],
                        "TASK0",
                        "train",
                    )

                    if (
                        (step % (100 * args.grad_acc_steps) == 0)
                        and step != 0
                        and args.use_wandb
                    ):
                        log_wandb(
                            {
                                "epoch": epoch_id,
                                "step": step,
                                "iter_id": iter_id,
                                "train_masked_loss_t": float(train_masked_loss_t),
                                "train_masked_loss_v": float(train_masked_loss_v),
                                "train_pair_match_loss": float(train_pair_match_loss),
                                "train_loss": loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            }
                        )
                        save(
                            save_path,
                            logger,
                            epoch_id,
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            tb_logger,
                            default_gpu,
                            loss,
                        )

            if (step % (100 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrainCC()

            if args.small and step >= num_train_optimization_steps:
                logger.info(f"Maximum #steps reached ({step}). Stopping training...")
                break

        # Do evaluation
        torch.set_grad_enabled(False)
        numBatches = len(valid_dataset)
        val_masked_loss_t = 0.0
        val_masked_loss_v = 0.0
        val_pair_match_loss = 0.0
        model.eval()
        for step, batch in enumerate(
            tqdm(valid_dataset, desc=f"Validating Epoch {epoch_id}")
        ):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_match,
                image_feat,
                image_loc,
                image_cls,
                obj_labels,
                obj_confs,
                attr_labels,
                attr_confs,
                image_attrs,
                image_label,
                image_mask,
            ) = batch

            batch_size = input_ids.size(0)
            val_masked_loss_t, val_masked_loss_v, val_pair_match_loss = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                lm_label_ids,
                image_label,
                image_cls,
                obj_labels,
                obj_confs,
                attr_labels,
                attr_confs,
                image_attrs,
                is_match,
            )

            loss = val_masked_loss_t + val_masked_loss_v + val_pair_match_loss
            if n_gpu > 1:
                loss = loss.mean()
                val_masked_loss_t = val_masked_loss_t.mean()
                val_masked_loss_v = val_masked_loss_v.mean()
                val_pair_match_loss = val_pair_match_loss.mean()

            if default_gpu:
                tb_logger.step_val_CC(
                    epoch_id,
                    iter_id,
                    float(val_masked_loss_t),
                    float(val_masked_loss_v),
                    float(val_pair_match_loss),
                    "TASK0",
                    batch_size,
                    "val",
                )
                sys.stdout.write("%d / %d \r" % (step, numBatches))
                sys.stdout.flush()

        if default_gpu:
            tb_logger.showLossValCC()

            if args.use_wandb:
                log_wandb(
                    {
                        "val_masked_loss_t": float(val_masked_loss_t),
                        "val_masked_loss_v": float(val_masked_loss_v),
                        "val_pair_match_loss": float(val_pair_match_loss),
                        "val_loss": loss,
                    }
                )

        torch.set_grad_enabled(True)
        save(
            save_path,
            logger,
            epoch_id,
            model,
            optimizer,
            scheduler,
            global_step,
            tb_logger,
            default_gpu,
            loss,
        )

    if default_gpu:
        tb_logger.txt_close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
