# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Script heavily based on https://github.com/e-bug/volta/blob/main/train_task.py

import os
import sys
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict
import numpy as np
import torch
import torch.distributed as dist
from train_utils import (
    init_wandb,
    log_wandb,
    define_metric_wandb,
    parse_task_args,
)

from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from early_stopping import EarlyStopping
from volta.volta.config import BertConfig, M3PConfig
from volta.volta.optimization import RAdam
from volta.volta.encoders import BertForVLTasks, M3PForVLTasks
from volta.volta.train_utils import (
    freeze_layers,
    tbLogger,
    summary_parameters,
    save,
    resume,
)
from volta.volta.task_utils import (
    LoadDataset,
    LoadLoss,
    ForwardModelsTrain,
    ForwardModelsVal,
)

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
    base_lr = args.lr or task_cfg[task]["lr"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    if args.sweep:
        prefix += f"_seed_{args.seed}"
    save_path = os.path.join(args.output_dir, task_name + prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val = LoadDataset(
        args, config, task_cfg, args.task
    )

    # Logging
    tb_logger = tbLogger(
        args.logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps
    )

    # Wandb
    if args.use_wandb:
        init_wandb(args)
        define_metric_wandb("val_score")

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
    if task_cfg[task].get("embed_clf", None):
        logger.info(
            "Initializing classifier weight for %s from pretrained word embeddings..."
            % task
        )
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if "bert.embeddings.word_embeddings.weight" in k:
                word_embeddings = v.detach().clone()
                break
        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (
                    torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)
                ).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith("clfs_dict.%s.logit_fc.3" % task):
                module.weight.data = answers_word_embed_tensor.to(
                    device=module.weight.data.device
                )

    # Optimization details
    freeze_layers(model)
    criterion = LoadLoss(args, task_cfg, args.task)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": args.weight_decay}
                ]
    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    # num_train_optim_steps = (task2num_iters[task] * args.optim_train_epochs // args.grad_acc_steps)
    max_epoch = args.num_epoch or task_cfg[task]["num_epoch"]
    num_train_optim_steps = task2num_iters[task] * max_epoch // args.grad_acc_steps
    if args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=base_lr,
            eps=args.adam_epsilon,
            betas=args.adam_betas,
            correct_bias=args.adam_correct_bias,
        )

    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
    if args.lr_scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps
        )
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

    # Early Stopping
    if args.do_early_stopping:
        es = EarlyStopping(patience=args.patience)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, max_score = resume(
        args.resume_file, model, optimizer, scheduler, tb_logger
    )

    # Move to GPU(s)
    model.to(device)
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
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Save starting model
    if start_epoch == 0:
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
            max_score,
        )

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        print("***** Running training *****")
        print("  Num Iters: ", task2num_iters[task])
        print("  Batch size: ", batch_size)
        print("  Num steps: %d" % num_train_optim_steps)

    # Train
    scores = 0
    for epoch_id in tqdm(range(start_epoch, max_epoch), desc="Epoch"):
        epoch_score = 0
        epoch_loss = []
        model.train()
        for step, batch in enumerate(dl_train):
            iter_id = (
                start_iter_id + step // args.grad_acc_steps + (epoch_id * len(dl_train))
            )

            loss, score = ForwardModelsTrain(
                config, task_cfg, device, task, batch, model, criterion
            )
            scores += score
            epoch_score += score

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

                if args.optim != "bert":
                    if (
                        global_step < warmup_steps
                        or args.lr_scheduler == "warmup_linear"
                    ):
                        scheduler.step()

                model.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train(
                        epoch_id,
                        iter_id,
                        float(loss),
                        float(scores / args.grad_acc_steps),
                        optimizer.param_groups[0]["lr"],
                        task,
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
                                "train_score": float(scores / args.grad_acc_steps),
                                "train_loss": loss,
                            }
                        )
                    epoch_loss.append(float(loss))
                    scores = 0

            if (step % (100 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrain()

            # Decide whether to evaluate task
            if iter_id != 0 and iter_id % (args.eval_steps - 1) == 0:
                score, _, _, _ = evaluate(
                    config,
                    dl_val,
                    task_cfg,
                    device,
                    task,
                    model,
                    criterion,
                    epoch_id,
                    step,
                    default_gpu,
                    tb_logger,
                )
                if score > max_score:
                    max_score = score
                    save(
                        save_path,
                        logger,
                        iter_id,
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        tb_logger,
                        default_gpu,
                        max_score,
                        is_best=True,
                    )

        torch.cuda.empty_cache()

        eval_score, eval_loss, vil_logit, target = evaluate(
            config,
            dl_val,
            task_cfg,
            device,
            task,
            model,
            criterion,
            epoch_id,
            step,
            default_gpu,
            tb_logger,
            args.max_val_batches,
        )

        if eval_score > max_score:
            max_score = eval_score
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
                max_score,
                is_best=True,
            )
        elif (not args.save_best_only) and (
            (epoch_id + 1) % args.save_every_num_epochs == 0
        ):
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
                max_score,
            )

        if args.use_wandb:
            log_wandb(
                {
                    "epoch": epoch_id,
                    "train_score": float(epoch_score),
                    "train_loss": np.mean(epoch_loss),
                    "val_score": eval_score,
                    "val_loss": eval_loss,
                }
            )

        if args.do_early_stopping:
            if es.step(eval_loss):
                logger.info(
                    f"Early stop criterion is met. Terminating training after epoch {epoch_id+1}"
                )
                break

    tb_logger.txt_close()
    print("Best Validation score: %.3f " % (max_score * 100.0))


def evaluate(
    config,
    dataloader_val,
    task_cfg,
    device,
    task_id,
    model,
    criterion,
    epoch_id,
    step,
    default_gpu,
    tb_logger,
    num_batches=-1,
):
    model.eval()
    vil_logit = 0
    target = ""
    for i, batch in enumerate(dataloader_val):
        if i == (num_batches - 1):
            break
        loss, score, batch_size, vil_logit, target = ForwardModelsVal(
            config, task_cfg, device, task_id, batch, model, criterion
        )
        tb_logger.step_val(
            epoch_id, float(loss), float(score), task_id, batch_size, "val"
        )
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(dataloader_val)))
            sys.stdout.flush()

    score, loss = tb_logger.showLossVal(task_id)

    return score, float(loss), vil_logit, target


if __name__ == "__main__":
    main()
