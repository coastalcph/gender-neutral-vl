import argparse
import sys
import wandb
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--annotations_path",
        default="datasets/conceptual_caption/annotations",
        type=str,
        help="The corpus annotations directory.",
    )
    parser.add_argument(
        "--features_path",
        default="/home/sxk199/mnt/mscoco/images",
        type=str,
        help="The corpus image features directory.",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="Name of the pretraining task",
    )
    parser.add_argument(
        "--masked_splits",
        default=None,
        type=str,
        help="Path to file containing the img_id in each split",
    )
    parser.add_argument(
        "--dataset_splits_dir",
        default=None,
        type=str,
        help="Path to directory with dataset_masked_splits.json (gender-mappings folder)",
    )
    # Model
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        choices=[
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-base-multilingual",
            "bert-base-chinese",
        ],
        help="Bert pre-trained model",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/vilbert_base.json",
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    # Text
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    # Training
    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="grad_acc_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--p_neutral_cap",
        default=0.15,
        type=float,
        help="Initial probability of inserting a gender neutral sentence.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Train on a reduced version of CC with the same training steps as COCO",
    )
    # Scheduler
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=None,
        type=float,
        help="Number of training steps to perform linear learning rate warmup for. "
        "It overwrites --warmup_proportion.",
    )
    parser.add_argument(
        "--last_epoch", default=None, type=int, help="Last epoch if resuming training"
    )
    # Objective
    parser.add_argument(
        "--objective",
        default=0,
        type=int,
        help="Which objective to use \n"
        "0: with ITM loss, \n"
        "1: with ITM loss; for the not aligned pair, no masking objective, \n"
        "2: without ITM loss, do not sample negative pair.",
    )
    # Optimizer
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_betas",
        default=(0.9, 0.98),
        nargs="+",
        type=float,
        help="Betas for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--clip_grad_norm",
        default=0.0,
        type=float,
        help="Clip gradients within the specified range.",
    )
    # Seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization"
    )
    # Output
    parser.add_argument(
        "--output_dir",
        default="checkpoints",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--logdir",
        default="logs",
        type=str,
        help="The logging directory where the training logs will be written.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="If passed, will append a timestamp to output_dir.",
    )
    # Distributed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=25,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="whether use chunck for parallel training.",
    )
    # Wandb
    parser.add_argument(
        "--use_wandb", action="store_true", help="If passed, will log to wandb."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="lcp",
        help="Username or team name where you're sending runs.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MM-GB",
        help="Project name in wandb.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="", help="Name of the wandb run."
    )
    return parser.parse_args()


def parse_task_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument("--is_m3p", action="store_true", default=False, help="Use M3P.")
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    # Task
    parser.add_argument("--train_split", default="", type=str)
    parser.add_argument("--val_split", default="", type=str)
    parser.add_argument(
        "--tasks_config_file",
        default="config_tasks/vilbert_trainval_tasks.yml",
        type=str,
        help="The config file which specified the tasks details.",
    )
    parser.add_argument("--task", default="", type=str, help="training task number")
    parser.add_argument(
        "--train_annotations_jsonpath",
        default="",
        type=str,
        help="train_annotations_jsonpath",
    )
    parser.add_argument(
        "--val_annotations_jsonpath",
        default="",
        type=str,
        help="val_annotations_jsonpath",
    )
    parser.add_argument("--train_features_lmdbpath", default="", type=str)
    parser.add_argument(
        "--val_features_lmdbpath",
        default="",
        type=str,
        help="alternative features lmdb path",
    )
    # Training & Evaluation
    parser.add_argument(
        "--num_epoch",
        default=None,
        type=int,
        help="Max number of training epochs to perform.",
    )
    parser.add_argument(
        "--optim_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="grad_acc_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--drop_last", action="store_true", help="whether to drop last incomplete batch"
    )
    parser.add_argument(
        "--batch_size",
        default=30,
        type=int,
        help="overwrites the config_tasks batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=None,
        type=int,
        help="overwrites the config_tasks batch size",
    )
    parser.add_argument("--max_val_batches", default=-1, type=int)
    parser.add_argument("--loss", default="", type=str, help="alternative loss name")
    parser.add_argument(
        "--eval_steps", default=sys.maxsize, type=int, help="when to evaluate model"
    )
    parser.add_argument("--cache", default=5000, type=int)
    parser.add_argument(
        "--do_early_stopping",
        action="store_true",
        help="whether to implement early stopping over the validation loss",
    )
    parser.add_argument(
        "--patience",
        default=1,
        type=int,
        help="Patience in terms of epochs to implement in early stopping",
    )
    # Scheduler
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="overwrites the config_tasks learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="warmup_linear",
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=None,
        type=float,
        help="Number of training steps to perform linear learning rate warmup for. "
        "It overwrites --warmup_proportion.",
    )
    # Distributed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument("--num_val_workers", type=int, default=2)
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )
    # Optimization
    parser.add_argument(
        "--optim", default="AdamW", type=str, help="Name of the optimizer to use."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_betas",
        default=(0.9, 0.999),
        nargs="+",
        type=float,
        help="Betas for Adam optimizer.",
    )
    parser.add_argument(
        "--adam_correct_bias",
        default=False,
        action="store_true",
        help="Correct bias for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--clip_grad_norm",
        default=0.0,
        type=float,
        help="Clip gradients within the specified range.",
    )
    # Seed
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )
    # Output
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--logdir",
        default="logs",
        type=str,
        help="The logging directory where the training logs will be written.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument("--save_best_only", default=False, action="store_true")
    parser.add_argument("--save_every_num_epochs", default=1, type=int)
    # Wandb
    parser.add_argument(
        "--use_wandb", action="store_true", help="If passed, will log to wandb."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="lcp",
        help="Username or team name where you're sending runs.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MM-GB",
        help="Project name in wandb.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="", help="Name of the wandb run."
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Contatenate seed value to output dir"
    )
    return parser.parse_args()


def parse_retrieval_task():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument("--is_m3p", action="store_true", default=False, help="Use M3P.")
    # Output
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    # Task
    parser.add_argument(
        "--tasks_config_file",
        default="config_tasks/vilbert_trainval_tasks.yml",
        type=str,
        help="The config file which specified the tasks details.",
    )
    parser.add_argument("--task", default="", type=str, help="training task number")
    parser.add_argument("--val_annotations_jsonpath", default="", type=str)
    parser.add_argument("--val_features_lmdbpath", default="", type=str)
    parser.add_argument("--num_subiters", default=2, type=int)
    parser.add_argument(
        "--caps_per_image", default=5, type=int, help="Num captions per image"
    )
    # Evaluation
    parser.add_argument(
        "--zero_shot", action="store_true", help="Zero-shot evaluation."
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size.")
    parser.add_argument(
        "--drop_last", action="store_true", help="whether to drop last incomplete batch"
    )
    # Seed
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    # Distributed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument("--num_val_workers", type=int, default=10)
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )

    return parser.parse_args()


# Wandb methods
def init_wandb(args: argparse):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=args,
        settings=wandb.Settings(start_method="fork"),
    )


def define_metric_wandb(metric: str):
    wandb.run.define_metric(metric, summary="max")


def log_wandb(metrics: Dict):
    wandb.log(metrics)
