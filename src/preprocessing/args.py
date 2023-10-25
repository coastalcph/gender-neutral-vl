import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="val", choices=["val", "train"], help="Split to preprocess"
    )
    parser.add_argument(
        "--input_dir",
        default="/Users/sxk199/PhD/data/mscoco/datasets/lxmert_splits/data/vqa",
        type=str,
        help="Path to the input folder with captions in json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    # COCO
    parser.add_argument(
        "--princeton_demo",
        default="/Users/sxk199/PhD/data/mscoco/datasets/COCO2014_VAL_DEMOGRAPHICS",
        type=str,
        help="Path to the folder with annotated data from Princeton (for comparison only)",
    )
    parser.add_argument(
        "--coco_karpathy",
        default="/Users/sxk199/PhD/data/mscoco/datasets/karpathy_caption_datasets/dataset_coco.json",
        type=str,
        help="Path to the Karpathy version of the COCO dataset",
    )

    return parser.parse_args()
