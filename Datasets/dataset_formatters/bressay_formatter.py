# BRESSAY dataset formatter for DANIEL (DAN format).
#
# Converts the ICDAR 2024 BRESSAY dataset (handwritten Brazilian-Portuguese
# essays) into the format expected by DANIEL:
#   Datasets/formatted/bressay_page/
#       train/<name>.png
#       valid/<name>.png
#       test/<name>.png
#       labels-bressay.pkl   -> {"charset": [...], "ground_truth": {train,valid,test}}
#
# Page-level transcriptions are kept *literal*: the BRESSAY inline annotations
# (##..##, --..--, @@???@@, $$..$$, ##--..--##, etc.) are preserved as-is, so
# the model learns to reproduce the official competition transcription format.
#
# Usage (from the DANIEL repo root):
#   python3 Datasets/dataset_formatters/bressay_formatter.py \
#       --bressay-root ../datasets/bressay \
#       --output-dir Datasets/formatted/bressay_page
#
# The image mean/std printed at the end must be copied into the training script
# (daniel_bressay_fine_tuning.py) for normalization.

import argparse
import os
import pickle
import re

import numpy as np
from PIL import Image

SET_FILE_TO_SPLIT = {
    "training": "train",
    "validation": "valid",
    "test": "test",
}


def read_split_names(sets_dir):
    """Return {split: [page_name, ...]} from the sets/*.txt partition files."""
    splits = {}
    for set_file, split in SET_FILE_TO_SPLIT.items():
        path = os.path.join(sets_dir, set_file + ".txt")
        with open(path, encoding="utf-8") as f:
            splits[split] = [line.strip() for line in f if line.strip()]
    return splits


def format_text_label(label):
    """Normalize whitespace while keeping paragraph breaks (blank lines).

    - strip trailing spaces on each line
    - collapse runs of >=2 blank lines into a single blank line
    - strip leading/trailing blank lines and spaces
    The literal annotation markers are left untouched.
    """
    lines = [line.rstrip() for line in label.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n ")


def build(bressay_root, output_dir, labels_name):
    pages_dir = os.path.join(bressay_root, "data", "pages")
    sets_dir = os.path.join(bressay_root, "sets")
    splits = read_split_names(sets_dir)

    ground_truth = {}
    charset = set()
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for split, names in splits.items():
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        split_gt = {}
        for name in names:
            src_png = os.path.join(pages_dir, name, name + ".png")
            src_txt = os.path.join(pages_dir, name, name + ".txt")

            with open(src_txt, encoding="utf-8") as f:
                text = format_text_label(f.read())

            filename = name + ".png"
            dst_png = os.path.join(output_dir, split, filename)
            img = Image.open(src_png).convert("L")
            img.save(dst_png)

            # Accumulate stats from the training set only (used for normalization).
            if split == "train":
                arr = np.asarray(img, dtype=np.float64)
                pixel_sum += arr.sum()
                pixel_sq_sum += (arr ** 2).sum()
                pixel_count += arr.size

            charset.update(text)
            split_gt[filename] = {
                "text": text,
                "nb_cols": 1,
                "pages": [{"text": text, "nb_cols": 1, "paragraphs": []}],
            }
        ground_truth[split] = split_gt
        print(f"[{split}] {len(split_gt)} pages")

    charset.discard("")
    formatted = {
        "charset": sorted(charset),
        "ground_truth": ground_truth,
    }
    labels_path = os.path.join(output_dir, labels_name)
    with open(labels_path, "wb") as f:
        pickle.dump(formatted, f)

    mean = pixel_sum / pixel_count
    std = (pixel_sq_sum / pixel_count - mean ** 2) ** 0.5
    print(f"\nLabels written to {labels_path}")
    print(f"Charset size: {len(formatted['charset'])}")
    print(f"Charset: {''.join(c for c in formatted['charset'] if c != chr(10))!r}")
    print(f"\n>>> Copy these into the training script (normalization):")
    print(f'    "mean": [{mean:.8f}],')
    print(f'    "std": [{std:.8f}],')


def main():
    parser = argparse.ArgumentParser(description="Format BRESSAY for DANIEL.")
    parser.add_argument(
        "--bressay-root", default="../datasets/bressay",
        help="Path to the bressay dataset root (containing data/ and sets/).",
    )
    parser.add_argument(
        "--output-dir", default="Datasets/formatted/bressay_page",
        help="Destination folder for the formatted dataset.",
    )
    parser.add_argument(
        "--labels-name", default="labels-bressay.pkl",
        help="Name of the labels pickle file.",
    )
    args = parser.parse_args()
    build(args.bressay_root, args.output_dir, args.labels_name)


if __name__ == "__main__":
    main()
