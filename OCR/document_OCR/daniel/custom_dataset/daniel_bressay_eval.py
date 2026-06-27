# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
#
# Standalone evaluation of a fine-tuned DANIEL/BRESSAY checkpoint.
# Mirrors daniel_bressay_fine_tuning.py but runs mode='eval' on a chosen split,
# computing CER/WER and saving per-image predictions.
#
#   python3 OCR/document_OCR/daniel/custom_dataset/daniel_bressay_eval.py --set valid
#
# NOTE: run() -> evaluate_single_model() forces load_epoch="best" and loads the
# FIRST file matching "best" in the checkpoints dir (get_checkpoint does not sort).
# So keep ONLY the desired best_N.pt in outputs/daniel_bressay/checkpoints/ before
# evaluating (e.g. the highest best_N = lowest valid CER).
# Run on a FREE GPU (not while training) to avoid OOM from two models.
import os
import sys

import click
from torch.optim import Adam

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))))

from basic.encoders import FCN_Encoder_WidtherFeature
from conf.conf_utils import merge_config_dicts, complete_dataset_params
from conf.training.base import default_training_cfg
from conf.dataset.base import default_dataset_config_factory
from conf.model.base import default_model_cfg
from OCR.document_OCR.daniel.trainer_daniel import run


@click.command()
@click.option('--set', 'set_name', default='valid', help="Split to evaluate: valid or test")
def main(set_name):
    dataset_name = "bressay"
    dataset_level = "page"
    dataset_variant = ""

    specific_dataset_cfg = {
        "config": {
            "layout_tokens_mode": 'MULTI',
            "mean": [223.86277424],
            "std": [46.69775119],
            "labels_name": "labels-bressay.pkl",
            "height_divisor": 16,
            "max_size": {"max_height": 850, "max_width": 800},
            "synthetic_data": None,
        },
    }
    specific_model_cfg = {
        "models": {"encoder": FCN_Encoder_WidtherFeature},
        "use_checkpointing": True,
    }
    specific_training_cfg = {
        "output_folder": "daniel_bressay",
        "load_epoch": "best",          # evaluate the best checkpoint (see note above)
        "valid_batch_size": 4,
        "test_batch_size": 4,
        "optimizers": {"all": {"class": Adam, "args": {"lr": 1e-5, "amsgrad": False}}},
        "eval_metrics": ["cer", "wer"],
        "force_cpu": False,
        "max_char_prediction": 2300,
        "use_wandb": False,
    }

    params = {}
    params['model_params'] = merge_config_dicts(default_model_cfg, specific_model_cfg)
    default_dataset_cfg = default_dataset_config_factory(dataset_name, dataset_level, dataset_variant)
    params['dataset_params'] = merge_config_dicts(default_dataset_cfg, specific_dataset_cfg)
    params['training_params'] = merge_config_dicts(default_training_cfg, specific_training_cfg)
    params['dataset_params'] = complete_dataset_params(params['dataset_params'], params['model_params'], params['training_params'])
    params["model_params"]["max_char_prediction"] = params["training_params"]["max_char_prediction"]

    run(params, mode='eval', dataset_names=[dataset_name],
        metrics=["cer", "wer"], set_names=[set_name])


if __name__ == "__main__":
    main()
