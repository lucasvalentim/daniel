# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
#
# Fine-tuning of DANIEL on the BRESSAY dataset (page level, literal annotations).
# Adapted from daniel_custom_dataset_fine_tuning.py.
#
# Prerequisites:
#   1. Format BRESSAY:  python3 Datasets/dataset_formatters/bressay_formatter.py
#      -> creates Datasets/formatted/bressay_page/{train,valid,test} + labels-bressay.pkl
#   2. Download the tokenizer into basic/subwords/tokenizer-daniel/ and replace_dict.pkl
#   3. Download pretrained weights into
#      outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt
#
# Launch:  python3 OCR/document_OCR/daniel/custom_dataset/daniel_bressay_fine_tuning.py
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
from basic.utils import init_wandb

from conf.conf_utils import merge_config_dicts, complete_dataset_params
from conf.training.base import default_training_cfg
from conf.dataset.base import default_dataset_config_factory
from conf.model.base import default_model_cfg

from OCR.document_OCR.daniel.trainer_daniel import run


@click.command()
@click.option('--mode', default='train', help='Execution mode, should be train, synth or eval')
def main(mode):
    dataset_name = "bressay"
    dataset_level = "page"
    dataset_variant = ""  # no semantic/layout tokens -> folder Datasets/formatted/bressay_page

    specific_dataset_cfg = {
        "config": {
            "layout_tokens_mode": 'MULTI',  # keep MULTI to match the pretrained tokenizer vocabulary
            "mean": [223.86277424],  # BRESSAY train mean pixel value (from bressay_formatter.py)
            "std": [46.69775119],    # BRESSAY train std pixel value (from bressay_formatter.py)
            "labels_name": "labels-bressay.pkl",  # name of the labels file
            "height_divisor": 16,  # Image height will be divided by 16
            # Cap the largest pages so the decoder cross-attention fits in 40GB
            # (full-res pages OOM on A100-40GB; aspect ratio is preserved).
            "max_size": {"max_height": 850, "max_width": 800},
            "synthetic_data": None,  # no synthetic data for direct fine-tuning
        },
    }

    specific_model_cfg = {
        "models": {
            "encoder": FCN_Encoder_WidtherFeature  # architecture of the encoder
        },
        "use_checkpointing": True,  # gradient checkpointing to consume less GPU memory
        "transfer_learning": {
            # model_name: [state_dict_name, checkpoint_path, learnable, strict]
            "encoder": ["encoder", "outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt", True, True],
            "decoder": ["decoder", "outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt", True, False],
        }
    }

    specific_training_cfg = {
        "output_folder": "daniel_bressay",  # folder name for checkpoints and results
        "max_nb_epochs": 50000,  # maximum number of epochs before to stop
        "load_epoch": "last",  # ["best", "last"]: last to continue training, best to evaluate
        "batch_size": 1,  # mini-batch size for training
        "valid_batch_size": 4,  # mini-batch size for validation
        "test_batch_size": 4,  # mini-batch size for test
        "optimizers": {
            "all": {
                "class": Adam,
                "args": {
                    "lr": 1e-5,
                    "amsgrad": False,
                }
            },
        },
        "eval_on_valid": True,  # eval and log metrics on validation set during training
        "eval_on_valid_interval": 5,  # interval (in epochs) to evaluate during training
        "focus_metric": "cer",  # metric to determine best epoch
        "expected_metric_value": "low",  # ["high", "low"] best for the focus metric
        "set_name_focus_metric": "{}-valid".format(dataset_name),  # dataset to select best weights
        "train_metrics": ["loss_ce", "cer", "wer", "syn_max_lines"],  # training metrics
        "eval_metrics": ["cer", "wer"],  # evaluation metrics
        "force_cpu": False,  # True for debug purposes
        "max_char_prediction": 2300,  # max number of subwords in a predicted sequence
        "teacher_forcing_scheduler": {
            "min_error_rate": 0.3,
            "max_error_rate": 0.3,
            "total_num_steps": 5e6,
        },
        "use_wandb": False,
    }

    params = {}

    params['model_params'] = merge_config_dicts(default_model_cfg, specific_model_cfg)

    default_dataset_cfg = default_dataset_config_factory(dataset_name, dataset_level, dataset_variant)
    params['dataset_params'] = merge_config_dicts(default_dataset_cfg, specific_dataset_cfg)
    params['training_params'] = merge_config_dicts(default_training_cfg, specific_training_cfg)

    params['dataset_params'] = complete_dataset_params(params['dataset_params'], params['model_params'], params['training_params'])

    params["model_params"]["max_char_prediction"] = params["training_params"]["max_char_prediction"]
    if params["training_params"].get("use_wandb", False):
        init_wandb(projet_name="daniel", exp_id='bressay-fine-tuning', params=params, dataset_name=dataset_name)

    if mode == "eval":
        # set_names must be a list (the default ("test") is a bare string)
        run(params, mode=mode, dataset_names=[dataset_name], metrics=["cer", "wer"], set_names=["test"])
    else:
        run(params, mode=mode, dataset_names=[dataset_name])


if __name__ == "__main__":
    main()
