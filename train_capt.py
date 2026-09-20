"""
train_capt.py
=============
Drop-in entry point for CAPT on BiomedCLIP. It is a copy of BiomedCoOp's
``train.py`` with (1) the ``CAPT`` config node registered in ``extend_cfg`` and
(2) the CAPT trainer imported so Dassl can find it. Use this instead of
``train.py`` when the trainer is ``CAPT_BiomedCLIP``.

Everything else (dataset registration, argument parsing) is unchanged, so all
BiomedCoOp datasets and scripts keep working.
"""

import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# datasets (register them with Dassl)
import datasets.busi
import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.kneexray
import datasets.dermamnist
import datasets.octmnist

# base trainers (needed to build the Stage-1 model / for comparison)
import trainers.CoOp.coop_biomedclip
import trainers.BiomedCoOp.biomedcoop_biomedclip
# CAPT trainer
import trainers.CAPT.capt_biomedclip


def print_args(args, cfg):
    print("***************\n** Arguments **\n***************")
    for key in sorted(args.__dict__.keys()):
        print("{}: {}".format(key, args.__dict__[key]))
    print("************\n** Config **\n************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # ---- CoOp (base model option) ----
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = "a photo of a"
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    # ---- BiomedCoOp (base model option / comparison) ----
    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIOMEDCOOP.CSC = False
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 2.0
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 0.5
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50

    # ---- CAPT ----
    cfg.TRAINER.CAPT = CN()
    cfg.TRAINER.CAPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.CAPT.CSC = False
    cfg.TRAINER.CAPT.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.CAPT.N_CTX = 4
    cfg.TRAINER.CAPT.PREC = "fp32"
    cfg.TRAINER.CAPT.K_PAIRS = 3          # confusion pairs per sample (<= n_cls-1)
    cfg.TRAINER.CAPT.ALPHA_S = 5.0        # dynamic alpha scale  (Eq. 13)
    cfg.TRAINER.CAPT.ALPHA_GAMMA = 0.5    # dynamic alpha exponent
    cfg.TRAINER.CAPT.MGDE_TOPK = 2        # experts kept by the router
    cfg.TRAINER.CAPT.BETA = 0.1           # residual strength of the MGDE output
    cfg.TRAINER.CAPT.CONF_LAMBDA = 1.0    # weight of the confusion InfoNCE loss
    cfg.TRAINER.CAPT.BASE_CKPT = ""       # dir of the Stage-1 base checkpoint
    cfg.TRAINER.CAPT.BASE_EPOCH = None    # epoch of the base checkpoint


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="resume checkpoint dir")
    parser.add_argument("--seed", type=int, default=-1, help="fixed seed if >=0")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
    parser.add_argument("--config-file", type=str, default="", help="method config")
    parser.add_argument("--dataset-config-file", type=str, default="", help="dataset config")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="backbone name")
    parser.add_argument("--head", type=str, default="", help="head name")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="dir for eval-only")
    parser.add_argument("--load-epoch", type=int, help="epoch to load for eval")
    parser.add_argument("--no-train", action="store_true", help="do not train")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config from the command line")
    args = parser.parse_args()
    main(args)
