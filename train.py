# -*- coding: utf-8 -*-
"""
Training script for Fire Detection Competition (3-class: battery, board, fire).
Uses PicoDet-M with PaddleDetection framework (v13).

Usage:
    python train.py                          # Default config
    python train.py --config configs/xxx.yml # Custom config
    python train.py -r output/N              # Resume from checkpoint N
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PADDLE_DET = os.path.join(BASE_DIR, 'PaddleDetection')
sys.path.insert(0, PADDLE_DET)

CONFIG_FILE = os.path.join(BASE_DIR, "configs", "picodet_m_fire_v13.yml")

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.cli import ArgsParser, merge_args
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger


def main():
    parser = ArgsParser()
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume (e.g. -r output/49)")

    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    has_config = any(a in ('-c', '--config') for a in argv)
    if not has_config:
        argv = ['--config', CONFIG_FILE] + argv

    FLAGS = parser.parse_args(argv)
    cfg = load_config(FLAGS.config)

    cfg.TrainDataset.dataset_dir = BASE_DIR
    cfg.EvalDataset.dataset_dir = BASE_DIR
    cfg.TestDataset.dataset_dir = BASE_DIR

    cfg.use_gpu = True
    cfg.use_npu = False
    cfg.use_xpu = False
    cfg.use_mlu = False

    import paddle
    paddle.set_device('gpu')

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    logger = setup_logger('train')
    logger.info(f"Config: {FLAGS.config}")
    logger.info(f"Num classes: {cfg.num_classes}")
    logger.info(f"Epochs: {cfg.epoch}")
    logger.info(f"Batch size: {cfg.TrainReader['batch_size']}")
    logger.info(f"Base LR: {cfg.LearningRate['base_lr']}")

    trainer = Trainer(cfg, mode='train')

    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    do_eval = getattr(FLAGS, 'eval', True)
    trainer.train(do_eval)


if __name__ == "__main__":
    main()
