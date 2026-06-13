# -*- coding: utf-8 -*-
"""Export trained PicoDet-M model to Paddle Inference format.

Usage:
    python export_model.py
"""
import os
import sys
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PADDLE_DET = os.path.join(BASE_DIR, 'PaddleDetection')
sys.path.insert(0, PADDLE_DET)

import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer


def find_best_weights():
    ckpt_dir = os.path.join(BASE_DIR, "output")
    for fname in ['best_model.pdparams', 'model_final.pdparams']:
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            return path
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "*.pdparams")))
    return checkpoints[-1] if checkpoints else None


def main():
    paddle.set_device('gpu')

    config_path = os.path.join(BASE_DIR, "configs", "picodet_m_fire_v13.yml")
    cfg = load_config(config_path)
    cfg.TrainDataset.dataset_dir = BASE_DIR
    cfg.EvalDataset.dataset_dir = BASE_DIR
    cfg.TestDataset.dataset_dir = BASE_DIR

    weights_path = find_best_weights()
    if not weights_path:
        print("ERROR: No checkpoint found in output/")
        sys.exit(1)

    print(f"Loading weights: {weights_path}")

    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(weights_path)

    output_dir = os.path.join(BASE_DIR, "model")
    trainer.export(output_dir)

    # trainer.export creates a subdirectory; move files up
    import shutil
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    subdir = os.path.join(output_dir, subdirs[0])
    for f in os.listdir(subdir):
        src = os.path.join(subdir, f)
        dst = os.path.join(output_dir, f)
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.move(src, dst)
    os.rmdir(subdir)

    # Verify exported files
    print("\nExported files:")
    total = 0
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}: {size_mb:.2f} MB")
            total += os.path.getsize(fpath)
    print(f"Total: {total / (1024*1024):.1f} MB (limit: 200 MB)")


if __name__ == "__main__":
    main()
