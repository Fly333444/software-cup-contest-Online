# -*- coding: utf-8 -*-
"""Export trained Faster R-CNN firebig model to Paddle Inference format."""
import os
import sys
import glob
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PADDLE_DET = os.path.join(BASE_DIR, 'PaddleDetection')
sys.path.insert(0, PADDLE_DET)

import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer


CONFIG_FILE = os.path.join(BASE_DIR, "configs", "faster_rcnn_r50_fpn_firebig_b01.yml")

def find_best_weights():
    config_dir = os.path.join(BASE_DIR, "output", os.path.splitext(os.path.basename(CONFIG_FILE))[0])
    for fname in ['best_model.pdparams', 'model_final.pdparams']:
        path = os.path.join(config_dir, fname)
        if os.path.exists(path):
            return path
    # Fallback to root output
    ckpt_dir = os.path.join(BASE_DIR, "output")
    for fname in ['best_model.pdparams', 'model_final.pdparams']:
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            return path
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "*.pdparams")))
    return checkpoints[-1] if checkpoints else None

def main():
    paddle.set_device('gpu')

    config_path = CONFIG_FILE
    cfg = load_config(config_path)

    weights_path = find_best_weights()
    if not weights_path:
        print("ERROR: No checkpoint found in output/")
        sys.exit(1)

    print(f"Loading weights: {weights_path}")

    # Override NMS settings to match training config
    # Faster R-CNN 用 BBoxPostProcess.nms，PicoDet 用 PicoHeadV2.nms
    if hasattr(cfg, 'BBoxPostProcess') and hasattr(cfg.BBoxPostProcess, 'nms'):
        cfg.BBoxPostProcess.nms.score_threshold = 0.025
        cfg.BBoxPostProcess.nms.nms_threshold = 0.6
        cfg.BBoxPostProcess.nms.keep_top_k = 100
        print("NMS overridden: score_threshold=0.025, nms_threshold=0.6")
    elif hasattr(cfg, 'PicoHeadV2') and hasattr(cfg.PicoHeadV2, 'nms'):
        cfg.PicoHeadV2.nms.score_threshold = 0.025
        cfg.PicoHeadV2.nms.nms_threshold = 0.6
        cfg.PicoHeadV2.nms.keep_top_k = 100
        cfg.PicoHeadV2.nms.nms_top_k = 1000
        print("NMS overridden: score_threshold=0.025, nms_threshold=0.6")

    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(weights_path)

    output_dir = os.path.join(BASE_DIR, "model")
    # Clean old export
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
    os.makedirs(output_dir, exist_ok=True)
    trainer.export(output_dir)

    # trainer.export creates a subdirectory; move files up
    subdirs = [d for d in os.listdir(output_dir)
               if os.path.isdir(os.path.join(output_dir, d))]
    if subdirs:
        subdir = os.path.join(output_dir, subdirs[0])
        for f in os.listdir(subdir):
            src = os.path.join(subdir, f)
            dst = os.path.join(output_dir, f)
            if os.path.isfile(dst):
                os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(subdir)

    # Verify
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
