# -*- coding: utf-8 -*-
"""
Evaluate trained model on validation set.
Computes COCO mAP and per-class F1 scores.

Usage:
    python evaluate.py
"""
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PADDLE_DET = os.path.join(BASE_DIR, 'PaddleDetection')
sys.path.insert(0, PADDLE_DET)

from ppdet.engine import Trainer
from ppdet.core.workspace import load_config


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_per_class_f1(gt_path, dt_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)

    cat_names = {1: 'battery', 2: 'board', 3: 'fire'}
    f1_scores = {}

    print("\n" + "=" * 60)
    print("Per-class F1 Scores @ IoU=0.50")
    print("=" * 60)
    print(f"{'Class':>12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
    print("-" * 50)

    for cat_id in [1, 2, 3]:
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [cat_id]
        coco_eval.params.iouThrs = [0.5]
        coco_eval.evaluate()
        coco_eval.accumulate()

        precision = coco_eval.eval['precision'][0, :, 0, 0, 2]
        recall = coco_eval.eval['recall'][:, 0, 0, 2]

        max_prec = float(np.max(precision)) if len(precision) > 0 else 0.0
        max_rec = float(np.max(recall)) if len(recall) > 0 else 0.0
        f1 = compute_f1(max_prec, max_rec)
        f1_scores[cat_id] = f1

        print(f"{cat_names[cat_id]:>12s}  {max_prec:10.4f}  {max_rec:10.4f}  {f1:10.4f}")

    print("-" * 50)
    mean_f1 = np.mean(list(f1_scores.values()))
    print(f"{'Mean':>12s}  {'':>10s}  {'':>10s}  {mean_f1:10.4f}")
    print("=" * 60)

    return mean_f1


def main():
    config_path = os.path.join(BASE_DIR, "configs", "picodet_l_fire_v15.yml")
    cfg = load_config(config_path)
    cfg.TrainDataset.dataset_dir = BASE_DIR
    cfg.EvalDataset.dataset_dir = BASE_DIR

    ckpt_dir = os.path.join(BASE_DIR, "output")
    weights_path = None
    for fname in ['best_model.pdparams', 'model_final.pdparams']:
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            weights_path = path
            break

    if not weights_path:
        import glob
        checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "*.pdparams")))
        if checkpoints:
            weights_path = checkpoints[-1]

    if not weights_path:
        print("ERROR: No checkpoint found!")
        sys.exit(1)

    print(f"Loading: {weights_path}")

    trainer = Trainer(cfg, mode='eval')
    trainer.load_weights(weights_path)
    trainer.evaluate()

    dt_path = os.path.join(BASE_DIR, "bbox.json")
    gt_path = os.path.join(BASE_DIR, "A_train", "val.json")
    compute_per_class_f1(gt_path, dt_path)


if __name__ == "__main__":
    main()
