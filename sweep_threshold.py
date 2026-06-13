# -*- coding: utf-8 -*-
"""Sweep score thresholds to find the optimal value for Mean F1.

Usage:
    python sweep_threshold.py
"""
import os
import sys
import json
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PaddleDetection'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PaddleDetection', 'deploy', 'python'))

import numpy as np
import paddle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml

from preprocess import preprocess, Resize, NormalizeImage, Permute


class SweepPredictor:
    """Lightweight predictor matching predict.py logic for threshold sweeping."""

    def __init__(self, model_dir):
        from paddle.inference import Config, create_predictor

        config = Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams'),
        )
        config.enable_use_gpu(500, 0)
        config.switch_ir_optim(False)  # Match predict.py: disable on RTX 4060
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)

        self.predictor = create_predictor(config)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)

        self.preprocess_ops = []
        for op_info in yml_conf.get('Preprocess', []):
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            self.preprocess_ops.append(eval(op_type)(**new_op_info))

    def predict_batch(self, imgs, im_info):
        """Stack identically-sized images directly (keep_ratio=False → all 416×416)."""
        inputs = {}
        im_shape = []
        scale_factor = []
        for e in im_info:
            im_shape.append(np.array((e['im_shape'],)).astype('float32'))
            scale_factor.append(np.array((e['scale_factor'],)).astype('float32'))
        inputs['image'] = np.stack([np.array(img, dtype=np.float32) for img in imgs], axis=0)
        if 'im_shape' in self.input_names:
            inputs['im_shape'] = np.stack(im_shape, axis=0)
        if 'scale_factor' in self.input_names:
            inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

        for name in self.input_names:
            t = self.predictor.get_input_handle(name)
            t.copy_from_cpu(inputs[name])
        self.predictor.run()

        num_outs = int(len(self.output_names) / 2)
        np_boxes = self.predictor.get_output_handle(self.output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(self.output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_mean_f1(gt_path, dt_path):
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)

    f1_scores = {}
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
        f1_scores[cat_id] = compute_f1(max_prec, max_rec)

    return float(np.mean(list(f1_scores.values()))), f1_scores


def run_sweep(predictor, val_json, image_dir, thresholds):
    """Run prediction at each threshold and compute Mean F1."""
    with open(val_json) as f:
        val_data = json.load(f)

    image_map = {img['id']: img['file_name'] for img in val_data['images']}
    image_list = [(img_id, os.path.join(image_dir, image_map[img_id]))
                  for img_id in sorted(image_map.keys())]

    all_detections = []
    batch_size = 64
    for start in range(0, len(image_list), batch_size):
        batch = image_list[start:start + batch_size]
        imgs, im_infos, img_ids = [], [], []
        for img_id, im_path in batch:
            im, im_info = preprocess(im_path, predictor.preprocess_ops)
            imgs.append(im)
            im_infos.append(im_info)
            img_ids.append(img_id)

        det_results = predictor.predict_batch(imgs, im_infos)
        boxes = det_results['boxes']
        boxes_num = det_results['boxes_num'].astype('int32').flatten()

        box_start = 0
        for img_id, n in zip(img_ids, boxes_num):
            box_end = box_start + int(n)
            for det in boxes[box_start:box_end]:
                score = float(det[1])
                x1, y1, x2, y2 = map(float, det[2:6])
                all_detections.append({
                    "image_id": int(img_id),
                    "category_id": int(det[0]) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                })
            box_start = box_end

    results = {}
    for threshold in thresholds:
        if isinstance(threshold, dict):
            print(f"\n  threshold={json.dumps(threshold)} ... ", end='', flush=True)
        else:
            print(f"\n  threshold={threshold:.2f} ... ", end='', flush=True)

        if isinstance(threshold, dict):
            filtered = [d for d in all_detections
                        if d['score'] >= threshold.get(str(d['category_id']), 0.2)]
        else:
            filtered = [d for d in all_detections if d['score'] >= threshold]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(filtered, f)
            dt_path = f.name

        mean_f1, per_class = compute_mean_f1(val_json, dt_path)
        key = json.dumps(threshold) if isinstance(threshold, dict) else threshold
        results[key] = {'mean_f1': mean_f1, 'per_class': per_class}
        os.unlink(dt_path)
        print(f"Mean F1={mean_f1:.4f}  (battery={per_class[1]:.4f} board={per_class[2]:.4f} fire={per_class[3]:.4f})")

    return results


def run_per_class_sweep(predictor, val_json, image_dir):
    """Grid search per-class thresholds."""
    t_battery = [0.05, 0.08, 0.10, 0.12, 0.15]
    t_board = [0.10, 0.15, 0.20, 0.25, 0.30]
    t_fire = [0.10, 0.15, 0.20, 0.25, 0.30]

    best_mean = 0.0
    best_thresholds = None
    best_per_class = None

    total = len(t_battery) * len(t_board) * len(t_fire)
    count = 0

    for tb in t_battery:
        for tbo in t_board:
            for tf in t_fire:
                thresholds = {"1": tb, "2": tbo, "3": tf}
                key = f"battery={tb} board={tbo} fire={tf}"
                print(f"  [{count+1}/{total}] {key} ... ", end='', flush=True)

                results = run_sweep(predictor, val_json, image_dir, [thresholds])
                r = results[json.dumps(thresholds)]

                if r['mean_f1'] > best_mean:
                    best_mean = r['mean_f1']
                    best_thresholds = thresholds
                    best_per_class = r['per_class']

                count += 1

    return best_thresholds, best_mean, best_per_class


def main():
    paddle.enable_static()
    model_dir = os.path.join(SCRIPT_DIR, "model")
    val_json = os.path.join(SCRIPT_DIR, "A_train", "val.json")
    image_dir = os.path.join(SCRIPT_DIR, "A_train", "Image")

    if not os.path.exists(os.path.join(model_dir, 'model.pdmodel')):
        print("ERROR: model/ not found. Run export_model.py first.")
        sys.exit(1)

    print("Loading predictor...")
    predictor = SweepPredictor(model_dir)

    # --- Global threshold sweep ---
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n=== Global Threshold Sweep ({len(thresholds)} values) ===")
    results = run_sweep(predictor, val_json, image_dir, thresholds)

    best_t = max(results, key=lambda t: results[t]['mean_f1'])
    print(f"\nBest global threshold: {best_t:.2f} (Mean F1={results[best_t]['mean_f1']:.4f})")
    print(f"  battery: {results[best_t]['per_class'][1]:.4f}")
    print(f"  board:   {results[best_t]['per_class'][2]:.4f}")
    print(f"  fire:    {results[best_t]['per_class'][3]:.4f}")

    # --- Per-class threshold sweep ---
    print(f"\n=== Per-Class Threshold Grid Search (125 combinations) ===")
    best_pc_t, best_pc_mean, best_pc_f1 = run_per_class_sweep(predictor, val_json, image_dir)
    print(f"\nBest per-class thresholds: {best_pc_t}")
    print(f"Mean F1={best_pc_mean:.4f}")
    print(f"  battery: {best_pc_f1[1]:.4f}")
    print(f"  board:   {best_pc_f1[2]:.4f}")
    print(f"  fire:    {best_pc_f1[3]:.4f}")

    print(f"\npredict.py usage:")
    print(f"  python predict.py data.txt result.json {best_t:.2f}")
    print(f'  python predict.py data.txt result.json \'{json.dumps(best_pc_t)}\'')


if __name__ == '__main__':
    main()
