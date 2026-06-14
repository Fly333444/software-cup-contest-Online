# -*- coding: utf-8 -*-
"""
3-class object detection inference (battery / board / fire)
Called by evaluation system:
    python predict.py <data_txt> <result_json> [threshold]

v15: PicoDet-L (LCNet 2.0x) inference
"""
import os
import sys
import time
import json
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PaddleDetection'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PaddleDetection', 'deploy', 'python'))

import numpy as np
import paddle
from paddle.inference import Config, create_predictor

from preprocess import preprocess, Resize, NormalizeImage, Permute


class PredictConfig:
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf.get('arch', 'PicoDet')
        self.preprocess_infos = yml_conf.get('Preprocess', [])
        self.labels = yml_conf.get('label_list', [])
        self.mask = yml_conf.get('mask', False)
        self.use_dynamic_shape = yml_conf.get('use_dynamic_shape', False)
        self.tracker = yml_conf.get('tracker', None)
        self.nms = yml_conf.get('NMS', None)
        self.fpn_stride = yml_conf.get('fpn_stride', None)
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))


def get_test_images(infer_file):
    """Read image paths from data.txt. Supports relative and absolute paths."""
    infer_dir = os.path.dirname(os.path.abspath(infer_file))
    with open(infer_file, 'r') as f:
        lines = f.readlines()
    images = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = line.replace('\\', '/')
        if not os.path.isabs(line):
            line = os.path.join(infer_dir, line)
        images.append(line)
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images


def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )
    config.enable_use_gpu(500, 0)
    config.switch_ir_optim(False)  # Disable graph optim to avoid fused_conv2d_add_act on RTX 4060
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    # Disable TensorRT locally (RTX 4060 CC 8.9 cuDNN incompatibility with fused ops)
    # Will enable on V100 for submission

    predictor = create_predictor(config)
    return predictor, config


def create_inputs(imgs, im_info, input_names):
    """Stack identically-sized images directly (keep_ratio=False → all same size)."""
    inputs = {}

    if 'image' in input_names:
        im_shape = []
        scale_factor = []
        for e in im_info:
            im_shape.append(np.array((e['im_shape'],)).astype('float32'))
            scale_factor.append(np.array((e['scale_factor'],)).astype('float32'))
        inputs['image'] = np.stack([np.array(img, dtype=np.float32) for img in imgs], axis=0)

    if 'im_shape' in input_names:
        inputs['im_shape'] = np.stack(im_shape, axis=0)

    if 'scale_factor' in input_names:
        inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    return inputs


class Detector(object):
    def __init__(self, pred_config, model_dir):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        print('Model inputs:', self.input_names)
        print('Model outputs:', self.output_names)
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def predict_batch(self, imgs, im_info):
        inputs = create_inputs(imgs, im_info, self.input_names)
        for name in self.input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        self.predictor.run()

        # PicoDet output: [bboxes, bbox_num, ...]
        num_outs = int(len(self.output_names) / 2)
        np_boxes = self.predictor.get_output_handle(
            self.output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(
            self.output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)


def predict_image(detector, image_list, result_path, threshold, batch_size=64):
    c_results = {"result": []}

    for start in range(0, len(image_list), batch_size):
        batch_paths = image_list[start:start + batch_size]
        input_im_lst = []
        input_im_info_lst = []
        image_ids = []
        for im_path in batch_paths:
            im, im_info = preprocess(im_path, detector.preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
            image_ids.append(os.path.splitext(os.path.basename(im_path))[0])

        det_results = detector.predict_batch(input_im_lst, input_im_info_lst)
        boxes = det_results['boxes']
        boxes_num = det_results['boxes_num'].astype('int32').flatten()

        box_start = 0
        for image_id, n in zip(image_ids, boxes_num):
            box_end = box_start + int(n)
            im_boxes = boxes[box_start:box_end]
            box_start = box_end

            for det in im_boxes:
                score = float(det[1])
                cls_id = int(det[0]) + 1

                # per-class threshold (v6.0 feature) or global
                if isinstance(threshold, dict):
                    t = threshold.get(str(cls_id), threshold.get(cls_id, 0.2))
                else:
                    t = threshold
                if score < t:
                    continue

                x1, y1, x2, y2 = map(float, det[2:6])
                c_results["result"].append({
                    "image_id": image_id,
                    "type": cls_id,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "segmentation": []
                })

    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)
    print("Results written to", result_path)
    print("Detections:", len(c_results["result"]))


def main(infer_txt, result_path, det_model_path, threshold):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, threshold)


if __name__ == '__main__':
    start_time = time.time()
    paddle.enable_static()

    det_model_path = os.path.join(SCRIPT_DIR, "model")
    threshold = 0.30  # v2.2 默认, 训练后可 sweep 覆盖

    if len(sys.argv) > 3:
        arg = sys.argv[3]
        if arg.startswith('{'):
            threshold = json.loads(arg)
        else:
            threshold = float(arg)

    infer_txt = sys.argv[1]
    result_path = sys.argv[2]

    main(infer_txt, result_path, det_model_path, threshold)
    total_time = time.time() - start_time
    print('Total time: {:.2f}s'.format(total_time))
