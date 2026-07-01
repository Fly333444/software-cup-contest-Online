# -*- coding: utf-8 -*-
"""
# B榜 single-class firebig inference (Faster R-CNN R50 FPN)
核心: RPN先提候选区域→再分类, 640×480原生输入不变形, 每图输出1个firebig框

Called by evaluation system:
    python predict.py <data_txt> <result_json> [threshold]

Self-contained — no PaddleDetection dependency needed.
"""
import os
import sys
import time
import json
import yaml

import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===== Self-contained preprocess (avoids imgaug/keypoint_preprocess imports) =====

def decode_image(im_file, im_info):
    """Read RGB image from file path."""
    with open(im_file, 'rb') as f:
        im_read = f.read()
    data = np.frombuffer(im_read, dtype='uint8')
    im = cv2.imdecode(data, 1)
    if im is None:
        raise ValueError(f"Failed to decode image: {im_file}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


class Resize(object):
    def __init__(self, target_size, keep_ratio=False, interp=2):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        origin_shape = im.shape[:2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_y = im_scale
            im_scale_x = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array([im_scale_y, im_scale_x]).astype('float32')
        return im, im_info


class NormalizeImage(object):
    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            im *= 1.0 / 255.0
        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class Permute(object):
    def __call__(self, im, im_info):
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info


def preprocess(im, preprocess_ops):
    im_info = {
        'scale_factor': np.array([1., 1.], dtype=np.float32),
        'im_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info


class PredictConfig:
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf.get('arch', 'PPYOLOE')
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
    try:
        config.enable_use_gpu(500, 0)
    except Exception:
        config.disable_gpu()  # 服务端无GPU时回退CPU推理
    config.switch_ir_optim(False)  # RTX 4060 CUDNN compatibility
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)

    predictor = create_predictor(config)
    return predictor, config


def create_inputs(imgs, im_info, input_names):
    """Stack and pad images for batch inference."""
    inputs = {}
    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(e['im_shape'].astype('float32'))
        scale_factor.append(e['scale_factor'].astype('float32'))

    # Pad to same size for stacking
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = []
    for img in imgs:
        c, h, w = img.shape
        pad = np.zeros((c, max_h, max_w), dtype=np.float32)
        pad[:, :h, :w] = img
        padded.append(pad)

    inputs['image'] = np.stack(padded, axis=0)
    if 'im_shape' in input_names:
        inputs['im_shape'] = np.stack(im_shape, axis=0)
    if 'scale_factor' in input_names:
        inputs['scale_factor'] = np.stack(scale_factor, axis=0)

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

        # PP-YOLOE output: [bboxes, bbox_num, ...]
        num_outs = int(len(self.output_names) / 2)
        np_boxes = self.predictor.get_output_handle(
            self.output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(
            self.output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)


def predict_image(detector, image_list, result_path, threshold, batch_size=64):
    """
    B榜推理核心:
    1. 模型输出所有检测框（模型内部已完成NMS+坐标映射到原图空间）
    2. 过滤: score >= threshold
    3. 按 image_id 分组, 选置信度最高的框
    4. type 恒为 1 (firebig)
    5. 无框的图不输出条目
    """
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

            # Collect all qualified detections for this image
            candidates = []
            for det in im_boxes:
                score = float(det[1])
                cls_id = int(det[0])  # 0-indexed, firebig=0 → type=1

                if isinstance(threshold, dict):
                    t = threshold.get(str(cls_id), threshold.get(cls_id, 0.025))
                else:
                    t = threshold
                if score < t:
                    continue

                x1, y1, x2, y2 = map(float, det[2:6])
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                area = w * h
                candidates.append({
                    "image_id": image_id,
                    "type": 1,  # B榜只输出 firebig (type=1)
                    "x": x1, "y": y1,
                    "width": w, "height": h,
                    "segmentation": [],
                    "area": area,
                    "score": score,
                })

            # 无检出框 → 该图片不输出条目（符合B榜规范）
            if not candidates:
                continue

            # 算法核心: 选最高分框（实验证实 max_score IoU>=0.5=96%, max_area=3%）
            # 如果有多个高分框分数接近（差距<10%）, 从中选面积最大的
            candidates.sort(key=lambda c: c["score"], reverse=True)
            top_score = candidates[0]["score"]

            # 分数极低的最高分框 → 视为不可靠检测，跳过该图
            if top_score < 0.15:
                continue

            close_candidates = [c for c in candidates if c["score"] >= top_score * 0.9]
            best = max(close_candidates, key=lambda c: c["area"])

            # 坐标安全截断到图像范围内
            best["x"] = max(0.0, best["x"])
            best["y"] = max(0.0, best["y"])
            best["width"] = min(best["width"], 640.0 - best["x"])
            best["height"] = min(best["height"], 480.0 - best["y"])

            del best["area"], best["score"]
            c_results["result"].append(best)

    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)
    print("Results written to", result_path)
    print("Images with detections:", len(c_results["result"]),
          "/", len(image_list))


def main(infer_txt, result_path, det_model_path, threshold):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, threshold)


if __name__ == '__main__':
    start_time = time.time()
    paddle.enable_static()

    det_model_path = os.path.join(SCRIPT_DIR, "model")
    threshold = 0.3  # 置信度阈值：滤除低分误检（调低 NMS 阈值是为了不过早丢弃，此处二次精选）

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
