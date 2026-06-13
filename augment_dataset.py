# -*- coding: utf-8 -*-
"""Board-only Copy-Paste augmentation for v12 PP-YOLOE+_l training.

Board class has only 72 annotations (vs 580 fire, 96 battery).
This script extracts board objects and pastes them onto fire-only images
to create augmented training samples with balanced class distribution.

Outputs A_train/train_augmented.json (original + augmented annotations)
"""
import os
import sys
import json
import random
import copy
import numpy as np
import cv2
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'A_train', 'Image')
TRAIN_JSON = os.path.join(BASE_DIR, 'A_train', 'train.json')
OUTPUT_JSON = os.path.join(BASE_DIR, 'A_train', 'train_augmented.json')

random.seed(42)
np.random.seed(42)


def compute_iou(box1, box2):
    """IoU in COCO xywh format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return inter / (area1 + area2 - inter + 1e-8)


def paste_board(bg_img, board_img, board_bbox, existing_boxes):
    """Paste a board object onto background. Returns (result_img, new_bbox) or (img, None) on failure."""
    bh, bw = board_img.shape[:2]

    # Random scale 0.7~1.3
    scale = random.uniform(0.7, 1.3)
    nw, nh = max(8, int(bw * scale)), max(8, int(bh * scale))
    board_resized = cv2.resize(board_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Random rotation [-20, 20]
    angle = random.uniform(-20, 20)
    if abs(angle) > 1:
        center = (nw // 2, nh // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        rw, rh = int(nh * sin_a + nw * cos_a), int(nh * cos_a + nw * sin_a)
        M[0, 2] += rw // 2 - center[0]
        M[1, 2] += rh // 2 - center[1]
        board_rot = cv2.warpAffine(board_resized, M, (rw, rh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
    else:
        board_rot = board_resized
        rw, rh = nw, nh

    # Random brightness 0.8~1.3
    brightness = random.uniform(0.8, 1.3)
    board_final = np.clip(board_rot * brightness, 0, 255).astype(np.uint8)

    bg_h, bg_w = bg_img.shape[:2]
    if rw > bg_w or rh > bg_h:
        return bg_img, None

    best_x, best_y, best_overlap = 0, 0, 1.0
    for _ in range(50):
        px = random.randint(0, bg_w - rw)
        py = random.randint(0, bg_h - rh)
        new_box = [px, py, rw, rh]
        max_ov = max([compute_iou(new_box, eb) for eb in existing_boxes]) if existing_boxes else 0
        if max_ov == 0:
            best_x, best_y = px, py
            break
        if max_ov < best_overlap:
            best_x, best_y, best_overlap = px, py, max_ov
            if best_overlap < 0.15:
                break
    else:
        if best_overlap >= 0.5:
            return bg_img, None

    # Paste with mask-based blending
    paste_h = min(rh, bg_h - best_y)
    paste_w = min(rw, bg_w - best_x)
    board_final = board_final[:paste_h, :paste_w]
    mask = (board_final.sum(axis=2) > 10).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[:, :, None]

    roi = bg_img[best_y:best_y + paste_h, best_x:best_x + paste_w].astype(np.float32)
    bg_img[best_y:best_y + paste_h, best_x:best_x + paste_w] = \
        (board_final.astype(np.float32) * mask + roi * (1 - mask)).astype(np.uint8)

    return bg_img, [best_x, best_y, paste_w, paste_h]


def main():
    with open(TRAIN_JSON) as f:
        coco = json.load(f)

    img_map = {img['id']: img for img in coco['images']}
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image
    img_anns = {}
    for ann in coco['annotations']:
        img_anns.setdefault(ann['image_id'], []).append(ann)

    # Identify images by class
    board_imgs = set()
    fire_imgs = set()
    battery_imgs = set()
    for img_id, anns in img_anns.items():
        cats = set(a['category_id'] for a in anns)
        if 2 in cats:
            board_imgs.add(img_id)
        if 3 in cats:
            fire_imgs.add(img_id)
        if 1 in cats:
            battery_imgs.add(img_id)

    print(f"Board images: {len(board_imgs)}, Fire images: {len(fire_imgs)}, Battery: {len(battery_imgs)}")

    # Extract board objects
    board_objects = []
    for ann in coco['annotations']:
        if ann['category_id'] == 2:
            img_info = img_map[ann['image_id']]
            img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
            board_objects.append({
                'img_path': img_path,
                'bbox': ann['bbox'],
                'image_id': ann['image_id'],
            })

    print(f"Total board annotations: {len(board_objects)}")

    # Fire-only images (no board, no battery) as backgrounds
    fire_only = [iid for iid in fire_imgs if iid not in board_imgs and iid not in battery_imgs]
    print(f"Fire-only images (background candidates): {len(fire_only)}")

    # Cache all images
    img_cache = {}
    for img_id, img_info in img_map.items():
        img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
        if os.path.exists(img_path):
            img_cache[img_id] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Crop all board objects
    cropped_boards = []
    for bo in board_objects:
        img = img_cache[bo['image_id']]
        x, y, w, h = map(int, bo['bbox'])
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(img.shape[1], x + w + 2), min(img.shape[0], y + h + 2)
        crop = img[y1:y2, x1:x2].copy()
        if crop.shape[0] > 5 and crop.shape[1] > 5:
            cropped_boards.append(crop)

    print(f"Cropped board objects: {len(cropped_boards)}")

    if len(cropped_boards) == 0 or len(fire_only) == 0:
        print("ERROR: Not enough board crops or backgrounds. Aborting.")
        return

    # Generate augmented annotations by pasting board onto fire-only backgrounds
    new_images = list(coco['images'])
    new_annotations = list(coco['annotations'])
    max_img_id = max(img['id'] for img in coco['images'])
    max_ann_id = max(ann['id'] for ann in coco['annotations'])

    generated = 0
    board_pasted = 0

    # Augment each board-containing image: create 3 variants with extra pasted boards
    for img_id in board_imgs:
        for aug_idx in range(3):
            # Pick a random fire-only background
            bg_id = random.choice(fire_only)
            bg = img_cache[bg_id].copy()

            # Get existing annotations on this background
            existing_boxes = [a['bbox'] for a in img_anns.get(bg_id, [])]

            # Paste 1-2 board objects
            pasted_boxes = []
            for _ in range(random.randint(1, 2)):
                board_crop = random.choice(cropped_boards)
                bg, new_box = paste_board(bg, board_crop, None, existing_boxes + pasted_boxes)
                if new_box:
                    pasted_boxes.append(new_box)

            if not pasted_boxes:
                continue

            # Save augmented image
            max_img_id += 1
            aug_filename = f"aug_v12_{max_img_id:06d}.jpg"
            aug_path = os.path.join(IMAGE_DIR, aug_filename)
            cv2.imwrite(aug_path, cv2.cvtColor(bg, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            new_images.append({
                'id': max_img_id,
                'file_name': aug_filename,
                'width': bg.shape[1],
                'height': bg.shape[0],
            })

            for pb in pasted_boxes:
                max_ann_id += 1
                new_annotations.append({
                    'id': max_ann_id,
                    'image_id': max_img_id,
                    'category_id': 2,  # board
                    'bbox': pb,
                    'area': pb[2] * pb[3],
                    'iscrowd': 0,
                })
                board_pasted += 1

            generated += 1

    print(f"Generated {generated} augmented images with {board_pasted} board annotations")

    # Also create battery-only augmented images (less critical, but helps)
    battery_crops = []
    for ann in coco['annotations']:
        if ann['category_id'] == 1:
            img_info = img_map[ann['image_id']]
            img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
            if os.path.exists(img_path):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                x, y, w, h = map(int, ann['bbox'])
                x1, y1 = max(0, x - 2), max(0, y - 2)
                x2, y2 = min(img.shape[1], x + w + 2), min(img.shape[0], y + h + 2)
                crop = img[y1:y2, x1:x2].copy()
                if crop.shape[0] > 5 and crop.shape[1] > 5:
                    battery_crops.append(crop)

    print(f"Cropped battery objects: {len(battery_crops)}")

    if len(battery_crops) > 0:
        battery_pasted = 0
        for img_id in board_imgs | battery_imgs:
            for aug_idx in range(1):  # 1x battery copy-paste
                bg_id = random.choice(fire_only)
                bg = img_cache[bg_id].copy()
                existing_boxes = [a['bbox'] for a in img_anns.get(bg_id, [])]

                # Paste 1 battery
                battery_crop = random.choice(battery_crops)
                bg, new_box = paste_board(bg, battery_crop, None, existing_boxes)
                if new_box is None:
                    continue

                max_img_id += 1
                aug_filename = f"aug_v12_{max_img_id:06d}.jpg"
                cv2.imwrite(os.path.join(IMAGE_DIR, aug_filename),
                            cv2.cvtColor(bg, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

                new_images.append({
                    'id': max_img_id,
                    'file_name': aug_filename,
                    'width': bg.shape[1],
                    'height': bg.shape[0],
                })

                max_ann_id += 1
                new_annotations.append({
                    'id': max_ann_id,
                    'image_id': max_img_id,
                    'category_id': 1,
                    'bbox': new_box,
                    'area': new_box[2] * new_box[3],
                    'iscrowd': 0,
                })
                battery_pasted += 1
                generated += 1

        print(f"Battery pasted: {battery_pasted}")

    # Final stats
    cat_counts = Counter()
    for ann in new_annotations:
        cat_counts[ann['category_id']] += 1

    print(f"\nFinal dataset: {len(new_images)} images, {len(new_annotations)} annotations")
    for cid, name in sorted(cat_map.items()):
        print(f"  {name}: {cat_counts.get(cid, 0)}")

    # Write output
    output = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': copy.deepcopy(coco['categories']),
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f)
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
