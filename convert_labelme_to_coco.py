# -*- coding: utf-8 -*-
"""
Convert LabelMe JSON annotations to COCO format for PaddleDetection training.
Splits data into train (80%) and val (20%) sets.
"""
import json
import os
import random
from glob import glob
from collections import defaultdict

random.seed(42)

# Paths
LABEL_DIR = "A_train/label"
IMAGE_DIR = "A_train/Image"
OUTPUT_DIR = "A_train"

# Category mapping
CATEGORIES = [
    {"id": 1, "name": "battery"},
    {"id": 2, "name": "board"},
    {"id": 3, "name": "fire"},
]
CAT_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def labelme_to_coco(image_ids, label_dir, image_dir, output_path):
    """Convert a subset of LabelMe annotations to COCO format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }
    ann_id = 1

    for idx, img_id in enumerate(image_ids):
        json_path = os.path.join(label_dir, f"{img_id}.json")
        img_path = os.path.join(image_dir, f"{img_id}.jpg")

        with open(json_path, "r") as f:
            label_data = json.load(f)

        img_w = label_data.get("imageWidth", 1920)
        img_h = label_data.get("imageHeight", 1080)

        coco["images"].append({
            "id": idx + 1,
            "file_name": f"{img_id}.jpg",
            "width": img_w,
            "height": img_h,
        })

        for shape in label_data.get("shapes", []):
            label_name = shape.get("label", "")
            if label_name not in CAT_NAME_TO_ID:
                continue

            points = shape.get("points", [])
            if len(points) < 4:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y

            if w <= 0 or h <= 0:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": idx + 1,
                "category_id": CAT_NAME_TO_ID[label_name],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=None)

    return len(coco["images"]), len(coco["annotations"])


def main():
    # Collect all image IDs (without extension)
    label_files = glob(os.path.join(LABEL_DIR, "*.json"))
    all_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in label_files])
    print(f"Total images: {len(all_ids)}")

    # Count total annotations per class
    class_counts = defaultdict(int)
    for img_id in all_ids:
        with open(os.path.join(LABEL_DIR, f"{img_id}.json"), "r") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            if shape["label"] in CAT_NAME_TO_ID:
                class_counts[shape["label"]] += 1
    print(f"Class distribution: {dict(class_counts)}")

    # Shuffle and split: 80% train, 20% val
    random.shuffle(all_ids)
    split = int(len(all_ids) * 0.8)
    train_ids = sorted(all_ids[:split])
    val_ids = sorted(all_ids[split:])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Convert to COCO
    n_train_imgs, n_train_anns = labelme_to_coco(
        train_ids, LABEL_DIR, IMAGE_DIR,
        os.path.join(OUTPUT_DIR, "train.json")
    )
    n_val_imgs, n_val_anns = labelme_to_coco(
        val_ids, LABEL_DIR, IMAGE_DIR,
        os.path.join(OUTPUT_DIR, "val.json")
    )

    print(f"Train: {n_train_imgs} images, {n_train_anns} annotations")
    print(f"Val:   {n_val_imgs} images, {n_val_anns} annotations")
    print("COCO JSON files written to A_train/")


if __name__ == "__main__":
    main()
