# -*- coding: utf-8 -*-
"""
Generate ImageSets for full training (all data → train, no val split).
Scans B_data/annotations/*.xml, generates ImageSets/Main/train.txt (all images).

Usage:
    python split_dataset.py
"""
import os
import glob
from collections import defaultdict
import defusedxml.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANNO_DIR = os.path.join(SCRIPT_DIR, "B_data", "annotations")
IMGSET_DIR = os.path.join(SCRIPT_DIR, "B_data", "ImageSets", "Main")


def parse_xml_boxes(xml_path):
    """Parse VOC XML and return list of (class_name, xmin, ymin, xmax, ymax)."""
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def main():
    # Collect all file IDs from XML annotations
    xml_files = sorted(glob.glob(os.path.join(ANNO_DIR, "*.xml")))
    all_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in xml_files])
    print(f"Total images with annotations: {len(all_ids)}")

    # Statistics
    class_counts = defaultdict(int)
    fire_count = 0
    firebig_count = 0
    for xml_file in xml_files:
        boxes = parse_xml_boxes(xml_file)
        for name, _, _, _, _ in boxes:
            class_counts[name] += 1
            if name == 'fire':
                fire_count += 1
            elif name == 'firebig':
                firebig_count += 1

    print(f"Class distribution: {dict(class_counts)}")
    print(f"  fire: {fire_count}, firebig: {firebig_count}")
    print(f"  firebig ratio: {firebig_count}/{len(all_ids)} images")

    # 全量训练: 所有图片 → train.txt
    os.makedirs(IMGSET_DIR, exist_ok=True)

    # 需要生成 PaddleDetection VOCDataSet 格式: 每行 "图片路径 标注路径"
    with open(os.path.join(IMGSET_DIR, "train.txt"), "w") as f:
        for fid in all_ids:
            f.write(f"{fid}.jpg annotations/{fid}.xml\n")

    # Also create label_list.txt
    label_list_path = os.path.join(SCRIPT_DIR, "B_data", "label_list.txt")
    with open(label_list_path, "w") as f:
        f.write("fire\nfirebig\n")

    print(f"Train: {len(all_ids)} images (full)")
    print(f"  fire boxes: {fire_count}")
    print(f"  firebig boxes: {firebig_count}")
    print(f"\nWrote: {IMGSET_DIR}/train.txt")
    print(f"Wrote: B_data/label_list.txt")
    print("Done.")


if __name__ == "__main__":
    main()
