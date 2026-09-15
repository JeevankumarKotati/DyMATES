"""
DyMATES - Unified 4-Class Dataset Preparation Engine
Structures images & annotations into 4 aligned classes:
  0: with_helmet
  1: without_helmet
  2: motorcycle
  3: license_plate
"""

import os
import glob
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_MC = os.path.join(BASE_DIR, "dataset_multiclass")

print("=" * 75)
print("  DyMATES — UNIFIED 4-CLASS DATASET PREPARATION ENGINE")
print("=" * 75)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATASET_MC, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_MC, split, "labels"), exist_ok=True)

# 1. Collect all images and label files from dataset, dataset_prepared, dataset_large
image_label_pairs = []

search_sources = [
    os.path.join(BASE_DIR, "dataset_prepared"),
    os.path.join(BASE_DIR, "dataset"),
    os.path.join(BASE_DIR, "dataset_large")
]

for src in search_sources:
    if not os.path.exists(src):
        continue
    for root, dirs, files in os.walk(src):
        if "images" in root:
            parent_lbl = os.path.join(os.path.dirname(root), "labels")
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_p = os.path.join(root, f)
                    bname, _ = os.path.splitext(f)
                    lbl_p = os.path.join(parent_lbl, f"{bname}.txt")
                    if not os.path.exists(lbl_p):
                        lbl_p = os.path.join(root, f"{bname}.txt")
                    if os.path.exists(lbl_p):
                        image_label_pairs.append((img_p, lbl_p))

print(f"[1/3] Total matched image-label pairs discovered: {len(image_label_pairs)}")

# Deduplicate by filename
unique_pairs = {}
for img_p, lbl_p in image_label_pairs:
    fname = os.path.basename(img_p)
    if fname not in unique_pairs:
        unique_pairs[fname] = (img_p, lbl_p)

pairs_list = list(unique_pairs.values())
random.seed(42)
random.shuffle(pairs_list)

print(f"[2/3] Unique matched dataset pairs: {len(pairs_list)}")

n_total = len(pairs_list)
n_train = int(0.80 * n_total)
n_val = int(0.10 * n_total)

splits_data = {
    "train": pairs_list[:n_train],
    "val": pairs_list[n_train:n_train + n_val],
    "test": pairs_list[n_train + n_val:]
}

print(f"[3/3] Copying and re-indexing class IDs into {DATASET_MC}...")

class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
split_counts = {}

for split_name, plist in splits_data.items():
    copied = 0
    img_dst_dir = os.path.join(DATASET_MC, split_name, "images")
    lbl_dst_dir = os.path.join(DATASET_MC, split_name, "labels")

    for img_p, lbl_p in plist:
        fname = os.path.basename(img_p)
        bname, _ = os.path.splitext(fname)

        shutil.copy2(img_p, os.path.join(img_dst_dir, fname))

        # Parse and re-index labels
        new_lines = []
        with open(lbl_p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        raw_cid = int(parts[0])
                        # Map raw_cid to 0..3 range
                        cid = raw_cid if raw_cid in [0, 1, 2, 3] else 0
                        xc, yc, w, h = parts[1:]
                        new_lines.append(f"{cid} {xc} {yc} {w} {h}")
                        class_counts[cid] = class_counts.get(cid, 0) + 1
                    except ValueError:
                        continue

        with open(os.path.join(lbl_dst_dir, f"{bname}.txt"), "w") as f:
            f.write("\n".join(new_lines) + "\n")

        copied += 1
    split_counts[split_name] = copied

yaml_content = f"""path: {DATASET_MC}
train: train/images
val: val/images
test: test/images

names:
  0: with_helmet
  1: without_helmet
  2: motorcycle
  3: license_plate
"""
yaml_path = os.path.join(BASE_DIR, "dataset_multiclass.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("\n" + "=" * 75)
print("  UNIFIED 4-CLASS DATASET PREPARATION COMPLETE!")
print(f"  • Train Pairs: {split_counts.get('train', 0)}")
print(f"  • Val Pairs:   {split_counts.get('val', 0)}")
print(f"  • Test Pairs:  {split_counts.get('test', 0)}")
print(f"  • Class BBox Counts: {class_counts}")
print(f"  • YAML Config: {yaml_path}")
print("=" * 75 + "\n")
