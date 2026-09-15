"""
DyMATES - Hugging Face Large Dataset Importer
Downloads keremberke/helmet-detection (5,000+ images) and converts to YOLO v11 format.
"""

import os
import shutil
from datasets import load_dataset
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_LARGE = os.path.join(BASE_DIR, "dataset_large")

print("=" * 75)
print("  DyMATES — HUGGING FACE DATASET IMPORTER (keremberke/helmet-detection)")
print("=" * 75)

os.makedirs(os.path.join(DATASET_LARGE, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_LARGE, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(DATASET_LARGE, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_LARGE, "val", "labels"), exist_ok=True)
os.makedirs(os.path.join(DATASET_LARGE, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_LARGE, "test", "labels"), exist_ok=True)


def convert_split(hf_data, split_name):
    print(f"\nProcessing split '{split_name}' with {len(hf_data)} samples...")
    target_split = split_name if split_name in ["train", "val", "test"] else "train"
    if target_split == "validation":
        target_split = "val"

    img_dir = os.path.join(DATASET_LARGE, target_split, "images")
    lbl_dir = os.path.join(DATASET_LARGE, target_split, "labels")

    saved_count = 0
    for idx, item in enumerate(hf_data):
        img = item["image"]
        width, height = img.size

        file_id = f"hf_{split_name}_{idx:05d}"
        img_filename = f"{file_id}.jpg"
        lbl_filename = f"{file_id}.txt"

        img_path = os.path.join(img_dir, img_filename)
        lbl_path = os.path.join(lbl_dir, lbl_filename)

        # Save image
        img.convert("RGB").save(img_path, "JPEG", quality=90)

        # Convert COCO bbox [x_min, y_min, w, h] → YOLO [class_id, x_center, y_center, w, h]
        objects = item.get("objects", {})
        bboxes = objects.get("bbox", [])
        categories = objects.get("categories", [])

        yolo_lines = []
        for bbox, cat in zip(bboxes, categories):
            x_min, y_min, box_w, box_h = bbox
            x_center = (x_min + box_w / 2.0) / width
            y_center = (y_min + box_h / 2.0) / height
            norm_w = box_w / width
            norm_h = box_h / height

            # Map category ID to 0: helmet, 1: no-helmet
            # In keremberke/helmet-detection: 0 is helmet, 1 is head/no-helmet
            cls_id = 0 if cat == 0 else 1
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        saved_count += 1
        if (idx + 1) % 500 == 0 or (idx + 1) == len(hf_data):
            print(f"  → Saved {idx + 1}/{len(hf_data)} images in '{target_split}' split")

    return saved_count


def main():
    print("[1/2] Loading dataset 'keremberke/helmet-detection' from Hugging Face...")
    ds = load_dataset("keremberke/helmet-detection", "full")
    print(f"  ✔ Dataset loaded successfully! Splits: {list(ds.keys())}")

    total_converted = 0
    for split in ds.keys():
        total_converted += convert_split(ds[split], split)

    # Generate dataset_large.yaml
    yaml_content = f"""path: {DATASET_LARGE}
train: train/images
val: val/images
test: test/images

names:
  0: helmet
  1: no-helmet
  2: motorcycle
  3: license-plate
"""
    yaml_path = os.path.join(BASE_DIR, "dataset_large.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 75)
    print("  EXPANDED DATASET CONVERSION COMPLETE!")
    print(f"  • Total Images Processed: {total_converted}")
    print(f"  • YAML Config Generated:  {yaml_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
