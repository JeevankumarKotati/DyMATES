"""
DyMATES - Unzip and Structure Large Safety Helmet Dataset
Extracts downloaded HuggingFace dataset archives into dataset_large/
"""

import os
import glob
import zipfile
import shutil
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_LARGE = os.path.join(BASE_DIR, "dataset_large")

print("=" * 75)
print("  DyMATES — UNZIP AND PREPARE LARGE DATASET SUITE")
print("=" * 75)

# Find downloaded HF dataset zip
hf_zip = os.path.expanduser("~/.cache/huggingface/hub/datasets--harijawahar--Helmet_Detection/snapshots/c94aa836f7ad5f5a417f3d53d062793afff0fcf4/helmet detection dataset.v10i.coco (3).zip")
pzalavad_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--pzalavad--HelmetDataset/snapshots/c8647293d538fe3aeda3539a3f4274d1b5ea6762/data")

temp_extract = os.path.join(BASE_DIR, "temp_hf_extract")
os.makedirs(temp_extract, exist_ok=True)

if os.path.exists(hf_zip):
    print(f"[1/3] Extracting {hf_zip}...")
    with zipfile.ZipFile(hf_zip, "r") as z:
        z.extractall(temp_extract)
    print("  ✔ COCO zip extracted.")

print(f"[2/3] Structuring YOLO splits inside {DATASET_LARGE}...")

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATASET_LARGE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_LARGE, split, "labels"), exist_ok=True)

# Parse COCO JSON files if present in extracted folder
def convert_coco_json(json_path, img_dir, target_split):
    if not os.path.exists(json_path):
        return 0
    with open(json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    
    # Map categories: 'helmet' or 'with helmet' -> 0, others -> 1
    cat_map = {}
    for cid, cname in categories.items():
        cname_lower = cname.lower()
        if "helmet" in cname_lower and "no" not in cname_lower and "without" not in cname_lower:
            cat_map[cid] = 0
        else:
            cat_map[cid] = 1

    img_annotations = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    count = 0
    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]
        
        src_img = os.path.join(img_dir, fname)
        if not os.path.exists(src_img):
            continue

        bname, _ = os.path.splitext(fname)
        dst_img = os.path.join(DATASET_LARGE, target_split, "images", fname)
        shutil.copy2(src_img, dst_img)

        # Convert annotations to YOLO txt
        anns = img_annotations.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            bbox = ann["bbox"] # [x, y, w, h]
            cid = ann["category_id"]
            yolo_cls = cat_map.get(cid, 0)
            
            x, y, w, h = bbox
            x_center = (x + w / 2.0) / width
            y_center = (y + h / 2.0) / height
            norm_w = w / width
            norm_h = h / height
            
            yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        dst_lbl = os.path.join(DATASET_LARGE, target_split, "labels", f"{bname}.txt")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        count += 1

    return count

c_tr = convert_coco_json(os.path.join(temp_extract, "train", "_annotations.coco.json"), os.path.join(temp_extract, "train"), "train")
c_va = convert_coco_json(os.path.join(temp_extract, "valid", "_annotations.coco.json"), os.path.join(temp_extract, "valid"), "val")
c_te = convert_coco_json(os.path.join(temp_extract, "test", "_annotations.coco.json"), os.path.join(temp_extract, "test"), "test")

print(f"  → COCO Dataset converted: Train={c_tr}, Val={c_va}, Test={c_te}")

# Merge workspace images into dataset_large
print("[3/3] Merging workspace dataset_prepared and dataset images...")
ws_imgs = glob.glob(os.path.join(BASE_DIR, "dataset_prepared", "train", "images", "*.*"))
for img_p in ws_imgs:
    fname = os.path.basename(img_p)
    bname, _ = os.path.splitext(fname)
    lbl_p = os.path.join(BASE_DIR, "dataset_prepared", "train", "labels", f"{bname}.txt")
    
    shutil.copy2(img_p, os.path.join(DATASET_LARGE, "train", "images", fname))
    if os.path.exists(lbl_p):
        shutil.copy2(lbl_p, os.path.join(DATASET_LARGE, "train", "labels", f"{bname}.txt"))

ws_val_imgs = glob.glob(os.path.join(BASE_DIR, "dataset_prepared", "val", "images", "*.*"))
for img_p in ws_val_imgs:
    fname = os.path.basename(img_p)
    bname, _ = os.path.splitext(fname)
    lbl_p = os.path.join(BASE_DIR, "dataset_prepared", "val", "labels", f"{bname}.txt")
    
    shutil.copy2(img_p, os.path.join(DATASET_LARGE, "val", "images", fname))
    if os.path.exists(lbl_p):
        shutil.copy2(lbl_p, os.path.join(DATASET_LARGE, "val", "labels", f"{bname}.txt"))

ws_test_imgs = glob.glob(os.path.join(BASE_DIR, "dataset", "test", "images", "*.*"))
for img_p in ws_test_imgs:
    fname = os.path.basename(img_p)
    bname, _ = os.path.splitext(fname)
    lbl_p = os.path.join(BASE_DIR, "dataset", "test", "labels", f"{bname}.txt")
    
    shutil.copy2(img_p, os.path.join(DATASET_LARGE, "test", "images", fname))
    if os.path.exists(lbl_p):
        shutil.copy2(lbl_p, os.path.join(DATASET_LARGE, "test", "labels", f"{bname}.txt"))

# Clean up temp
shutil.rmtree(temp_extract, ignore_errors=True)

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
print("  LARGE DATASET STRUCTURING COMPLETE!")
print(f"  • Train Images: {len(os.listdir(os.path.join(DATASET_LARGE, 'train', 'images')))}")
print(f"  • Val Images:   {len(os.listdir(os.path.join(DATASET_LARGE, 'val', 'images')))}")
print(f"  • Test Images:  {len(os.listdir(os.path.join(DATASET_LARGE, 'test', 'images')))}")
print(f"  • YAML Config:  {yaml_path}")
print("=" * 75 + "\n")
