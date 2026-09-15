"""
DyMATES - Expanded Dataset Fetcher
Downloads and prepares a large-scale dataset for YOLOv11 helmet and traffic violation detection.
"""

import os
import sys
import zipfile
import urllib.request
import glob
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_LARGE = os.path.join(BASE_DIR, "dataset_large")

print("=" * 75)
print("  DyMATES — EXPANDED DATASET ACQUISITION SUITE")
print("=" * 75)

ZIP_PATH = os.path.join(BASE_DIR, "helmet_dataset.zip")


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / max(1, total_size))
    mb_downloaded = (count * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  → Downloading dataset: {percent}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)")
    sys.stdout.flush()


def prepare_dataset():
    if not os.path.exists(ZIP_PATH):
        print(f"\n[1/3] Fetching dataset archive from open release...")
        # Direct dataset download link
        try:
            req = urllib.request.Request(
                "https://github.com/yurizzzzz/Helmet-Detection-YoloV5/releases/download/v1.0/dataset.zip",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as f:
                total_size = int(resp.headers.get("content-length", 0))
                block_size = 8192
                count = 0
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    count += 1
                    download_progress(count, block_size, total_size)
            print("\n  ✔ Download complete!")
        except Exception as e:
            print(f"\n  ⚠ Primary download link failed: {e}. Trying fallback open source repository...")
            urllib.request.urlretrieve(
                "https://github.com/sidpro-hash/Helmet-Detection-YOLOv5/releases/download/v1.0/dataset.zip",
                ZIP_PATH,
                download_progress
            )
            print("\n  ✔ Fallback download complete!")

    print(f"\n[2/3] Extracting archive to {DATASET_LARGE}...")
    temp_extract = os.path.join(BASE_DIR, "temp_dataset_extract")
    os.makedirs(temp_extract, exist_ok=True)
    
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(temp_extract)
    print("  ✔ Extraction finished.")

    print(f"\n[3/3] Organizing dataset splits (train / val / test)...")
    os.makedirs(os.path.join(DATASET_LARGE, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_LARGE, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_LARGE, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_LARGE, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_LARGE, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_LARGE, "test", "labels"), exist_ok=True)

    # Collect images and labels from extracted directory
    all_imgs = glob.glob(os.path.join(temp_extract, "**", "*.[jJ][pP][gG]"), recursive=True) + \
               glob.glob(os.path.join(temp_extract, "**", "*.[pP][nN][gG]"), recursive=True) + \
               glob.glob(os.path.join(temp_extract, "**", "*.[jJ][pP][eE][gG]"), recursive=True)

    print(f"  → Total raw images found: {len(all_imgs)}")
    
    # Merge existing workspace test/prepared images
    existing_imgs = glob.glob(os.path.join(BASE_DIR, "dataset_prepared", "**", "images", "*.*"), recursive=True) + \
                    glob.glob(os.path.join(BASE_DIR, "dataset", "test", "images", "*.*"))
    print(f"  → Merging workspace images: {len(existing_imgs)}")

    combined_imgs = list(set(all_imgs + existing_imgs))
    total_imgs = len(combined_imgs)
    
    n_train = int(0.80 * total_imgs)
    n_val = int(0.10 * total_imgs)

    train_imgs = combined_imgs[:n_train]
    val_imgs = combined_imgs[n_train:n_train + n_val]
    test_imgs = combined_imgs[n_train + n_val:]

    def copy_pair(img_list, target_split):
        copied = 0
        for img_p in img_list:
            fname = os.path.basename(img_p)
            bname, _ = os.path.splitext(fname)
            
            parent_dir = os.path.dirname(img_p)
            possible_lbl_paths = [
                os.path.join(parent_dir, "..", "labels", f"{bname}.txt"),
                os.path.join(parent_dir, f"{bname}.txt"),
                os.path.join(BASE_DIR, "dataset", "test", "labels", f"{bname}.txt"),
                os.path.join(BASE_DIR, "dataset_prepared", "train", "labels", f"{bname}.txt"),
                os.path.join(BASE_DIR, "dataset_prepared", "val", "labels", f"{bname}.txt"),
            ]
            
            lbl_p = None
            for p in possible_lbl_paths:
                if os.path.exists(p):
                    lbl_p = p
                    break

            dst_img = os.path.join(DATASET_LARGE, target_split, "images", fname)
            shutil.copy2(img_p, dst_img)

            dst_lbl = os.path.join(DATASET_LARGE, target_split, "labels", f"{bname}.txt")
            if lbl_p and os.path.exists(lbl_p):
                shutil.copy2(lbl_p, dst_lbl)
            else:
                with open(dst_lbl, "w") as f:
                    pass
            copied += 1
        return copied

    c_tr = copy_pair(train_imgs, "train")
    c_va = copy_pair(val_imgs, "val")
    c_te = copy_pair(test_imgs, "test")

    shutil.rmtree(temp_extract, ignore_errors=True)

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
    print("  DATASET PREPARATION COMPLETE!")
    print(f"  • Train Images: {c_tr}")
    print(f"  • Val Images:   {c_va}")
    print(f"  • Test Images:  {c_te}")
    print(f"  • Total Images: {c_tr + c_va + c_te}")
    print(f"  • YAML Config:  {yaml_path}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    prepare_dataset()
