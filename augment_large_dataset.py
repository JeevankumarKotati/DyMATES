"""
DyMATES - Data Augmentation & Expansion Engine
Expands dataset_large/train to 5,000+ training samples via geometric, CLAHE, and color augmentations.
"""

import os
import glob
import cv2
import numpy as np
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_LARGE = os.path.join(BASE_DIR, "dataset_large")
TRAIN_IMG_DIR = os.path.join(DATASET_LARGE, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_LARGE, "train", "labels")

print("=" * 75)
print("  DyMATES — DATASET AUGMENTATION & EXPANSION ENGINE")
print("=" * 75)


def augment_horizontal_flip(image, boxes):
    h, w = image.shape[:2]
    flipped_img = cv2.flip(image, 1)
    flipped_boxes = []
    for cls_id, xc, yc, bw, bh in boxes:
        new_xc = 1.0 - xc
        flipped_boxes.append((cls_id, new_xc, yc, bw, bh))
    return flipped_img, flipped_boxes


def augment_clahe(image, boxes):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced, boxes


def augment_brightness(image, boxes):
    factor = random.uniform(0.7, 1.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_img, boxes


def main():
    img_paths = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.*"))
    print(f"[1/2] Base training images: {len(img_paths)}")

    aug_functions = [
        ("hflip", augment_horizontal_flip),
        ("clahe", augment_clahe),
        ("bright", augment_brightness),
    ]

    created = 0
    target_total = 5000
    needed_per_image = max(1, (target_total - len(img_paths)) // len(img_paths) + 1)

    print(f"[2/2] Generating augmentations (targeting {target_total}+ samples)...")

    for idx, img_p in enumerate(img_paths):
        bname = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(TRAIN_LBL_DIR, f"{bname}.txt")

        img = cv2.imread(img_p)
        if img is None:
            continue

        boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        boxes.append((cls_id, xc, yc, bw, bh))

        # Generate augmentations
        for aug_idx in range(needed_per_image):
            aug_name, aug_fn = aug_functions[aug_idx % len(aug_functions)]
            new_img, new_boxes = aug_fn(img, boxes)

            new_fname = f"{bname}_aug_{aug_name}_{aug_idx}.jpg"
            new_lbl_fname = f"{bname}_aug_{aug_name}_{aug_idx}.txt"

            new_img_p = os.path.join(TRAIN_IMG_DIR, new_fname)
            new_lbl_p = os.path.join(TRAIN_LBL_DIR, new_lbl_fname)

            cv2.imwrite(new_img_p, new_img)

            yolo_lines = [f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cid, xc, yc, bw, bh in new_boxes]
            with open(new_lbl_p, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            created += 1

        if (idx + 1) % 500 == 0:
            print(f"  → Processed {idx + 1}/{len(img_paths)} base images ({created} augmented images generated)")

    final_count = len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.*")))
    print("\n" + "=" * 75)
    print("  AUGMENTATION COMPLETE!")
    print(f"  • Base Images:      {len(img_paths)}")
    print(f"  • Augmented Images: {created}")
    print(f"  • Total Train Set:  {final_count} images")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
