"""
DyMATES - GPU-Optimized Model Training Script
==============================================

Fine-tunes YOLOv11-Nano on the helmet/motorcycle/rider dataset.
Optimized for fast GPU training (1-2 hours target).

Usage:
    # On GPU server:
    python train.py                              # default: 50 epochs, GPU
    python train.py --epochs 50 --batch 16       # standard GPU training
    python train.py --epochs 30 --batch 8        # low-VRAM GPU
    python train.py --device cpu --batch 4       # CPU fallback (slow)
    
    # Quick test (5 epochs):
    python train.py --epochs 5 --batch 16        # verify setup works
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    exit(1)


def find_data_yaml(dataset_dir="dataset"):
    """Auto-detect the data.yaml file in the dataset directory."""
    dataset_path = Path(dataset_dir)

    candidates = [
        dataset_path / "data.yaml",
        dataset_path / "dataset.yaml",
    ]

    for f in dataset_path.rglob("*.yaml"):
        if "coco" not in f.name.lower():
            candidates.append(f)

    for c in candidates:
        if c.exists():
            print(f"[Train] Found dataset config: {c}")
            return str(c)

    return None


def fix_data_yaml_paths(data_yaml_path):
    """
    Fix the path in data.yaml to use the absolute path of the current machine.
    This is needed when transferring the dataset between machines (laptop → GPU server).
    """
    import yaml
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    existing_path = data.get('path', '')
    if existing_path and os.path.exists(existing_path):
        print(f"[Train] Valid dataset path confirmed: {existing_path}")
        return data
    
    # Get the dataset directory (parent of data.yaml)
    dataset_dir = str(Path(data_yaml_path).parent.resolve())
    
    if data.get('path') != dataset_dir:
        print(f"[Train] Fixing data.yaml path: {data.get('path')} → {dataset_dir}")
        data['path'] = dataset_dir
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"[Train] data.yaml updated successfully")
    
    return data


def train(args):
    """Run YOLOv11-Nano fine-tuning with GPU optimization."""

    # ── Find dataset config ───────────────────────────────────────
    data_yaml = args.data
    if not os.path.exists(data_yaml):
        found = find_data_yaml("dataset")
        if found:
            data_yaml = found
        else:
            print("\n[ERROR] Cannot find data.yaml!")
            print("Make sure your dataset is extracted to the 'dataset/' folder.")
            print("Expected structure:")
            print("  dataset/")
            print("    ├── train/images/")
            print("    ├── train/labels/")
            print("    ├── valid/images/  (or val/images/)")
            print("    ├── valid/labels/  (or val/labels/)")
            print("    └── data.yaml")
            return

    # Fix paths in data.yaml for this machine
    try:
        fix_data_yaml_paths(data_yaml)
    except Exception as e:
        print(f"[Train] Warning: Could not fix paths in data.yaml: {e}")
        print("[Train] Continuing with existing paths...")

    # ── Detect device ─────────────────────────────────────────────
    device = args.device
    try:
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "0"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                print(f"[Train] GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
                
                # Auto-adjust batch size based on GPU memory
                if gpu_mem < 6 and args.batch > 8:
                    print(f"[Train] Low GPU memory — reducing batch to 8")
                    args.batch = 8
                elif gpu_mem < 4 and args.batch > 4:
                    print(f"[Train] Very low GPU memory — reducing batch to 4")
                    args.batch = 4
            else:
                device = "cpu"
                print("[Train] No GPU detected. Using CPU (will be SLOW).")
                if args.batch > 4:
                    args.batch = 4
                    print(f"[Train] Reduced batch to {args.batch} for CPU")
    except ImportError:
        device = "cpu"
        print("[Train] PyTorch not fully configured. Using CPU.")

    print(f"\n{'='*60}")
    print(f"  DyMATES — YOLOv11-Nano Training")
    print(f"{'='*60}")
    print(f"  Dataset:  {data_yaml}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Img Size: {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print(f"  Device:   {device}")
    print(f"  Workers:  {args.workers}")
    print(f"{'='*60}\n")

    # ── Load base model ───────────────────────────────────────────
    base_model = args.base_model
    if not os.path.exists(base_model):
        base_model = "yolo11n.pt"
        print(f"[Train] Using base model: {base_model} (auto-downloads)")
    
    model = YOLO(base_model)

    # ── Train ─────────────────────────────────────────────────────
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs",
        name="dymates_v1",
        patience=15,           # early stopping
        save=True,
        save_period=5,         # checkpoint every 5 epochs
        plots=True,            # training plots (loss, mAP)
        verbose=True,
        exist_ok=True,
        # ── Augmentation (aggressive for small datasets) ──────────
        augment=True,
        mosaic=1.0,            # mosaic augmentation
        mixup=0.1,             # mild mixup
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,           # hue augmentation  
        hsv_s=0.7,             # saturation augmentation
        hsv_v=0.4,             # value (brightness) augmentation
        degrees=10.0,          # rotation
        translate=0.1,         # translation
        scale=0.5,             # scaling
        # ── Optimizer settings ────────────────────────────────────
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        # ── Workers ──────────────────────────────────────────────
        workers=args.workers,
    )

    # ── Copy best model to models/ ────────────────────────────────
    os.makedirs("models", exist_ok=True)
    
    # Search for best.pt in various possible locations
    possible_paths = [
        Path("runs/dymates_v1/weights/best.pt"),
        Path("runs/detect/dymates_v1/weights/best.pt"),
        Path("runs/detect/runs/dymates_v1/weights/best.pt"),
    ]
    
    best_found = None
    for p in possible_paths:
        if p.exists():
            best_found = p
            break
    
    # Also search recursively as fallback
    if not best_found:
        for p in Path("runs").rglob("best.pt"):
            best_found = p
            break
    
    if best_found:
        dest_multi = Path("models/dymates_yolov11n_multiclass_best.pt")
        dest_best = Path("models/best.pt")
        shutil.copy2(best_found, dest_multi)
        shutil.copy2(best_found, dest_best)
        print(f"\n[Train] ✅ Multi-class model saved to: {dest_multi}")
        print(f"[Train] ✅ Best model active at: {dest_best}")
    else:
        print("\n[Train] ⚠️ best.pt not found. Check the runs/ folder.")

    print("\n[Train] Training complete!")
    print("[Train] Check 'runs/dymates_v1/' for:")
    print("  - confusion_matrix.png")
    print("  - results.png (loss & mAP curves)")
    print("  - weights/best.pt")
    print("  - weights/last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyMATES - Train YOLOv11-Nano")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (auto-adjusted for GPU memory)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers (reduce if issues)")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt",
                        help="Base model to fine-tune from")
    args = parser.parse_args()
    train(args)
