# stylesense/models/train_cnn.py
"""
Fine-tuning pipeline for the ClothingCNN on your dataset.

Training strategy (progressive unfreezing):
  Phase 1 (warm-up):  Freeze backbone, train only the heads (5 epochs)
  Phase 2 (fine-tune): Unfreeze last 3 EfficientNet blocks (15 epochs)
  Phase 3 (full):      Unfreeze all, very low LR (optional, 5 epochs)

Dataset expected structure:
  data/raw_images/
    ├── tops/        ← category subdirs
    ├── bottoms/
    ├── shoes/
    └── ...
  data/catalog.csv   ← optional metadata CSV

Run:
    python models/train_cnn.py --epochs 20 --batch_size 32
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    RAW_IMAGES_DIR, MODELS_DIR, CATALOG_CSV,
    NUM_CATEGORIES, CATEGORY_NAMES, BATCH_SIZE, IMAGE_SIZE
)
from models.cnn_extractor import ClothingCNN, get_transforms


# ── Dataset ───────────────────────────────────────────────────────────────
class ClothingDataset(Dataset):
    """
    Supports two modes:
      1. Folder-based:  data/raw_images/<category>/<img>.jpg
      2. CSV-based:     catalog.csv with columns [image_path, category, formality, season]
    """

    SEASON_MAP = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}

    def __init__(self, root_dir: Path, catalog_csv: Path | None = None,
                 transform=None, augment: bool = False):
        self.transform = transform or get_transforms(augment=augment)
        self.samples   = []   # list of (image_path, category_idx, formality, season_idx)

        if catalog_csv and catalog_csv.exists():
            df = pd.read_csv(catalog_csv)
            for _, row in df.iterrows():
                cat_idx = CATEGORY_NAMES.index(row["category"]) if row["category"] in CATEGORY_NAMES else 0
                sea_idx = self.SEASON_MAP.get(str(row.get("season", "summer")), 1)
                formality = float(row.get("formality", 0.5))
                self.samples.append((row["image_path"], cat_idx, formality, sea_idx))
        else:
            # Folder-based: infer category from directory name
            for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
                cat_dir = root_dir / cat_name
                if not cat_dir.exists():
                    # try plural / alternate names
                    for alt in [cat_name + "s", cat_name.replace("top", "tops"),
                                 cat_name.replace("shoe", "shoes")]:
                        if (root_dir / alt).exists():
                            cat_dir = root_dir / alt
                            break
                if cat_dir.exists():
                    for img_path in cat_dir.glob("*.jpg"):
                        self.samples.append((str(img_path), cat_idx, 0.5, 1))
                    for img_path in cat_dir.glob("*.png"):
                        self.samples.append((str(img_path), cat_idx, 0.5, 1))

        print(f"[Dataset] {len(self.samples)} images loaded")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cat_idx, formality, sea_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {
            "image":    img,
            "category": torch.tensor(cat_idx, dtype=torch.long),
            "formality":torch.tensor(formality, dtype=torch.float),
            "season":   torch.tensor(sea_idx, dtype=torch.long),
        }


# ── Loss ──────────────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Weighted combination of:
      - CrossEntropy for category (main task)
      - BCELoss for formality regression
      - CrossEntropy for season
    """
    def __init__(self, w_cat=1.0, w_form=0.5, w_sea=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.bce  = nn.BCELoss()
        self.w    = {"cat": w_cat, "form": w_form, "sea": w_sea}

    def forward(self, outputs, batch):
        l_cat  = self.ce(outputs["category_logits"], batch["category"])
        l_form = self.bce(outputs["formality"], batch["formality"])
        l_sea  = self.ce(outputs["season_logits"],   batch["season"])
        total  = self.w["cat"]*l_cat + self.w["form"]*l_form + self.w["sea"]*l_sea
        return total, {"category": l_cat.item(), "formality": l_form.item(), "season": l_sea.item()}


# ── Training loop ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        imgs = batch["image"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if k != "image"}

        optimizer.zero_grad()
        outputs = model(imgs)
        loss, _ = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds  = outputs["category_logits"].argmax(1)
        correct += (preds == batch["category"]).sum().item()
        n += imgs.size(0)

    return {"loss": total_loss / n, "category_acc": correct / n}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    for batch in tqdm(loader, desc="  val  ", leave=False):
        imgs = batch["image"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if k != "image"}
        outputs = model(imgs)
        loss, _ = criterion(outputs, batch)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs["category_logits"].argmax(1)
        correct += (preds == batch["category"]).sum().item()
        n += imgs.size(0)
    return {"loss": total_loss / n, "category_acc": correct / n}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Data ──
    dataset = ClothingDataset(RAW_IMAGES_DIR, CATALOG_CSV if CATALOG_CSV.exists() else None)
    val_size  = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ──
    model = ClothingCNN(pretrained=True).to(device)
    criterion = MultiTaskLoss()
    history = []

    # ── Phase 1: Warm-up (freeze backbone) ──
    print("\n[Phase 1] Warm-up — training heads only")
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    for epoch in range(min(5, args.epochs)):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch+1, "phase": 1, **tr, **{f"val_{k}": v for k,v in vl.items()}})
        print(f"  E{epoch+1:02d}  train_loss={tr['loss']:.4f}  val_acc={vl['category_acc']:.3f}")

    # ── Phase 2: Fine-tune last 3 blocks ──
    remaining = args.epochs - 5
    if remaining > 0:
        print("\n[Phase 2] Fine-tuning last 3 EfficientNet blocks")
        model.unfreeze_backbone(layers_from_end=3)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        best_acc, patience_ctr = 0.0, 0
        for epoch in range(remaining):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
            vl = validate(model, val_loader, criterion, device)
            scheduler.step()
            history.append({"epoch": epoch+6, "phase": 2, **tr, **{f"val_{k}": v for k,v in vl.items()}})
            print(f"  E{epoch+6:02d}  train_loss={tr['loss']:.4f}  val_acc={vl['category_acc']:.3f}")

            if vl["category_acc"] > best_acc:
                best_acc = vl["category_acc"]
                MODELS_DIR.mkdir(exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "clothing_cnn_best.pt")
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= 5:
                    print("  [EarlyStopping] No improvement for 5 epochs")
                    break

    # ── Save final ──
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "clothing_cnn_final.pt")
    with open(MODELS_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Done] Best val accuracy: {best_acc:.4f}")
    print(f"       Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args)
