# stylesense/data/prepare_dataset.py
"""
Dataset Preparation Pipeline
==============================
Step 1: Organize raw images into catalog CSV
Step 2: Pre-compute CNN embeddings for all catalog items
Step 3: Build FAISS-style numpy index for fast retrieval

Run ONCE before training or serving:
    python data/prepare_dataset.py --images_dir data/raw_images --output_dir data/

Expected folder structure of raw_images/:
    raw_images/
      tops/
        img001.jpg, img002.jpg, ...
      bottoms/
        ...
      shoes/
        ...
      accessories/
        ...

(Optional) If you have a Kaggle fashion dataset, point --images_dir to it.
Supported datasets:
  - DeepFashion (https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
  - Fashion Product Images (Kaggle)
  - Polyvore Outfits
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from config import (
    RAW_IMAGES_DIR, EMBEDDINGS_DIR, CATALOG_CSV,
    CATEGORY_NAMES, BATCH_SIZE
)


CATEGORY_FOLDER_MAP = {
    "top":       ["tops", "shirts", "tshirts", "blouses", "upper_body"],
    "bottom":    ["bottoms", "pants", "jeans", "skirts", "lower_body"],
    "shoes":     ["shoes", "footwear", "sneakers", "boots"],
    "accessory": ["accessories", "bags", "hats", "jewelry"],
    "dress":     ["dresses", "full_body"],
    "outerwear": ["outerwear", "jackets", "coats"],
}

FABRIC_KEYWORDS = {
    "denim":   ["jean", "denim"],
    "cotton":  ["cotton", "tee", "shirt", "tshirt"],
    "linen":   ["linen"],
    "silk":    ["silk", "satin"],
    "wool":    ["wool", "knit", "sweater"],
    "leather": ["leather", "boot"],
    "fleece":  ["fleece", "sweat", "hoodie"],
    "chiffon": ["chiffon", "flowy", "blouse"],
}

FORMALITY_KEYWORDS = {
    0.8: ["suit", "blazer", "tuxedo", "formal", "dress-shirt", "gown"],
    0.6: ["chino", "polo", "loafer", "blouse", "midi"],
    0.4: ["casual", "shirt", "jeans", "sneaker"],
    0.2: ["tshirt", "hoodie", "sweatpant", "jogger", "flip-flop"],
}


def infer_fabric(name: str) -> str:
    name_lower = name.lower()
    for fabric, keywords in FABRIC_KEYWORDS.items():
        if any(k in name_lower for k in keywords):
            return fabric
    return "cotton"

def infer_formality(name: str, category: str) -> float:
    name_lower = name.lower() + " " + category.lower()
    for score, keywords in FORMALITY_KEYWORDS.items():
        if any(k in name_lower for k in keywords):
            return score
    return 0.4   # default: casual

def infer_season(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["wool", "coat", "fleece", "puffer"]): return "winter"
    if any(k in n for k in ["linen", "tank", "shorts", "sandal"]): return "summer"
    if any(k in n for k in ["trench", "denim", "jacket"]): return "autumn"
    return "all-season"


def build_catalog(images_dir: Path) -> pd.DataFrame:
    """Scan image folder structure and build catalog DataFrame."""
    rows = []
    item_id = 0

    for cat, folder_names in CATEGORY_FOLDER_MAP.items():
        for folder in folder_names:
            folder_path = images_dir / folder
            if not folder_path.exists():
                continue

            for img_path in sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png")):
                name = img_path.stem.replace("_", " ").replace("-", " ").title()
                rows.append({
                    "item_id":       f"item_{item_id:05d}",
                    "name":          name,
                    "category":      cat,
                    "color":         "unknown",   # will be filled by color extractor
                    "fabric":        infer_fabric(name),
                    "formality":     infer_formality(name, cat),
                    "season":        infer_season(name),
                    "occasion_tags": "casual,college",
                    "image_path":    str(img_path),
                    "embedding_path":"",
                })
                item_id += 1

    df = pd.DataFrame(rows)
    print(f"[Catalog] Found {len(df)} images across {df['category'].nunique()} categories")
    return df


def precompute_embeddings(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run CNN on all images and save embeddings as .npy files."""
    try:
        from models.cnn_extractor import ClothingCNNExtractor
        extractor = ClothingCNNExtractor()
    except ImportError:
        print("[Embeddings] torch/timm not available. Skipping embedding precomputation.")
        print("  Install: pip install torch torchvision timm")
        return df

    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Embeddings] Computing for {len(df)} items...")
    emb_paths = []
    batch_imgs, batch_ids, batch_paths = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(row["image_path"])
        if not img_path.exists():
            emb_paths.append("")
            continue

        batch_imgs.append(img_path)
        batch_ids.append(row["item_id"])
        batch_paths.append(str(img_path))

        if len(batch_imgs) == BATCH_SIZE:
            _process_batch(extractor, batch_imgs, batch_ids, emb_dir, emb_paths)
            batch_imgs, batch_ids, batch_paths = [], [], []

    if batch_imgs:
        _process_batch(extractor, batch_imgs, batch_ids, emb_dir, emb_paths)

    # Fill remaining (empty image paths)
    while len(emb_paths) < len(df):
        emb_paths.append("")

    df["embedding_path"] = emb_paths
    return df


def _process_batch(extractor, images, ids, emb_dir, out_list):
    try:
        embeddings = extractor.extract_batch(images)
        for iid, emb in zip(ids, embeddings):
            save_path = emb_dir / f"{iid}.npy"
            np.save(save_path, emb)
            out_list.append(str(save_path))
    except Exception as e:
        print(f"  [Error] Batch failed: {e}")
        for _ in ids:
            out_list.append("")


def add_dominant_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Extract dominant color names for catalog items."""
    try:
        from utils.color_analyzer import DominantColorExtractor
        extractor = DominantColorExtractor(n_colors=1)
    except ImportError:
        return df

    print("[Colors] Extracting dominant colors...")
    colors = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(str(row.get("image_path", "")))
        if img_path.exists():
            try:
                result = extractor.extract(img_path)
                colors.append(result[0]["name"] if result else "unknown")
            except Exception:
                colors.append("unknown")
        else:
            colors.append("unknown")
    df["color"] = colors
    return df


def main(args):
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  StyleSense Dataset Preparation Pipeline")
    print("=" * 55)

    # Step 1: Build catalog
    print("\n[Step 1/3] Building catalog from image folders...")
    df = build_catalog(images_dir)

    if df.empty:
        print("\n  No images found. Creating demo catalog...")
        # Import demo catalog from CatalogManager
        from utils.recommender import CatalogManager
        cm = CatalogManager(catalog_csv=Path("nonexistent"))
        df = cm.df

    # Step 2: Dominant colors
    print("\n[Step 2/3] Extracting dominant colors...")
    df = add_dominant_colors(df)

    # Step 3: CNN embeddings
    print("\n[Step 3/3] Pre-computing CNN embeddings...")
    df = precompute_embeddings(df, output_dir)

    # Save catalog
    catalog_path = output_dir / "catalog.csv"
    df.to_csv(catalog_path, index=False)
    print(f"\n[Done] Catalog saved: {catalog_path}")
    print(f"       Items: {len(df)}")
    print(f"       Categories: {df['category'].value_counts().to_dict()}")
    print(f"       With embeddings: {(df['embedding_path'] != '').sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="data/raw_images")
    parser.add_argument("--output_dir", default="data/")
    args = parser.parse_args()
    main(args)
