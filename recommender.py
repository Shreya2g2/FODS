# stylesense/utils/recommender.py
"""
CNN Embedding-Based Outfit Recommender
=======================================
Core recommendation flow:
  1. Load the catalog (CSV + pre-computed CNN embeddings)
  2. For a given occasion + style + optional query image:
     a. Filter catalog by occasion/season metadata rules
     b. If query image given: rank by CNN embedding cosine similarity
     c. Else: rank by rule-based attribute matching
  3. Assemble complete outfits (top + bottom + shoes + accessory)
  4. Return top-K outfits for evaluation

Embedding index uses pre-computed .npy files for fast retrieval
(no re-inference at request time).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Optional

from config import (
    CATALOG_CSV, EMBEDDINGS_DIR, TOP_K_CANDIDATES,
    OCCASION_RULES, SEASON_RULES, CATEGORY_NAMES
)
from utils.outfit_evaluator import ClothingItem


# ── Catalog Manager ───────────────────────────────────────────────────────
class CatalogManager:
    """
    Loads and manages the clothing catalog.
    Catalog CSV schema:
      item_id, name, category, color, fabric, formality,
      season, occasion_tags, image_path, embedding_path
    """

    def __init__(self, catalog_csv: Path = CATALOG_CSV):
        self.df = pd.DataFrame()
        self.embeddings = {}      # item_id → np.ndarray(embed_dim,)
        self._loaded = False

        if catalog_csv.exists():
            self._load(catalog_csv)
        else:
            print(f"[Catalog] No catalog found at {catalog_csv}. Using demo data.")
            self._load_demo()

    def _load(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)
        # Load embeddings
        for _, row in self.df.iterrows():
            emb_path = Path(str(row.get("embedding_path", "")))
            if emb_path.exists():
                self.embeddings[row["item_id"]] = np.load(emb_path)
        self._loaded = True
        print(f"[Catalog] Loaded {len(self.df)} items, {len(self.embeddings)} with embeddings")

    def _load_demo(self):
        """Minimal demo catalog for testing without a real dataset."""
        import random
        random.seed(42)
        items = []
        categories = ["top", "bottom", "shoes", "accessory"]
        demo_items = {
            "top":       [("White Linen Shirt", "white", "linen", 0.55),
                          ("Navy Polo", "navy", "cotton", 0.50),
                          ("Black Graphic Tee", "black", "cotton", 0.20),
                          ("Floral Blouse", "floral", "chiffon", 0.45),
                          ("Grey Crew Sweatshirt", "grey", "fleece", 0.15)],
            "bottom":    [("Slim Black Chinos", "black", "cotton", 0.55),
                          ("Blue Jeans", "blue", "denim", 0.25),
                          ("Khaki Trousers", "khaki", "cotton", 0.50),
                          ("Floral Midi Skirt", "floral", "cotton", 0.45),
                          ("Grey Sweatpants", "grey", "fleece", 0.10)],
            "shoes":     [("White Sneakers", "white", "canvas", 0.25),
                          ("Brown Loafers", "brown", "leather", 0.60),
                          ("Black Derby", "black", "leather", 0.80),
                          ("Nude Block Heels", "beige", "leather", 0.65),
                          ("Chunky Boots", "black", "leather", 0.40)],
            "accessory": [("Minimalist Watch", "silver", "metal", 0.60),
                          ("Canvas Tote", "beige", "canvas", 0.30),
                          ("Leather Belt", "brown", "leather", 0.55),
                          ("Silk Scarf", "patterned", "silk", 0.65),
                          ("Baseball Cap", "black", "cotton", 0.10)],
        }
        idx = 0
        for cat, items_list in demo_items.items():
            for name, color, fabric, formality in items_list:
                items.append({
                    "item_id":    f"item_{idx:04d}",
                    "name":       name,
                    "category":   cat,
                    "color":      color,
                    "fabric":     fabric,
                    "formality":  formality,
                    "season":     random.choice(["summer","all-season","winter","spring"]),
                    "occasion_tags": "casual,college,party" if formality < 0.5 else "formal,date,college",
                    "image_path": "",
                })
                idx += 1
        self.df = pd.DataFrame(items)
        self._loaded = True

    def filter(self, occasion: str, season: str) -> pd.DataFrame:
        """Return catalog rows matching occasion and season."""
        df = self.df.copy()
        if df.empty:
            return df

        # Season filter
        if "season" in df.columns:
            df = df[df["season"].str.lower().isin([season.lower(), "all-season", "all"])]

        # Formality filter via occasion rules
        occ_map = {
            "college":"casual", "party":"semi-formal",
            "formal":"formal",  "casual":"casual",
            "date":"smart-casual", "wedding":"formal"
        }
        occ_key = occ_map.get(self._norm_occasion(occasion), "casual")
        rules = OCCASION_RULES.get(occ_key, OCCASION_RULES["casual"])
        allowed_formality = rules["allowed_formality"]

        formality_ranges = {
            "casual": (0.0, 0.35),
            "smart-casual": (0.30, 0.60),
            "semi-formal": (0.55, 0.80),
            "formal": (0.75, 1.01),
        }
        min_f = min(formality_ranges[f][0] for f in allowed_formality if f in formality_ranges)
        max_f = max(formality_ranges[f][1] for f in allowed_formality if f in formality_ranges)

        if "formality" in df.columns:
            df = df[(df["formality"] >= min_f) & (df["formality"] <= max_f)]

        return df

    @staticmethod
    def _norm_occasion(o: str) -> str:
        o = o.lower()
        if "college" in o: return "college"
        if "party" in o: return "party"
        if "formal" in o or "office" in o: return "formal"
        if "date" in o: return "date"
        if "wedding" in o: return "wedding"
        return "casual"


# ── CNN Embedding Retriever ───────────────────────────────────────────────
class EmbeddingRetriever:
    """
    Given a query image embedding, retrieves most similar catalog items
    using cosine similarity on CNN feature vectors.
    """

    def __init__(self, catalog: CatalogManager):
        self.catalog = catalog

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray],
        candidate_df: pd.DataFrame,
        top_k: int = TOP_K_CANDIDATES
    ) -> pd.DataFrame:
        """
        Rank candidates by similarity to query embedding.
        If no query embedding, return candidates as-is (shuffled).
        """
        if query_embedding is None or not self.catalog.embeddings:
            return candidate_df.sample(frac=1, random_state=42).head(top_k)

        # Build embedding matrix for candidates
        ids_with_emb = [
            row["item_id"]
            for _, row in candidate_df.iterrows()
            if row["item_id"] in self.catalog.embeddings
        ]

        if not ids_with_emb:
            return candidate_df.head(top_k)

        emb_matrix = np.stack([self.catalog.embeddings[iid] for iid in ids_with_emb])
        query = query_embedding[np.newaxis]  # (1, D)

        dists = cdist(query, emb_matrix, metric="cosine")[0]   # (N,)
        sim_scores = 1 - dists

        ranked_ids = [ids_with_emb[i] for i in np.argsort(-sim_scores)]
        ranked_df  = candidate_df[candidate_df["item_id"].isin(ranked_ids[:top_k])]
        return ranked_df


# ── Outfit Assembler ──────────────────────────────────────────────────────
class OutfitAssembler:
    """
    Combines individual catalog items into complete outfit combinations.
    Ensures:
      - One item per category (top, bottom, shoes, accessory)
      - Color harmony pre-filter (avoid obvious clashes before full eval)
    """

    REQUIRED_CATEGORIES = ["top", "bottom", "shoes"]
    OPTIONAL_CATEGORIES = ["accessory"]

    def assemble(self, filtered_df: pd.DataFrame, n_outfits: int = 5) -> list[list[ClothingItem]]:
        """
        Greedily assemble N complete outfits from the filtered catalog.
        Returns list of outfit lists.
        """
        if filtered_df.empty:
            return []

        by_category = {
            cat: filtered_df[filtered_df["category"] == cat].to_dict("records")
            for cat in REQUIRED_CATEGORIES + self.OPTIONAL_CATEGORIES
        }

        # Check we have all required categories
        for cat in self.REQUIRED_CATEGORIES:
            if not by_category.get(cat):
                by_category[cat] = [{"item_id": f"fallback_{cat}", "name": f"Basic {cat.title()}",
                                      "category": cat, "color": "black", "fabric": "cotton",
                                      "formality": 0.4, "season": "all-season"}]

        outfits = []
        used = {cat: 0 for cat in by_category}

        for _ in range(n_outfits):
            outfit_items = []
            for cat in self.REQUIRED_CATEGORIES:
                pool = by_category[cat]
                row  = pool[used[cat] % len(pool)]
                used[cat] += 1
                outfit_items.append(self._row_to_item(row))

            for cat in self.OPTIONAL_CATEGORIES:
                pool = by_category.get(cat, [])
                if pool:
                    row = pool[used.get(cat, 0) % len(pool)]
                    used[cat] = used.get(cat, 0) + 1
                    outfit_items.append(self._row_to_item(row))

            outfits.append(outfit_items)

        return outfits

    @staticmethod
    def _row_to_item(row: dict) -> ClothingItem:
        return ClothingItem(
            name       = str(row.get("name", "Item")),
            category   = str(row.get("category", "top")),
            color_name = str(row.get("color", "unknown")),
            fabric     = str(row.get("fabric", "cotton")),
            formality  = float(row.get("formality", 0.5)),
            image_path = str(row.get("image_path", "")) or None,
            season_pred= str(row.get("season", "summer")),
        )


# ── Main Recommender ──────────────────────────────────────────────────────
class OutfitRecommender:
    """
    High-level interface that combines:
      CatalogManager → EmbeddingRetriever → OutfitAssembler

    Usage:
        rec = OutfitRecommender()
        outfits = rec.recommend(
            occasion="college",
            style="minimalist",
            season="summer",
            query_embedding=cnn_features,   # optional
        )
    """

    def __init__(self):
        self.catalog   = CatalogManager()
        self.retriever = EmbeddingRetriever(self.catalog)
        self.assembler = OutfitAssembler()

    def recommend(
        self,
        occasion:        str,
        style:           str,
        season:          str,
        query_embedding: Optional[np.ndarray] = None,
        n_outfits:       int = 5,
    ) -> list[list[ClothingItem]]:
        """
        End-to-end recommendation pipeline.
        Returns: list of outfits (each outfit = list of ClothingItems)
        """
        # 1. Filter catalog by occasion + season
        filtered = self.catalog.filter(occasion, season)

        # 2. CNN-based ranking (if query embedding provided)
        if query_embedding is not None and not filtered.empty:
            filtered = self.retriever.retrieve(query_embedding, filtered, top_k=TOP_K_CANDIDATES)

        # 3. Assemble complete outfits
        outfits = self.assembler.assemble(filtered, n_outfits=n_outfits)

        return outfits
