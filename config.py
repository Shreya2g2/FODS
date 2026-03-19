# stylesense/config.py
"""
Central configuration for StyleSense.
All hyperparameters, paths, and constants live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
STATIC_DIR  = ROOT_DIR / "static"

RAW_IMAGES_DIR      = DATA_DIR / "raw_images"
PROCESSED_DIR       = DATA_DIR / "processed"
EMBEDDINGS_DIR      = DATA_DIR / "embeddings"
CATALOG_CSV         = DATA_DIR / "catalog.csv"

# ── CNN / Feature Extraction ───────────────────────────────────────────────
CNN_BACKBONE        = "efficientnet_b3"   # timm model name
CNN_EMBED_DIM       = 1536               # EfficientNet-B3 feature dim
IMAGE_SIZE          = (224, 224)
BATCH_SIZE          = 32
NUM_WORKERS         = 4

# Category classifier (fine-tuned head)
NUM_CATEGORIES      = 8    # top, bottom, dress, outerwear, shoes, bag, hat, accessory
CATEGORY_NAMES      = [
    "top", "bottom", "dress", "outerwear",
    "shoes", "bag", "hat", "accessory"
]

# ── Color Analysis ─────────────────────────────────────────────────────────
N_DOMINANT_COLORS   = 5    # dominant colors to extract per image
COLOR_PALETTE_SIZE  = 16   # color groups for harmony checking

# Curated color harmony rules (Itten's color theory)
COMPLEMENTARY_PAIRS = [
    ("red", "green"), ("blue", "orange"), ("yellow", "purple"),
    ("navy", "beige"), ("black", "white"), ("brown", "cream"),
]
ANALOGOUS_FAMILIES = {
    "warm":    ["red", "orange", "yellow", "rust", "coral", "gold"],
    "cool":    ["blue", "purple", "teal", "navy", "mint", "grey"],
    "neutral": ["white", "black", "grey", "beige", "cream", "brown"],
    "earth":   ["brown", "khaki", "olive", "tan", "rust"],
}
CLASHING_PAIRS = [
    ("red", "pink"), ("orange", "red"), ("green", "blue-green"),
    ("yellow", "neon-green"),
]

# ── Occasion → Attribute Rules ─────────────────────────────────────────────
OCCASION_RULES = {
    "college": {
        "allowed_formality":  ["casual", "smart-casual"],
        "forbidden_items":    ["tuxedo", "gown", "stiletto"],
        "preferred_fabrics":  ["cotton", "denim", "jersey"],
        "color_vibe":         "relaxed",
    },
    "party": {
        "allowed_formality":  ["semi-formal", "smart-casual", "casual"],
        "forbidden_items":    ["pajamas", "sportswear"],
        "preferred_fabrics":  ["satin", "velvet", "silk", "sequin"],
        "color_vibe":         "bold",
    },
    "formal": {
        "allowed_formality":  ["formal", "semi-formal"],
        "forbidden_items":    ["jeans", "sneakers", "hoodies", "shorts"],
        "preferred_fabrics":  ["wool", "silk", "linen"],
        "color_vibe":         "muted",
    },
    "casual": {
        "allowed_formality":  ["casual"],
        "forbidden_items":    [],
        "preferred_fabrics":  ["cotton", "denim", "jersey", "linen"],
        "color_vibe":         "any",
    },
    "date": {
        "allowed_formality":  ["smart-casual", "semi-formal"],
        "forbidden_items":    ["sportswear", "pajamas"],
        "preferred_fabrics":  ["silk", "satin", "cotton"],
        "color_vibe":         "romantic",
    },
    "wedding": {
        "allowed_formality":  ["formal", "semi-formal"],
        "forbidden_items":    ["white_dress", "jeans", "sneakers"],
        "preferred_fabrics":  ["silk", "chiffon", "lace"],
        "color_vibe":         "elegant",
    },
}

# ── Season → Fabric / Layer Rules ─────────────────────────────────────────
SEASON_RULES = {
    "summer": {
        "preferred_fabrics": ["cotton", "linen", "chiffon"],
        "avoid_fabrics":     ["wool", "fleece", "velvet"],
        "layers":            0,
        "colors":            "bright",
    },
    "winter": {
        "preferred_fabrics": ["wool", "fleece", "velvet", "corduroy"],
        "avoid_fabrics":     ["chiffon", "linen"],
        "layers":            2,
        "colors":            "deep",
    },
    "spring": {
        "preferred_fabrics": ["cotton", "linen", "jersey"],
        "avoid_fabrics":     ["heavy-wool"],
        "layers":            1,
        "colors":            "pastel",
    },
    "autumn": {
        "preferred_fabrics": ["wool", "denim", "corduroy", "leather"],
        "avoid_fabrics":     ["chiffon"],
        "layers":            1,
        "colors":            "earth",
    },
}

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────
SCORE_WEIGHTS = {
    "color_harmony":         0.30,
    "occasion_fit":          0.30,
    "style_consistency":     0.25,
    "season_appropriateness":0.15,
}

# ── Recommendation ─────────────────────────────────────────────────────────
TOP_K_CANDIDATES    = 20   # retrieve top-K from embedding index before re-ranking
SIMILARITY_METRIC   = "cosine"
