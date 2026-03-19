# stylesense/models/cnn_extractor.py
"""
CNN-based visual feature extractor.

Architecture:
  EfficientNet-B3 (pretrained on ImageNet via timm)
  → Global Average Pooling
  → L2-normalized 1536-d embedding

Also includes a lightweight fine-tune head for:
  - Category classification  (8 classes)
  - Formality regression     (0.0 → 1.0)
  - Season classification    (4 classes)

Usage:
    extractor = ClothingCNNExtractor()
    embedding = extractor.extract(image_path)          # (1536,)
    preds     = extractor.classify(image_path)         # dict of predictions
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from config import (
    CNN_BACKBONE, CNN_EMBED_DIM, IMAGE_SIZE,
    NUM_CATEGORIES, CATEGORY_NAMES
)


# ── Image preprocessing pipeline ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(augment: bool = False):
    """Return train (augmented) or val/test transform pipeline."""
    if augment:
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(IMAGE_SIZE[0] * 1.14)),     # slight oversize then center-crop
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Main CNN Model ────────────────────────────────────────────────────────
class ClothingCNN(nn.Module):
    """
    Multi-task CNN:
      - Backbone: EfficientNet-B3 (pretrained, frozen at first)
      - Head 1: Category classifier   → 8-class softmax
      - Head 2: Formality regressor   → sigmoid scalar
      - Head 3: Season classifier     → 4-class softmax
    """

    def __init__(self, num_categories: int = NUM_CATEGORIES, pretrained: bool = True):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("Install timm:  pip install timm")

        # ── Backbone ──
        self.backbone = timm.create_model(
            CNN_BACKBONE,
            pretrained=pretrained,
            num_classes=0,          # remove default classifier
            global_pool="avg",      # global average pooling
        )
        embed_dim = self.backbone.num_features   # 1536 for B3

        # ── Shared projection (bottleneck) ──
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # ── Task heads ──
        self.head_category = nn.Linear(512, num_categories)
        self.head_formality = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.head_season    = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor):
        """
        Args:  x : (B, 3, H, W)
        Returns: dict with 'embedding', 'category_logits', 'formality', 'season_logits'
        """
        feat = self.backbone(x)            # (B, embed_dim)
        proj = self.proj(feat)             # (B, 512)

        return {
            "embedding":       nn.functional.normalize(feat, dim=1),   # L2-normed raw feat
            "proj":            proj,
            "category_logits": self.head_category(proj),                # (B, 8)
            "formality":       self.head_formality(proj).squeeze(1),    # (B,)
            "season_logits":   self.head_season(proj),                  # (B, 4)
        }

    def freeze_backbone(self):
        """Freeze backbone for warm-up training (train heads only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, layers_from_end: int = 3):
        """Gradually unfreeze last N blocks for fine-tuning."""
        all_layers = list(self.backbone.children())
        for layer in all_layers[-layers_from_end:]:
            for p in layer.parameters():
                p.requires_grad = True


# ── Inference wrapper ─────────────────────────────────────────────────────
class ClothingCNNExtractor:
    """
    High-level inference interface.
    Handles image loading, batching, and post-processing.
    """

    SEASON_NAMES = ["spring", "summer", "autumn", "winter"]

    def __init__(self, checkpoint_path: str | None = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ClothingCNN(pretrained=(checkpoint_path is None))
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[CNN] Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[CNN] Using ImageNet-pretrained {CNN_BACKBONE} (no fine-tune checkpoint)")

        self.model.to(self.device).eval()
        self.transform = get_transforms(augment=False)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _load_image(self, source) -> torch.Tensor:
        """Load PIL / path / numpy array → (1, 3, H, W) tensor."""
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(source.astype("uint8")).convert("RGB")
        elif isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(source)}")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, image_source) -> np.ndarray:
        """
        Extract L2-normalized CNN embedding.
        Returns: np.ndarray of shape (CNN_EMBED_DIM,)
        """
        tensor = self._load_image(image_source)
        out = self.model(tensor)
        return out["embedding"].cpu().numpy()[0]

    @torch.no_grad()
    def classify(self, image_source) -> dict:
        """
        Run full multi-task classification.

        Returns:
            {
              "category":  "top",
              "category_probs": {"top": 0.82, ...},
              "formality":  0.35,           # 0=casual, 1=formal
              "season":    "summer",
              "season_probs": {"summer": 0.61, ...},
              "embedding": np.ndarray
            }
        """
        tensor = self._load_image(image_source)
        out = self.model(tensor)

        cat_probs  = torch.softmax(out["category_logits"], dim=1)[0].cpu().numpy()
        sea_probs  = torch.softmax(out["season_logits"],   dim=1)[0].cpu().numpy()
        formality  = float(out["formality"][0].cpu())
        embedding  = out["embedding"][0].cpu().numpy()

        return {
            "category":      CATEGORY_NAMES[int(cat_probs.argmax())],
            "category_probs": dict(zip(CATEGORY_NAMES, cat_probs.tolist())),
            "formality":      round(formality, 3),
            "season":         self.SEASON_NAMES[int(sea_probs.argmax())],
            "season_probs":   dict(zip(self.SEASON_NAMES, sea_probs.tolist())),
            "embedding":      embedding,
        }

    @torch.no_grad()
    def extract_batch(self, image_sources: list) -> np.ndarray:
        """
        Extract embeddings for a list of images (batched for efficiency).
        Returns: np.ndarray of shape (N, CNN_EMBED_DIM)
        """
        tensors = torch.cat([self._load_image(s) for s in image_sources], dim=0)
        out = self.model(tensors)
        return out["embedding"].cpu().numpy()


# ── GradCAM Explainability ────────────────────────────────────────────────
class OutfitGradCAM:
    """
    Generates GradCAM heatmaps showing WHICH regions of the image
    the CNN focused on when classifying.

    Useful for the 'Explainability' USP — visually show the user
    what parts of their outfit the model found most relevant.
    """

    def __init__(self, extractor: ClothingCNNExtractor):
        self.extractor = extractor
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            self.GradCAM = GradCAM
            self.ClassifierOutputTarget = ClassifierOutputTarget
            self._available = True
        except ImportError:
            print("[GradCAM] Install grad-cam: pip install grad-cam")
            self._available = False

    def generate(self, image_source, target_category_idx: int | None = None) -> np.ndarray | None:
        """
        Generate GradCAM heatmap for the given image.

        Args:
            image_source:         path, PIL image, or numpy array
            target_category_idx:  class index to explain (None = predicted class)

        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
            or None if grad-cam not installed.
        """
        if not self._available:
            return None

        model = self.extractor.model
        # Target the last conv block of EfficientNet
        target_layers = [model.backbone.blocks[-1]]

        tensor = self.extractor._load_image(image_source)

        # Wrap the backbone + category head as the target
        class WrappedModel(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x)["category_logits"]

        with self.GradCAM(model=WrappedModel(model), target_layers=target_layers) as cam:
            targets = None
            if target_category_idx is not None:
                targets = [self.ClassifierOutputTarget(target_category_idx)]
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            return grayscale_cam[0]   # (H, W)
