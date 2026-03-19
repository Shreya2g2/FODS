# stylesense/utils/color_analyzer.py
"""
Color Analysis Module
=====================
1. Extract dominant colors from clothing images (using K-Means on pixel RGB)
2. Map raw RGB → named color groups (warm, cool, neutral, earth)
3. Check color harmony between outfit items (Itten's color theory)
4. Generate color harmony score (0-100)

No deep learning here — pure computer vision + color theory rules.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import colorsys

try:
    from colorthief import ColorThief
    COLORTHIEF_AVAILABLE = True
except ImportError:
    COLORTHIEF_AVAILABLE = False

from config import (
    N_DOMINANT_COLORS, ANALOGOUS_FAMILIES,
    COMPLEMENTARY_PAIRS, CLASHING_PAIRS, COLOR_PALETTE_SIZE
)


# ── Named Color Mapping ───────────────────────────────────────────────────
# Maps approximate RGB ranges → color names
# Extended set for fashion

NAMED_COLORS = {
    "white":      (255, 255, 255),
    "black":      (10,  10,  10),
    "grey":       (128, 128, 128),
    "red":        (200, 30,  30),
    "coral":      (255, 100, 80),
    "orange":     (230, 120, 30),
    "yellow":     (240, 210, 30),
    "gold":       (200, 160, 50),
    "green":      (50,  150, 50),
    "olive":      (100, 120, 40),
    "teal":       (30,  150, 140),
    "mint":       (150, 220, 180),
    "blue":       (40,  80,  200),
    "navy":       (20,  30,  100),
    "sky-blue":   (100, 180, 230),
    "purple":     (130, 40,  180),
    "lavender":   (190, 150, 220),
    "pink":       (240, 100, 160),
    "blush":      (240, 180, 180),
    "maroon":     (120, 20,  40),
    "brown":      (120, 70,  30),
    "tan":        (190, 150, 100),
    "beige":      (220, 200, 170),
    "cream":      (240, 230, 200),
    "khaki":      (180, 170, 100),
    "rust":       (180, 80,  30),
    "charcoal":   (55,  55,  55),
    "off-white":  (240, 238, 228),
}

def _rgb_distance(c1, c2) -> float:
    return sum((a-b)**2 for a,b in zip(c1,c2)) ** 0.5

def rgb_to_name(rgb: tuple) -> str:
    """Find closest named color to an (R, G, B) tuple."""
    closest = min(NAMED_COLORS.items(), key=lambda x: _rgb_distance(rgb, x[1]))
    return closest[0]

def rgb_to_hsl(rgb: tuple) -> tuple:
    """Convert RGB (0-255) → HSL (hue°, sat%, light%)."""
    r, g, b = [x/255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return round(h * 360, 1), round(s * 100, 1), round(l * 100, 1)

def get_color_family(color_name: str) -> str:
    """Return the family (warm/cool/neutral/earth) for a named color."""
    for family, members in ANALOGOUS_FAMILIES.items():
        if color_name in members:
            return family
    return "unknown"


# ── Dominant Color Extraction ─────────────────────────────────────────────
class DominantColorExtractor:
    """
    Extract dominant colors from a clothing image.
    Strategy:
      1. Remove background (assume center-crop is clothing)
      2. K-Means clustering on pixel RGB values
      3. Return top-N clusters by size
    """

    def __init__(self, n_colors: int = N_DOMINANT_COLORS):
        self.n_colors = n_colors

    def extract(self, image_source) -> list[dict]:
        """
        Returns list of dominant colors:
        [{"rgb": (R,G,B), "name": "navy", "family": "cool", "proportion": 0.45}, ...]
        """
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source.astype("uint8")).convert("RGB")
        elif isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        else:
            raise ValueError(f"Unsupported type: {type(image_source)}")

        # Center crop to reduce background influence
        w, h = img.size
        margin = 0.15
        img = img.crop((w*margin, h*margin, w*(1-margin), h*(1-margin)))

        # Resize for speed
        img = img.resize((150, 150))

        # K-Means on pixel values
        pixels = np.array(img).reshape(-1, 3).astype(np.float32)

        # Filter near-white pixels (background)
        mask = ~((pixels > 230).all(axis=1))
        pixels = pixels[mask]

        if len(pixels) < 10:
            return [{"rgb": (128,128,128), "name": "grey", "family": "neutral", "proportion": 1.0}]

        colors = self._kmeans(pixels, self.n_colors)
        total = sum(c["count"] for c in colors)
        result = []
        for c in sorted(colors, key=lambda x: -x["count"]):
            rgb = tuple(int(v) for v in c["center"])
            name = rgb_to_name(rgb)
            result.append({
                "rgb":        rgb,
                "name":       name,
                "family":     get_color_family(name),
                "proportion": round(c["count"] / total, 3),
                "hex":        "#{:02X}{:02X}{:02X}".format(*rgb),
                "hsl":        rgb_to_hsl(rgb),
            })
        return result

    @staticmethod
    def _kmeans(pixels: np.ndarray, k: int, max_iter: int = 20) -> list:
        """Lightweight K-Means (avoids sklearn import for this module)."""
        np.random.seed(42)
        idx = np.random.choice(len(pixels), k, replace=False)
        centers = pixels[idx].copy()

        for _ in range(max_iter):
            dists  = np.linalg.norm(pixels[:, None] - centers[None], axis=2)  # (N, k)
            labels = dists.argmin(axis=1)
            new_centers = np.array([
                pixels[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                for j in range(k)
            ])
            if np.allclose(centers, new_centers, atol=1.0):
                break
            centers = new_centers

        return [{"center": centers[j], "count": int((labels == j).sum())} for j in range(k)]


# ── Color Harmony Checker ─────────────────────────────────────────────────
class ColorHarmonyChecker:
    """
    Evaluates color harmony across an outfit (list of clothing items).

    Rules implemented:
      1. Neutral anchor rule   — every outfit should have ≥1 neutral
      2. 3-color limit         — max 3 distinct non-neutral colors
      3. Complementary bonus   — complementary pairs score higher
      4. Clash penalty         — clashing pairs score lower
      5. Analogous bonus       — items from same family score higher
      6. Tone consistency      — all warm, all cool, or neutral mix
    """

    def check(self, item_colors: list[list[dict]]) -> dict:
        """
        Args:
            item_colors: list of dominant-color results per outfit item
                         e.g. [[{"name":"navy",...}, ...], [{"name":"beige",...},...]]

        Returns:
            {
              "score": 0-100,
              "pass": True/False,
              "primary_colors": ["navy", "beige", "white"],
              "families": ["cool", "neutral"],
              "issues": [...],
              "positives": [...],
              "explanation": "..."
            }
        """
        # Extract primary (most dominant) color per item
        primaries = []
        for item in item_colors:
            if item:
                primaries.append(item[0])   # highest proportion

        if not primaries:
            return {"score": 50, "pass": True, "explanation": "No color data available."}

        names   = [c["name"]   for c in primaries]
        families = [c["family"] for c in primaries]

        score    = 70   # base score
        issues   = []
        positives = []

        # ── Rule 1: Neutral anchor ──
        neutrals = [n for n in names if get_color_family(n) == "neutral"]
        if neutrals:
            score += 10
            positives.append(f"Good neutral anchor ({', '.join(neutrals)})")
        else:
            score -= 10
            issues.append("No neutral anchor — all saturated colors can overwhelm")

        # ── Rule 2: Color count ──
        non_neutral = [n for n in names if get_color_family(n) != "neutral"]
        if len(set(non_neutral)) <= 2:
            score += 10
            positives.append("Focused color palette (≤2 accent colors)")
        elif len(set(non_neutral)) == 3:
            positives.append("Three accent colors — can work if harmonious")
        else:
            score -= 15
            issues.append(f"Too many colors ({len(set(non_neutral))} accents) — consider simplifying")

        # ── Rule 3: Complementary pairs ──
        for c1, c2 in COMPLEMENTARY_PAIRS:
            if c1 in names and c2 in names:
                score += 8
                positives.append(f"Complementary pair: {c1} + {c2} ✓")

        # ── Rule 4: Clashing pairs ──
        for c1, c2 in CLASHING_PAIRS:
            if c1 in names and c2 in names:
                score -= 20
                issues.append(f"Clashing colors: {c1} + {c2}")

        # ── Rule 5: Analogous family ──
        unique_families = set(f for f in families if f != "neutral" and f != "unknown")
        if len(unique_families) == 1:
            score += 8
            positives.append(f"Cohesive {list(unique_families)[0]} color family")
        elif len(unique_families) == 0:
            score += 5
            positives.append("All neutrals — classic and safe")

        score = max(0, min(100, score))
        return {
            "score":          score,
            "pass":           score >= 55,
            "primary_colors": names,
            "families":       list(set(families)),
            "issues":         issues,
            "positives":      positives,
            "explanation":    self._explain(names, families, score, issues, positives),
        }

    @staticmethod
    def _explain(names, families, score, issues, positives) -> str:
        color_str = ", ".join(names[:3])
        if score >= 75:
            base = f"The colors ({color_str}) work well together."
        elif score >= 55:
            base = f"The colors ({color_str}) are acceptable but have room for improvement."
        else:
            base = f"The color combination ({color_str}) has significant harmony issues."

        detail = ""
        if positives: detail += " " + positives[0] + "."
        if issues:    detail += " Issue: " + issues[0] + "."
        return base + detail
