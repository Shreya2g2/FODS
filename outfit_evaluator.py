# stylesense/utils/outfit_evaluator.py
"""
Outfit Suitability Evaluator
=============================
Rule-based + CNN-informed evaluation of outfit suitability.

Checks:
  1. Color Harmony        — via ColorHarmonyChecker
  2. Occasion Match       — CNN formality score vs occasion formality requirement
  3. Style Consistency    — CNN category predictions + style rules
  4. Season Appropriateness — fabric + layer analysis

Each check returns:
  - pass (bool)
  - score (0-100)
  - explanation (str)
  - suggestions (list[str])

Final weighted score → overall verdict + rich explanation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import (
    OCCASION_RULES, SEASON_RULES, SCORE_WEIGHTS, CATEGORY_NAMES
)
from utils.color_analyzer import ColorHarmonyChecker, DominantColorExtractor


# ── Data classes ──────────────────────────────────────────────────────────
@dataclass
class ClothingItem:
    """Represents one clothing item with CNN predictions + user metadata."""
    name:          str
    category:      str = "top"
    color_name:    str = "unknown"
    fabric:        str = "cotton"
    formality:     float = 0.5       # 0=casual, 1=formal (from CNN regression)
    image_path:    Optional[str] = None
    cnn_features:  Optional[np.ndarray] = None
    dominant_colors: list = field(default_factory=list)
    season_pred:   str = "summer"


@dataclass
class OutfitEvaluation:
    """Full evaluation result for an outfit."""
    overall_score:   float
    verdict:         str           # "Good" | "Mixed" | "Needs Work"
    checks:          dict          # per-dimension results
    explanation:     str
    suggestions:     list[str]
    alternative_items: list[dict]
    color_palette:   list[str]


# ── Occasion Matcher ──────────────────────────────────────────────────────
class OccasionMatcher:
    """
    Checks whether CNN-predicted formality aligns with occasion requirements.
    Maps formality score (0-1) to categorical level.
    """

    FORMALITY_LEVELS = {
        (0.0, 0.30): "casual",
        (0.30, 0.60): "smart-casual",
        (0.60, 0.80): "semi-formal",
        (0.80, 1.01): "formal",
    }

    def formality_level(self, score: float) -> str:
        for (lo, hi), label in self.FORMALITY_LEVELS.items():
            if lo <= score < hi:
                return label
        return "casual"

    def check(self, items: list[ClothingItem], occasion: str) -> dict:
        occasion_key = self._normalize_occasion(occasion)
        rules = OCCASION_RULES.get(occasion_key, OCCASION_RULES["casual"])

        allowed = rules["allowed_formality"]
        forbidden = rules.get("forbidden_items", [])

        # Average formality across items
        avg_formality = np.mean([item.formality for item in items]) if items else 0.5
        level = self.formality_level(avg_formality)

        score = 100
        issues, positives = [], []

        # Formality match
        if level in allowed:
            positives.append(f"Formality level '{level}' matches {occasion}")
        else:
            score -= 35
            issues.append(f"Formality '{level}' doesn't suit {occasion} (needs: {', '.join(allowed)})")

        # Forbidden items check
        for item in items:
            for forbidden_item in forbidden:
                if forbidden_item.lower() in item.name.lower() or forbidden_item.lower() in item.category.lower():
                    score -= 20
                    issues.append(f"'{item.name}' is typically not appropriate for {occasion}")

        score = max(0, min(100, score))
        return {
            "score": score,
            "pass": score >= 55,
            "formality_detected": level,
            "formality_required": allowed,
            "issues": issues,
            "positives": positives,
            "explanation": self._explain(level, allowed, occasion, score),
        }

    @staticmethod
    def _normalize_occasion(occasion: str) -> str:
        o = occasion.lower()
        if "college" in o or "campus" in o: return "college"
        if "party" in o or "night" in o: return "party"
        if "formal" in o or "office" in o or "work" in o: return "formal"
        if "date" in o: return "date"
        if "wedding" in o or "festival" in o: return "wedding"
        return "casual"

    @staticmethod
    def _explain(level, allowed, occasion, score) -> str:
        if score >= 75:
            return f"The outfit's formality ({level}) is well-suited for a {occasion} setting."
        elif score >= 55:
            return f"The outfit works for {occasion}, though some items may be borderline appropriate."
        else:
            return (f"The formality level ({level}) doesn't match {occasion} requirements "
                    f"({', '.join(allowed)} needed). Consider adjusting key pieces.")


# ── Style Consistency Checker ─────────────────────────────────────────────
class StyleConsistencyChecker:
    """
    Checks if outfit items form a coherent style narrative.
    Uses CNN category distribution + style preference matching.
    """

    # Items that typically don't mix well
    STYLE_CLASHES = [
        ({"sportswear", "hoodie", "sweatpants"}, {"formal shirt", "blazer", "dress pants"}),
        ({"evening gown", "cocktail dress"},      {"sneakers", "flip flops"}),
        ({"suit jacket"},                         {"shorts", "ripped jeans"}),
    ]

    STYLE_RULES = {
        "minimalist":  {"preferred_colors": ["black","white","grey","beige","navy"],
                        "avoid":            ["neon","sequin","print"]},
        "streetwear":  {"preferred_items":  ["hoodie","sneakers","cap","joggers"],
                        "avoid":            ["formal","gown","suit"]},
        "classic":     {"preferred_fabrics":["wool","cotton","linen"],
                        "avoid":            ["sequin","neon","ripped"]},
        "bohemian":    {"preferred_items":  ["flowy","floral","linen","boho"],
                        "avoid":            ["suit","blazer","formal"]},
        "smart-casual":{"preferred_items":  ["chinos","polo","loafers","blazer"],
                        "avoid":            ["sweatpants","flip flops"]},
        "trendy":      {},  # no strict rules — trendy is flexible
    }

    def check(self, items: list[ClothingItem], style_pref: str) -> dict:
        style_key = style_pref.lower().replace(" / ", "-").replace(" ", "-")
        rules = self.STYLE_RULES.get(style_key, {})

        score = 75
        issues, positives = [], []

        # Check style-specific rules
        preferred = rules.get("preferred_items", []) + rules.get("preferred_colors", [])
        avoid     = rules.get("avoid", [])

        for item in items:
            item_str = (item.name + " " + item.color_name + " " + item.fabric).lower()
            if any(p in item_str for p in preferred):
                score += 5
                positives.append(f"'{item.name}' fits {style_pref} style")

            if any(a in item_str for a in avoid):
                score -= 15
                issues.append(f"'{item.name}' feels inconsistent with {style_pref} style")

        # Generic mix check: tops + bottoms should have compatible formality
        tops    = [i for i in items if i.category in ("top", "outerwear")]
        bottoms = [i for i in items if i.category == "bottom"]
        if tops and bottoms:
            form_diff = abs(
                np.mean([t.formality for t in tops]) -
                np.mean([b.formality for b in bottoms])
            )
            if form_diff < 0.2:
                positives.append("Top and bottom have consistent formality")
                score += 5
            elif form_diff > 0.45:
                issues.append("Top and bottom formality levels don't match")
                score -= 15

        score = max(0, min(100, score))
        return {
            "score": score,
            "pass": score >= 55,
            "issues": issues,
            "positives": positives,
            "explanation": self._explain(style_pref, score, issues, positives),
        }

    @staticmethod
    def _explain(style, score, issues, positives) -> str:
        if score >= 75:
            return f"The outfit maintains strong {style} style coherence."
        elif score >= 55:
            pos = positives[0] if positives else "Some elements work"
            return f"Mostly consistent with {style} style. {pos}."
        else:
            iss = issues[0] if issues else "Style conflicts detected"
            return f"Style inconsistency with {style} aesthetic: {iss}."


# ── Season Checker ────────────────────────────────────────────────────────
class SeasonChecker:
    def check(self, items: list[ClothingItem], season: str) -> dict:
        rules = SEASON_RULES.get(season.lower(), SEASON_RULES["summer"])
        preferred = rules["preferred_fabrics"]
        avoid     = rules["avoid_fabrics"]

        score = 80
        issues, positives = [], []

        for item in items:
            fab = item.fabric.lower()
            if any(p in fab for p in preferred):
                positives.append(f"'{item.fabric}' is ideal for {season}")
                score += 5
            if any(a in fab for a in avoid):
                issues.append(f"'{item.fabric}' is too heavy/light for {season}")
                score -= 15

        # CNN season prediction alignment
        season_mismatches = [
            i for i in items
            if i.season_pred and i.season_pred != season.lower()
            and i.season_pred != "unknown"
        ]
        if len(season_mismatches) > len(items) / 2:
            score -= 10
            issues.append(f"Several items seem better suited for another season")

        score = max(0, min(100, score))
        return {
            "score": score,
            "pass": score >= 55,
            "issues": issues,
            "positives": positives,
            "explanation": (
                f"The outfit is {'well' if score >= 75 else 'somewhat' if score >= 55 else 'not well'} "
                f"suited for {season}. "
                + (issues[0] if issues else positives[0] if positives else "")
            ),
        }


# ── Master Evaluator ──────────────────────────────────────────────────────
class OutfitEvaluator:
    """
    Orchestrates all checks and generates the final OutfitEvaluation.
    This is the core engine of StyleSense.
    """

    def __init__(self):
        self.color_extractor   = DominantColorExtractor()
        self.color_checker     = ColorHarmonyChecker()
        self.occasion_matcher  = OccasionMatcher()
        self.style_checker     = StyleConsistencyChecker()
        self.season_checker    = SeasonChecker()

    def evaluate(
        self,
        items:      list[ClothingItem],
        occasion:   str,
        style_pref: str,
        season:     str,
    ) -> OutfitEvaluation:
        """
        Full pipeline:
          1. Extract dominant colors (if images available)
          2. Run all 4 checks
          3. Compute weighted final score
          4. Generate natural language explanation
          5. Generate alternative suggestions
        """

        # ── Step 1: Color extraction ──
        for item in items:
            if not item.dominant_colors and item.image_path:
                try:
                    item.dominant_colors = self.color_extractor.extract(item.image_path)
                except Exception:
                    pass

        # ── Step 2: All checks ──
        color_result   = self.color_checker.check([i.dominant_colors for i in items])
        occasion_result= self.occasion_matcher.check(items, occasion)
        style_result   = self.style_checker.check(items, style_pref)
        season_result  = self.season_checker.check(items, season)

        checks = {
            "color_harmony":          color_result,
            "occasion_fit":           occasion_result,
            "style_consistency":      style_result,
            "season_appropriateness": season_result,
        }

        # ── Step 3: Weighted score ──
        weighted = sum(
            SCORE_WEIGHTS[k] * checks[k]["score"]
            for k in SCORE_WEIGHTS
        )

        # ── Step 4: Verdict ──
        if weighted >= 72:
            verdict = "Good"
        elif weighted >= 50:
            verdict = "Mixed"
        else:
            verdict = "Needs Work"

        # ── Step 5: Explanation ──
        explanation = self._generate_explanation(
            items, occasion, style_pref, season, checks, weighted, verdict
        )

        # ── Step 6: Alternative suggestions ──
        alternatives = self._generate_alternatives(items, checks, occasion, style_pref, season)

        # ── Color palette ──
        palette = list({c["name"] for item in items for c in item.dominant_colors[:1]})

        return OutfitEvaluation(
            overall_score   = round(weighted, 1),
            verdict         = verdict,
            checks          = checks,
            explanation     = explanation,
            suggestions     = self._collect_issues(checks),
            alternative_items = alternatives,
            color_palette   = palette,
        )

    @staticmethod
    def _collect_issues(checks: dict) -> list[str]:
        issues = []
        for check in checks.values():
            issues.extend(check.get("issues", []))
        return issues

    @staticmethod
    def _generate_explanation(items, occasion, style, season, checks, score, verdict) -> str:
        item_names = ", ".join(i.name for i in items[:3])
        color_exp  = checks["color_harmony"]["explanation"]
        occ_exp    = checks["occasion_fit"]["explanation"]

        if verdict == "Good":
            return (
                f"Your outfit ({item_names}) works well overall for a {occasion} occasion. "
                f"{color_exp} {occ_exp} "
                f"The ensemble reflects a coherent {style} aesthetic for {season}."
            )
        elif verdict == "Mixed":
            issues = [c for check in checks.values() for c in check.get("issues", [])]
            issue_str = issues[0] if issues else "some elements could be improved"
            return (
                f"Your outfit ({item_names}) has potential but needs refinement. "
                f"{color_exp} However, {issue_str}. "
                f"With a few adjustments, this could work well for {occasion}."
            )
        else:
            issues = [c for check in checks.values() for c in check.get("issues", [])]
            return (
                f"The outfit ({item_names}) struggles to suit the {occasion} occasion. "
                f"{occ_exp} Additionally: {'; '.join(issues[:2])}. "
                f"See the alternative suggestions below for an improved look."
            )

    def _generate_alternatives(self, items, checks, occasion, style, season) -> list[dict]:
        alts = []
        occ_key = self.occasion_matcher._normalize_occasion(occasion)
        rules = OCCASION_RULES.get(occ_key, OCCASION_RULES["casual"])

        # Color-based suggestions
        if not checks["color_harmony"]["pass"]:
            issues = checks["color_harmony"]["issues"]
            if issues and "clash" in issues[0].lower():
                alts.append({
                    "replace":  "clashing colored item",
                    "with":     "a white, black, or beige neutral",
                    "reason":   "Neutrals resolve color clashes and let other pieces shine",
                    "priority": "high"
                })

        # Occasion-based suggestions
        if not checks["occasion_fit"]["pass"]:
            needed_formality = rules["allowed_formality"][0]
            alts.append({
                "replace":  "current top",
                "with":     self._suggest_formality_item(needed_formality, "top"),
                "reason":   f"Better formality match for {occasion}",
                "priority": "high"
            })

        # Season-based suggestions
        if not checks["season_appropriateness"]["pass"]:
            sea_rules = SEASON_RULES.get(season.lower(), {})
            preferred_fab = sea_rules.get("preferred_fabrics", ["cotton"])[0]
            alts.append({
                "replace":  "current fabric choice",
                "with":     f"a {preferred_fab}-based garment",
                "reason":   f"{preferred_fab.capitalize()} is ideal for {season}",
                "priority": "medium"
            })

        # Generic style upgrade
        if len(alts) == 0:
            alts.append({
                "replace":  "current footwear",
                "with":     "clean minimal sneakers or loafers",
                "reason":   "Footwear elevates the overall look significantly",
                "priority": "low"
            })

        return alts[:3]

    @staticmethod
    def _suggest_formality_item(formality: str, category: str) -> str:
        suggestions = {
            "casual":      {"top": "graphic tee or casual shirt", "bottom": "jeans or chinos"},
            "smart-casual":{"top": "polo or clean button-down", "bottom": "slim chinos"},
            "semi-formal": {"top": "dress shirt or fitted blouse", "bottom": "dress trousers"},
            "formal":      {"top": "formal shirt with blazer", "bottom": "suit trousers"},
        }
        return suggestions.get(formality, {}).get(category, "appropriate garment for the occasion")
