# stylesense/api/main.py
"""
StyleSense FastAPI Backend
===========================
Endpoints:
  POST /recommend       → text-based recommendation (no image)
  POST /analyze-image   → upload image → CNN analysis + recommendation
  POST /evaluate        → evaluate a user-described outfit
  GET  /health          → health check

Run:
    uvicorn api.main:app --reload --port 8000
"""

import io
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import numpy as np
from PIL import Image

from utils.recommender import OutfitRecommender
from utils.outfit_evaluator import OutfitEvaluator, ClothingItem

# Lazy-load CNN (heavy import)
_cnn_extractor = None
def get_cnn():
    global _cnn_extractor
    if _cnn_extractor is None:
        try:
            from models.cnn_extractor import ClothingCNNExtractor
            _cnn_extractor = ClothingCNNExtractor()
        except Exception as e:
            print(f"[CNN] Could not load: {e}. Image analysis disabled.")
    return _cnn_extractor


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="StyleSense API",
    description="Explainable Outfit Recommendation System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
recommender = OutfitRecommender()
evaluator   = OutfitEvaluator()


# ── Request/Response Schemas ──────────────────────────────────────────────
class RecommendRequest(BaseModel):
    occasion:   str = "college"
    style:      str = "minimalist"
    season:     str = "summer"
    n_outfits:  int = 3

class OutfitItemInput(BaseModel):
    name:     str
    category: str = "top"
    color:    str = "unknown"
    fabric:   str = "cotton"

class EvaluateRequest(BaseModel):
    items:    list[OutfitItemInput]
    occasion: str = "college"
    style:    str = "minimalist"
    season:   str = "summer"

class CheckResult(BaseModel):
    score:       float
    passed:      bool
    explanation: str
    issues:      list[str]
    positives:   list[str]

class EvaluationResponse(BaseModel):
    overall_score:  float
    verdict:        str
    checks:         dict
    explanation:    str
    suggestions:    list[str]
    alternatives:   list[dict]
    color_palette:  list[str]

class RecommendationResponse(BaseModel):
    outfits:    list[list[dict]]
    evaluations:list[EvaluationResponse]


# ── Helper ────────────────────────────────────────────────────────────────
def items_to_response(items: list[ClothingItem]) -> list[dict]:
    return [
        {
            "name":      i.name,
            "category":  i.category,
            "color":     i.color_name,
            "fabric":    i.fabric,
            "formality": i.formality,
        }
        for i in items
    ]

def eval_to_response(eval_result) -> dict:
    checks_out = {}
    for k, v in eval_result.checks.items():
        checks_out[k] = {
            "score":       v.get("score", 0),
            "passed":      v.get("pass", False),
            "explanation": v.get("explanation", ""),
            "issues":      v.get("issues", []),
            "positives":   v.get("positives", []),
        }
    return {
        "overall_score": eval_result.overall_score,
        "verdict":       eval_result.verdict,
        "checks":        checks_out,
        "explanation":   eval_result.explanation,
        "suggestions":   eval_result.suggestions,
        "alternatives":  eval_result.alternative_items,
        "color_palette": eval_result.color_palette,
    }


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/recommend", response_model=dict)
def recommend(req: RecommendRequest):
    """
    Text-based outfit recommendation.
    No image needed — uses catalog + rules.
    """
    outfits = recommender.recommend(
        occasion  = req.occasion,
        style     = req.style,
        season    = req.season,
        n_outfits = req.n_outfits,
    )

    results = []
    for outfit_items in outfits[:req.n_outfits]:
        eval_result = evaluator.evaluate(
            items     = outfit_items,
            occasion  = req.occasion,
            style_pref= req.style,
            season    = req.season,
        )
        results.append({
            "items":      items_to_response(outfit_items),
            "evaluation": eval_to_response(eval_result),
        })

    return {"outfits": results, "count": len(results)}


@app.post("/analyze-image", response_model=dict)
async def analyze_image(
    file:     UploadFile = File(...),
    occasion: str = "college",
    style:    str = "minimalist",
    season:   str = "summer",
):
    """
    Upload a clothing image → CNN analysis + outfit recommendations.
    The CNN extracts:
      - Category prediction (top/bottom/shoes/etc)
      - Formality score
      - Season prediction
      - 1536-d embedding for similarity search
    """
    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # CNN inference
    cnn = get_cnn()
    cnn_preds = None
    query_embedding = None

    if cnn:
        cnn_preds = cnn.classify(pil_image)
        query_embedding = cnn_preds["embedding"]

    # Build ClothingItem from CNN predictions
    if cnn_preds:
        uploaded_item = ClothingItem(
            name      = f"Your {cnn_preds['category'].title()}",
            category  = cnn_preds["category"],
            color_name= "detected",
            formality = cnn_preds["formality"],
            season_pred = cnn_preds["season"],
        )
    else:
        uploaded_item = ClothingItem(name="Uploaded Item", category="top")

    # Get complementary recommendations
    outfits = recommender.recommend(
        occasion        = occasion,
        style           = style,
        season          = season,
        query_embedding = query_embedding,
        n_outfits       = 3,
    )

    # Prepend uploaded item to each outfit
    results = []
    for outfit_items in outfits:
        # Replace same-category item with uploaded item
        outfit = [i for i in outfit_items if i.category != uploaded_item.category]
        outfit.insert(0, uploaded_item)

        eval_result = evaluator.evaluate(
            items      = outfit,
            occasion   = occasion,
            style_pref = style,
            season     = season,
        )
        results.append({
            "items":      items_to_response(outfit),
            "evaluation": eval_to_response(eval_result),
        })

    response = {
        "cnn_analysis": {
            "category":      cnn_preds["category"] if cnn_preds else "unknown",
            "formality":     cnn_preds["formality"] if cnn_preds else 0.5,
            "season_match":  cnn_preds["season"] if cnn_preds else "unknown",
            "category_probs":cnn_preds["category_probs"] if cnn_preds else {},
        } if cnn_preds else {"note": "CNN not loaded, install torch+timm"},
        "outfits": results,
        "count": len(results),
    }

    return response


@app.post("/evaluate", response_model=dict)
def evaluate_outfit(req: EvaluateRequest):
    """
    Evaluate a user-described outfit (no image needed).
    User describes items by name + category + color + fabric.
    """
    items = [
        ClothingItem(
            name      = i.name,
            category  = i.category,
            color_name= i.color,
            fabric    = i.fabric,
        )
        for i in req.items
    ]

    result = evaluator.evaluate(
        items      = items,
        occasion   = req.occasion,
        style_pref = req.style,
        season     = req.season,
    )

    return eval_to_response(result)
