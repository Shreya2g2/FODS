# StyleSense 👗
### Explainable Outfit Recommendation System

> **Suggests ✅ | Evaluates ✅ | Explains ✅ | Improves ✅**

---

## Architecture Overview

```
User Input (occasion + style + image?)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                   CNN PIPELINE                          │
│  EfficientNet-B3 (pretrained + fine-tuned)              │
│  ┌──────────┐  ┌───────────┐  ┌────────────────────┐   │
│  │ Category │  │ Formality │  │ Season Classifier   │   │
│  │ (8-class)│  │ Regressor │  │ (4-class)           │   │
│  └──────────┘  └───────────┘  └────────────────────┘   │
│        └────────────┬───────────────┘                   │
│              1536-d Embedding                           │
└─────────────────────┼───────────────────────────────────┘
                      │
          ┌───────────▼────────────┐
          │  Cosine Similarity     │
          │  Retrieval (Top-K)     │
          └───────────┬────────────┘
                      │
          ┌───────────▼────────────┐
          │   Outfit Assembler     │
          │  (top+bottom+shoes+    │
          │   accessory)           │
          └───────────┬────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │         4-CHECK EVALUATOR          │
    │  ① Color Harmony (K-Means + Rules) │
    │  ② Occasion Match (CNN formality)  │
    │  ③ Style Consistency (rules)       │
    │  ④ Season Fit (fabric analysis)    │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │   EXPLANATION GENERATOR            │
    │   "Outfit works because..."        │
    │   "Issue: colors clash because..." │
    │   "Try X instead of Y because..."  │
    └────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```bash
# Option A: Use your own images
python data/prepare_dataset.py --images_dir data/raw_images

# Option B: Use demo catalog (no images needed)
python data/prepare_dataset.py
```

### 3. (Optional) Fine-tune CNN on your data
```bash
python models/train_cnn.py --epochs 20 --batch_size 32
```
**Training Strategy (Progressive Unfreezing):**
- Phase 1 (epochs 1-5): Freeze backbone, train classification heads only
- Phase 2 (epochs 6-20): Unfreeze last 3 EfficientNet blocks, fine-tune end-to-end

### 4. Start the API server
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Explore in Jupyter
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## API Usage

### Text-based recommendation
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"occasion":"college","style":"minimalist","season":"summer"}'
```

### Image-based recommendation
```bash
curl -X POST http://localhost:8000/analyze-image \
  -F "file=@my_outfit.jpg" \
  -F "occasion=college" \
  -F "style=minimalist" \
  -F "season=summer"
```

### Evaluate a described outfit
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"name":"Red Graphic Tee","category":"top","color":"red","fabric":"cotton"},
      {"name":"Black Ripped Jeans","category":"bottom","color":"black","fabric":"denim"},
      {"name":"White Sneakers","category":"shoes","color":"white","fabric":"canvas"}
    ],
    "occasion": "college",
    "style": "streetwear",
    "season": "summer"
  }'
```

**Example Response:**
```json
{
  "overall_score": 74.5,
  "verdict": "Good",
  "checks": {
    "color_harmony":    {"score": 80, "passed": true,  "explanation": "Red+black+white is a classic bold combo."},
    "occasion_fit":     {"score": 75, "passed": true,  "explanation": "Casual formality suits college well."},
    "style_consistency":{"score": 70, "passed": true,  "explanation": "Cohesive streetwear aesthetic."},
    "season_appropriateness": {"score": 80, "passed": true, "explanation": "Cotton/canvas ideal for summer."}
  },
  "explanation": "Your outfit (Red Graphic Tee, Black Ripped Jeans, White Sneakers) works well for a college occasion. The red-black-white palette is a classic streetwear combination. The casual formality level matches campus settings perfectly.",
  "alternatives": [
    {"replace":"Red Graphic Tee","with":"White or black graphic tee","reason":"Neutral tee gives more outfit flexibility and reduces clash risk"}
  ]
}
```

---

## Key Technologies

| Module | Technology | Purpose |
|---|---|---|
| Feature Extraction | EfficientNet-B3 (timm) | Visual embeddings |
| Multi-task heads | PyTorch nn.Linear | Category / formality / season |
| Color Analysis | K-Means + Itten rules | Harmony checking |
| Similarity Search | Cosine distance (scipy) | Outfit retrieval |
| Explainability | GradCAM (grad-cam) | Visual attention maps |
| API | FastAPI | REST endpoints |
| Data augmentation | torchvision.transforms | Training robustness |

---

## Datasets You Can Use

| Dataset | Items | Notes |
|---|---|---|
| **Fashion Product Images** (Kaggle) | 44,000+ | Good for starting |
| **DeepFashion** | 800,000 | Best quality |
| **Polyvore Outfits** | 21,000 outfits | Has outfit labels |
| **iMaterialist** | 1M+ | Has fabric labels |

---

## Your USP — Explainability

Unlike other fashion apps that just output images, StyleSense:

1. **Explains WHY** an outfit works or doesn't  
2. **Scores each dimension** (color / occasion / style / season)  
3. **Suggests specific replacements** with reasons  
4. **Uses CNN attention maps** (GradCAM) to show which parts of a garment it focused on
