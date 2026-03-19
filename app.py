# app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random

# ── Page config (MUST be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="StyleSense ✨",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:wght@300;400;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #FDF6F0;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #F9E4EC 0%, #EDD6F0 50%, #D6E4F0 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #F0D6E4;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #8B4A6B;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.hero p {
    font-size: 1.1rem;
    color: #A07090;
    font-weight: 300;
    margin: 0;
}

/* ── Section cards ── */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.8rem;
    border: 1px solid #F0D6E4;
    box-shadow: 0 2px 16px rgba(180, 100, 140, 0.07);
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #8B4A6B;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Chips (occasion/style/season) ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.4rem;
}
.chip {
    background: #FDE8F0;
    color: #9B3A6A;
    border: 1.5px solid #F0C0D8;
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
}
.chip.active {
    background: #C4688A;
    color: white;
    border-color: #C4688A;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextArea"] > div > div {
    border-radius: 10px;
    border: 1.5px solid #F0C0D8 !important;
    background: #FFF5F8;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stFileUploader"] label {
    color: #8B4A6B !important;
    font-weight: 600;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #C4688A, #9B5BA8);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 0.65rem 2.5rem;
    font-size: 1rem;
    font-weight: 700;
    font-family: 'Lato', sans-serif;
    letter-spacing: 0.04em;
    transition: all 0.2s;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(196, 104, 138, 0.35);
}

/* ── Score bars ── */
.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.score-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #8B4A6B;
    width: 160px;
    flex-shrink: 0;
}
.score-bar-bg {
    flex: 1;
    height: 8px;
    background: #F0D6E4;
    border-radius: 10px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 10px;
}
.score-val {
    font-size: 0.8rem;
    font-weight: 700;
    width: 36px;
    text-align: right;
    color: #9B3A6A;
}

/* ── Verdict pill ── */
.verdict-good  { background:#D4F0E0; color:#1A6B3A; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }
.verdict-mixed { background:#FFF0D4; color:#7A5000; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }
.verdict-bad   { background:#FFE0E0; color:#8B1A1A; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }

/* ── Outfit item row ── */
.outfit-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #FFF5F8;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    border: 1px solid #F0D6E4;
}
.outfit-item .item-name { font-weight: 600; color: #5A2040; font-size: 0.9rem; }
.outfit-item .item-meta { font-size: 0.75rem; color: #B080A0; margin-left: auto; }

/* ── Alternative suggestion ── */
.alt-card {
    background: #FFF0F5;
    border-left: 4px solid #C4688A;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}
.alt-card .alt-title { font-weight: 700; color: #8B4A6B; font-size: 0.88rem; }
.alt-card .alt-reason { font-size: 0.78rem; color: #A07090; margin-top: 0.2rem; }

/* ── Check row ── */
.check-row {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}
.check-detail { color: #A07090; font-size: 0.78rem; margin-top: 0.15rem; }

/* ── Divider ── */
.pink-divider {
    border: none;
    border-top: 2px solid #F0D6E4;
    margin: 1.5rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #C0A0B0;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: mock recommendation engine ───────────────────────────────────
# Replace these with your real recommender.py / outfit_evaluator.py calls

OUTFIT_TEMPLATES = {
    "college": {
        "casual": [
            {"icon":"👕","type":"Top",      "name":"White Linen Shirt",     "color":"White",  "fabric":"Linen"},
            {"icon":"👖","type":"Bottom",   "name":"Slim Navy Chinos",      "color":"Navy",   "fabric":"Cotton"},
            {"icon":"👟","type":"Footwear", "name":"White Sneakers",        "color":"White",  "fabric":"Canvas"},
            {"icon":"🎒","type":"Accessory","name":"Mini Canvas Tote",      "color":"Beige",  "fabric":"Canvas"},
        ],
        "minimalist": [
            {"icon":"👔","type":"Top",      "name":"Beige Fitted Tee",      "color":"Beige",  "fabric":"Cotton"},
            {"icon":"👖","type":"Bottom",   "name":"Black Straight Jeans",  "color":"Black",  "fabric":"Denim"},
            {"icon":"👟","type":"Footwear", "name":"Clean White Sneakers",  "color":"White",  "fabric":"Leather"},
            {"icon":"⌚","type":"Accessory","name":"Minimal Silver Watch",  "color":"Silver", "fabric":"Metal"},
        ],
    },
    "party": {
        "trendy": [
            {"icon":"👗","type":"Top",      "name":"Satin Slip Dress",      "color":"Blush",  "fabric":"Satin"},
            {"icon":"👠","type":"Footwear", "name":"Strappy Heeled Sandals","color":"Nude",   "fabric":"Leather"},
            {"icon":"👜","type":"Accessory","name":"Mini Clutch Bag",       "color":"Gold",   "fabric":"Faux leather"},
            {"icon":"💍","type":"Accessory","name":"Layered Gold Necklace", "color":"Gold",   "fabric":"Metal"},
        ],
        "casual": [
            {"icon":"👕","type":"Top",      "name":"Sequin Crop Top",       "color":"Rose",   "fabric":"Sequin"},
            {"icon":"👖","type":"Bottom",   "name":"High-waist Satin Skirt","color":"Black",  "fabric":"Satin"},
            {"icon":"👠","type":"Footwear", "name":"Block Heel Mules",      "color":"Black",  "fabric":"Faux leather"},
            {"icon":"👜","type":"Accessory","name":"Chain Shoulder Bag",    "color":"Silver", "fabric":"Metal"},
        ],
    },
    "formal": {
        "classic": [
            {"icon":"👔","type":"Top",      "name":"Crisp White Dress Shirt","color":"White", "fabric":"Cotton"},
            {"icon":"👖","type":"Bottom",   "name":"Tailored Black Trousers","color":"Black", "fabric":"Wool blend"},
            {"icon":"👞","type":"Footwear", "name":"Oxford Leather Shoes",  "color":"Black",  "fabric":"Leather"},
            {"icon":"⌚","type":"Accessory","name":"Classic Leather Belt",  "color":"Black",  "fabric":"Leather"},
        ],
        "minimalist": [
            {"icon":"🥻","type":"Top",      "name":"Structured Blazer",     "color":"Ivory",  "fabric":"Linen"},
            {"icon":"👖","type":"Bottom",   "name":"Slim Dress Trousers",   "color":"Charcoal","fabric":"Wool"},
            {"icon":"👠","type":"Footwear", "name":"Pointed Toe Flats",     "color":"Nude",   "fabric":"Leather"},
            {"icon":"👜","type":"Accessory","name":"Structured Tote",       "color":"Tan",    "fabric":"Leather"},
        ],
    },
}

def get_outfit(occasion, style, season):
    occ_key   = occasion.split("/")[0].strip().lower()
    style_key = style.lower()
    templates = OUTFIT_TEMPLATES.get(occ_key, OUTFIT_TEMPLATES["college"])
    outfit    = templates.get(style_key, list(templates.values())[0])
    return outfit

def evaluate_outfit(outfit, occasion, style, season):
    """Mock evaluation — replace with outfit_evaluator.py calls."""
    random.seed(sum(ord(c) for c in occasion + style + season))

    color_score    = random.randint(65, 95)
    occasion_score = random.randint(60, 95)
    style_score    = random.randint(65, 90)
    season_score   = random.randint(70, 95)

    overall = round(
        0.30 * color_score +
        0.30 * occasion_score +
        0.25 * style_score +
        0.15 * season_score, 1
    )

    verdict = "Good" if overall >= 72 else "Mixed" if overall >= 52 else "Needs Work"

    checks = [
        {"aspect": "Color Harmony",    "pass": color_score >= 65,
         "score": color_score,
         "detail": "Colors complement each other well." if color_score >= 65
                   else "Some color combinations clash."},
        {"aspect": "Occasion Match",   "pass": occasion_score >= 65,
         "score": occasion_score,
         "detail": f"Outfit is appropriate for {occasion}." if occasion_score >= 65
                   else f"Some items don't suit {occasion}."},
        {"aspect": "Style Consistency","pass": style_score >= 65,
         "score": style_score,
         "detail": f"Cohesive {style} aesthetic." if style_score >= 65
                   else "Style feels a little mixed."},
        {"aspect": "Season Fit",       "pass": season_score >= 65,
         "score": season_score,
         "detail": f"Well suited for {season}." if season_score >= 65
                   else f"Some pieces aren't ideal for {season}."},
    ]

    if verdict == "Good":
        explanation = (
            f"✨ Your outfit works beautifully for a **{occasion}** occasion! "
            f"The color palette is harmonious and the {style} aesthetic comes through clearly. "
            f"A great choice for {season}."
        )
    elif verdict == "Mixed":
        explanation = (
            f"💭 This outfit has real potential for **{occasion}** but could use a small tweak. "
            f"The overall vibe is close — just one or two swaps would make it shine."
        )
    else:
        explanation = (
            f"💡 This combination struggles a bit for **{occasion}**. "
            f"The suggestions below will help you build a stronger look."
        )

    alternatives = [
        {"replace": outfit[0]["name"], "with": "A flowy floral blouse",
         "reason": "Adds softness and a more polished touch."},
        {"replace": outfit[2]["name"], "with": "Strappy sandals or block heels",
         "reason": "Elevates the outfit instantly for the occasion."},
    ]

    return {
        "overall": overall,
        "verdict": verdict,
        "checks":  checks,
        "explanation": explanation,
        "alternatives": alternatives,
        "scores": {
            "Color Harmony":    color_score,
            "Occasion Match":   occasion_score,
            "Style Consistency":style_score,
            "Season Fit":       season_score,
        }
    }

def score_color(score):
    if score >= 75: return "#C4688A"
    if score >= 55: return "#E8A86A"
    return "#E87070"


# ══════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>✨ StyleSense</h1>
  <p>Your personal AI outfit advisor — get recommendations, scores & style tips</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────
left, right = st.columns([1, 1.2], gap="large")

# ════════════════════════════
#  LEFT COLUMN — Inputs
# ════════════════════════════
with left:

    # ── Upload image ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📸 Upload Your Clothing (Optional)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a photo of your outfit or a clothing item",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True, caption="Your uploaded item")
        st.success("✅ Image uploaded! We'll factor this into your recommendation.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Occasion ──────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌸 What\'s the Occasion?</div>', unsafe_allow_html=True)
    occasion = st.selectbox(
        "Occasion",
        ["College / Campus", "Party / Night Out", "Formal / Office",
         "Casual / Everyday", "Date Night", "Wedding / Festival"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Style preference ──────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💫 Your Style Vibe</div>', unsafe_allow_html=True)
    style = st.selectbox(
        "Style",
        ["Minimalist", "Casual", "Classic", "Trendy", "Bohemian", "Smart Casual"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Season ────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌤️ Season</div>', unsafe_allow_html=True)
    season = st.radio(
        "Season",
        ["Spring 🌸", "Summer ☀️", "Autumn 🍂", "Winter ❄️"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Extra notes ───────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📝 Anything Else? (Optional)</div>', unsafe_allow_html=True)
    notes = st.text_area(
        "Notes",
        placeholder="e.g. I have warm skin tone, I prefer loose fits, I already own blue jeans...",
        height=90,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────
    analyze = st.button("✨ Analyze My Style")


# ════════════════════════════
#  RIGHT COLUMN — Results
# ════════════════════════════
with right:

    if not analyze:
        # Empty state
        st.markdown("""
        <div style="height:500px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center; text-align:center;
                    color:#C0A0B0;">
          <div style="font-size:4rem; margin-bottom:1rem;">👗</div>
          <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                      color:#C4688A; margin-bottom:0.5rem;">
            Your style analysis will appear here
          </div>
          <div style="font-size:0.9rem; max-width:280px; line-height:1.7;">
            Fill in your occasion, style vibe and season on the left,
            then hit <strong>Analyze My Style</strong> ✨
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Run evaluation ──
        season_clean = season.split(" ")[0]
        outfit  = get_outfit(occasion, style, season_clean)
        result  = evaluate_outfit(outfit, occasion, style, season_clean)

        # ── Verdict header ──
        verdict_class = {
            "Good": "verdict-good",
            "Mixed": "verdict-mixed",
            "Needs Work": "verdict-bad",
        }[result["verdict"]]

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
          <div style="font-family:'Playfair Display',serif; font-size:1.6rem; color:#8B4A6B;">
            Your Outfit Analysis
          </div>
          <span class="{verdict_class}">{result['verdict']}</span>
          <span style="margin-left:auto; font-size:2rem; font-weight:700; color:#C4688A;">
            {result['overall']}<span style="font-size:1rem;color:#C0A0B0;">/100</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Recommended outfit ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">👗 Recommended Outfit</div>', unsafe_allow_html=True)
        for item in outfit:
            st.markdown(f"""
            <div class="outfit-item">
              <span style="font-size:1.3rem">{item['icon']}</span>
              <div>
                <div class="item-name">{item['name']}</div>
                <div style="font-size:0.72rem;color:#B080A0">{item['type']}</div>
              </div>
              <span class="item-meta">{item['color']} · {item['fabric']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Score breakdown ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
        for label, score in result["scores"].items():
            color = score_color(score)
            st.markdown(f"""
            <div class="score-row">
              <span class="score-label">{label}</span>
              <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{score}%;background:{color}"></div>
              </div>
              <span class="score-val">{score}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Suitability checks ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✅ Suitability Checks</div>', unsafe_allow_html=True)
        for check in result["checks"]:
            icon = "✅" if check["pass"] else "❌"
            st.markdown(f"""
            <div class="check-row">
              <span style="font-size:1rem">{icon}</span>
              <div>
                <div style="font-weight:600;color:#8B4A6B;font-size:0.88rem">{check['aspect']}</div>
                <div class="check-detail">{check['detail']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Explanation ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💬 Why This Works</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#5A3050;line-height:1.75;font-size:0.92rem">{result["explanation"]}</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Alternatives ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Better Alternatives</div>', unsafe_allow_html=True)
        for alt in result["alternatives"]:
            st.markdown(f"""
            <div class="alt-card">
              <div class="alt-title">
                Replace <em>{alt['replace']}</em> → <strong>{alt['with']}</strong>
              </div>
              <div class="alt-reason">{alt['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Made with 💗 · StyleSense · Your AI Fashion Advisor
</div>
""", unsafe_allow_html=True)
