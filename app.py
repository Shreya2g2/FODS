# app.py
import streamlit as st
import numpy as np
from PIL import Image
import random
import colorsys

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="ELVA 🌸",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #FDF6F0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #F9E4EC 0%, #EDD6F0 50%, #D6E4F0 100%);
    border-radius: 20px;
    padding: 2.5rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #F0D6E4;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #8B4A6B;
    margin-bottom: 0.4rem;
}
.hero p { font-size: 1rem; color: #A07090; font-weight: 300; margin: 0; }

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.6rem;
    border: 1px solid #F0D6E4;
    box-shadow: 0 2px 16px rgba(180,100,140,0.07);
    margin-bottom: 1.4rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #8B4A6B;
    margin-bottom: 1rem;
}

.color-detected {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #FFF5F8;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    border: 1px solid #F0D6E4;
    margin-top: 0.75rem;
}
.swatch {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: 2px solid #F0D6E4;
    flex-shrink: 0;
}
.color-name { font-weight: 700; color: #8B4A6B; font-size: 0.95rem; }
.color-sub  { font-size: 0.75rem; color: #B080A0; margin-top: 0.1rem; }

.pairing-badge {
    display: inline-block;
    background: #FDE8F0;
    color: #9B3A6A;
    border: 1.5px solid #F0C0D8;
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0.2rem;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #C4688A, #9B5BA8);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 0.65rem 2.5rem;
    font-size: 1rem;
    font-weight: 700;
    font-family: 'Lato', sans-serif;
    width: 100%;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(196,104,138,0.35);
}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stFileUploader"] label {
    color: #8B4A6B !important;
    font-weight: 600;
}

.score-row { display:flex; align-items:center; gap:1rem; margin-bottom:0.75rem; }
.score-label { font-size:0.85rem; font-weight:600; color:#8B4A6B; width:160px; flex-shrink:0; }
.score-bar-bg { flex:1; height:8px; background:#F0D6E4; border-radius:10px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:10px; }
.score-val { font-size:0.8rem; font-weight:700; width:36px; text-align:right; color:#9B3A6A; }

.verdict-good  { background:#D4F0E0; color:#1A6B3A; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }
.verdict-mixed { background:#FFF0D4; color:#7A5000; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }
.verdict-bad   { background:#FFE0E0; color:#8B1A1A; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }

.outfit-item {
    display:flex; align-items:center; gap:0.75rem;
    background:#FFF5F8; border-radius:10px;
    padding:0.6rem 1rem; margin-bottom:0.5rem;
    border:1px solid #F0D6E4;
}
.uploaded-item {
    display:flex; align-items:center; gap:0.75rem;
    background: linear-gradient(135deg,#FDE8F0,#EDD6F0);
    border-radius:10px; padding:0.6rem 1rem;
    margin-bottom:0.5rem;
    border:2px solid #C4688A;
}

.alt-card {
    background:#FFF0F5; border-left:4px solid #C4688A;
    border-radius:0 10px 10px 0;
    padding:0.75rem 1rem; margin-bottom:0.6rem;
}
.alt-title { font-weight:700; color:#8B4A6B; font-size:0.88rem; }
.alt-reason { font-size:0.78rem; color:#A07090; margin-top:0.2rem; }

.check-row { display:flex; gap:0.6rem; align-items:flex-start; margin-bottom:0.6rem; }
.check-detail { color:#A07090; font-size:0.78rem; margin-top:0.15rem; }

.footer { text-align:center; color:#C0A0B0; font-size:0.78rem; margin-top:3rem; padding:1rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  IMAGE COLOR DETECTION ENGINE
# ══════════════════════════════════════════════════════════

def get_dominant_color(pil_img):
    """Extract dominant RGB using K-Means on pixels (no sklearn needed)."""
    img    = pil_img.convert("RGB").resize((80, 80))
    pixels = np.array(img).reshape(-1, 3).astype(float)
    # Remove near-white background
    mask   = ~((pixels > 220).all(axis=1))
    pixels = pixels[mask]
    if len(pixels) < 10:
        return (200, 180, 200)
    np.random.seed(42)
    centers = pixels[np.random.choice(len(pixels), 3, replace=False)].copy()
    for _ in range(15):
        dists  = np.linalg.norm(pixels[:,None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        new_c  = np.array([
            pixels[labels==j].mean(axis=0) if (labels==j).any() else centers[j]
            for j in range(3)
        ])
        if np.allclose(centers, new_c, atol=1): break
        centers = new_c
    counts  = [(labels==j).sum() for j in range(3)]
    dominant= centers[np.argmax(counts)]
    return tuple(int(v) for v in dominant)


def rgb_to_color_info(rgb):
    """Map RGB → named fashion color with family."""
    r, g, b = rgb
    h, s, v  = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h_deg    = h * 360

    if v < 0.25:
        name, family = "Black", "neutral"
    elif v > 0.88 and s < 0.12:
        name, family = "White", "neutral"
    elif s < 0.12:
        name, family = ("Light Grey" if v > 0.7 else "Charcoal"), "neutral"
    elif s < 0.25 and 15 < h_deg < 55:
        name, family = ("Beige" if v > 0.75 else "Tan"), "neutral"
    elif h_deg < 15 or h_deg >= 345:
        name, family = ("Red" if s > 0.5 else "Rose"), "warm"
    elif h_deg < 30:
        name, family = ("Orange" if s > 0.6 else "Rust"), "warm"
    elif h_deg < 55:
        name, family = ("Yellow" if s > 0.6 else "Gold"), "warm"
    elif h_deg < 80:
        name, family = "Yellow-Green", "warm"
    elif h_deg < 150:
        name, family = ("Olive Green" if s < 0.5 else "Green"), "cool"
    elif h_deg < 185:
        name, family = "Teal", "cool"
    elif h_deg < 225:
        name, family = ("Sky Blue" if v > 0.75 else "Blue"), "cool"
    elif h_deg < 260:
        name, family = ("Navy" if v < 0.4 else "Periwinkle"), "cool"
    elif h_deg < 290:
        name, family = ("Purple" if s > 0.4 else "Lavender"), "cool"
    elif h_deg < 320:
        name, family = ("Pink" if v > 0.7 else "Mauve"), "warm"
    else:
        name, family = ("Blush" if v > 0.8 else "Dusty Rose"), "warm"

    # Pastel or dark prefix
    if s < 0.35 and v > 0.82 and family != "neutral":
        name = "Pastel " + name
    elif v < 0.35 and family != "neutral":
        name = "Dark " + name

    return {"name": name, "family": family, "hex": f"#{r:02X}{g:02X}{b:02X}"}


# ── Color pairing rules ───────────────────────────────────
PAIRINGS = {
    "Black":       {"pairs":["White","Camel","Red","Blush","Cream"],       "avoid":["Dark Navy","Dark Brown"]},
    "White":       {"pairs":["Navy","Black","Camel","Pastel Blue","Red"],  "avoid":["Cream","Off-white"]},
    "Beige":       {"pairs":["White","Brown","Rust","Olive Green","Navy"], "avoid":["Yellow","Orange"]},
    "Tan":         {"pairs":["White","Black","Navy","Burgundy","Olive"],   "avoid":["Orange","Bright Yellow"]},
    "Light Grey":  {"pairs":["Blush","Navy","White","Lavender","Burgundy"],"avoid":["Beige","Bright Yellow"]},
    "Charcoal":    {"pairs":["White","Blush","Sky Blue","Camel","Mint"],   "avoid":["Dark Navy","Black"]},
    "Red":         {"pairs":["White","Black","Navy","Camel","Denim"],      "avoid":["Orange","Pink","Rust"]},
    "Rose":        {"pairs":["White","Grey","Navy","Denim","Camel"],       "avoid":["Orange","Red"]},
    "Blush":       {"pairs":["White","Navy","Grey","Camel","Burgundy"],    "avoid":["Orange","Yellow"]},
    "Dusty Rose":  {"pairs":["Grey","White","Navy","Sage Green","Camel"],  "avoid":["Bright Red","Orange"]},
    "Pink":        {"pairs":["White","Grey","Navy","Lavender","Camel"],    "avoid":["Red","Orange"]},
    "Mauve":       {"pairs":["White","Grey","Camel","Dusty Rose","Navy"],  "avoid":["Orange","Bright Red"]},
    "Orange":      {"pairs":["White","Navy","Denim","Olive","Brown"],      "avoid":["Red","Pink","Purple"]},
    "Rust":        {"pairs":["Camel","Cream","Olive Green","Denim","White"],"avoid":["Orange","Pink"]},
    "Gold":        {"pairs":["White","Black","Navy","Burgundy","Olive"],   "avoid":["Yellow","Orange"]},
    "Sky Blue":    {"pairs":["White","Navy","Grey","Camel","Blush"],       "avoid":["Bright Green","Purple"]},
    "Blue":        {"pairs":["White","Grey","Camel","Black","Blush"],      "avoid":["Bright Green","Orange"]},
    "Navy":        {"pairs":["White","Camel","Blush","Grey","Red"],        "avoid":["Black","Dark Blue"]},
    "Periwinkle":  {"pairs":["White","Grey","Blush","Camel","Lavender"],   "avoid":["Green","Orange"]},
    "Teal":        {"pairs":["White","Grey","Coral","Camel","Navy"],       "avoid":["Green","Purple"]},
    "Green":       {"pairs":["White","Camel","Denim","Brown","Cream"],     "avoid":["Red","Orange","Purple"]},
    "Olive Green": {"pairs":["White","Camel","Rust","Denim","Brown"],      "avoid":["Bright Green","Pink"]},
    "Purple":      {"pairs":["White","Grey","Black","Camel","Blush"],      "avoid":["Orange","Red","Green"]},
    "Lavender":    {"pairs":["White","Grey","Blush","Navy","Camel"],       "avoid":["Orange","Bright Yellow"]},
    "Pastel Blue": {"pairs":["White","Grey","Blush","Camel","Lavender"],   "avoid":["Bright Colors","Neon"]},
    "Pastel Pink": {"pairs":["White","Grey","Navy","Lavender","Camel"],    "avoid":["Red","Orange"]},
    "Pastel Green":{"pairs":["White","Beige","Camel","Blush","Cream"],     "avoid":["Bright Colors","Neon"]},
}

def get_pairings(color_info):
    name = color_info["name"]
    # exact
    if name in PAIRINGS:
        return PAIRINGS[name]
    # partial
    for key in PAIRINGS:
        if key.lower() in name.lower() or name.lower() in key.lower():
            return PAIRINGS[key]
    # fallback by family
    if color_info["family"] == "cool":
        return {"pairs":["White","Camel","Grey","Blush","Navy"],"avoid":["Bright warm colors"]}
    elif color_info["family"] == "warm":
        return {"pairs":["White","Navy","Grey","Denim","Camel"],"avoid":["Clashing warm tones"]}
    return {"pairs":["White","Black","Navy","Camel","Blush"],"avoid":["Neon colors"]}


def detect_garment_type(pil_img):
    w, h = pil_img.size
    ratio = h / w
    if ratio > 1.6: return "dress / full outfit"
    elif ratio > 1.1: return "top / shirt"
    else: return "bottom / skirt"


def build_outfit(color_info, garment_type, occasion, style, season):
    """Build a complete outfit around the uploaded item."""
    pairs = get_pairings(color_info)["pairs"]
    p1 = pairs[0] if len(pairs) > 0 else "White"
    p2 = pairs[1] if len(pairs) > 1 else "Navy"
    p3 = pairs[2] if len(pairs) > 2 else "Grey"

    is_top    = "top" in garment_type or "shirt" in garment_type
    is_bottom = "bottom" in garment_type or "skirt" in garment_type
    occ       = occasion.lower()
    is_formal = "formal" in occ or "office" in occ
    is_party  = "party" in occ or "night" in occ
    is_date   = "date" in occ
    is_college= "college" in occ or "campus" in occ

    label = color_info["name"] + " " + (
        "Top" if is_top else "Skirt / Bottom" if is_bottom else "Dress"
    )

    outfit = [{"icon":"⭐","type":"Your Uploaded Item","name":label,
               "color":color_info["name"],"fabric":"Your piece","uploaded":True}]

    if is_top or "dress" in garment_type:
        if is_formal:
            outfit += [
                {"icon":"👖","type":"Bottom",    "name":f"{p1} Tailored Trousers",  "color":p1,"fabric":"Wool blend","uploaded":False},
                {"icon":"👠","type":"Footwear",  "name":"Pointed-Toe Court Heels",  "color":"Nude","fabric":"Leather","uploaded":False},
                {"icon":"👜","type":"Bag",       "name":f"{p2} Structured Tote",    "color":p2,"fabric":"Leather","uploaded":False},
            ]
        elif is_party:
            outfit += [
                {"icon":"👖","type":"Bottom",    "name":f"{p1} Satin Mini Skirt",   "color":p1,"fabric":"Satin","uploaded":False},
                {"icon":"👠","type":"Footwear",  "name":"Strappy Heeled Sandals",   "color":"Gold","fabric":"Metallic","uploaded":False},
                {"icon":"👜","type":"Bag",       "name":"Gold Mini Clutch",         "color":"Gold","fabric":"Faux leather","uploaded":False},
            ]
        elif is_date:
            outfit += [
                {"icon":"👖","type":"Bottom",    "name":f"{p2} High-Waist Jeans",   "color":p2,"fabric":"Denim","uploaded":False},
                {"icon":"👠","type":"Footwear",  "name":"Nude Block Heel Mules",    "color":"Nude","fabric":"Leather","uploaded":False},
                {"icon":"👜","type":"Bag",       "name":f"{p1} Mini Shoulder Bag",  "color":p1,"fabric":"Leather","uploaded":False},
            ]
        elif is_college:
            outfit += [
                {"icon":"👖","type":"Bottom",    "name":f"{p1} Slim Jeans",         "color":p1,"fabric":"Denim","uploaded":False},
                {"icon":"👟","type":"Footwear",  "name":"White Sneakers",           "color":"White","fabric":"Canvas","uploaded":False},
                {"icon":"🎒","type":"Bag",       "name":f"{p2} Canvas Backpack",    "color":p2,"fabric":"Canvas","uploaded":False},
            ]
        else:
            outfit += [
                {"icon":"👖","type":"Bottom",    "name":f"{p1} Casual Shorts",      "color":p1,"fabric":"Cotton","uploaded":False},
                {"icon":"👟","type":"Footwear",  "name":"Slide Sandals",            "color":"Beige","fabric":"Leather","uploaded":False},
                {"icon":"🕶️","type":"Accessory", "name":f"{p3} Sunglasses",        "color":p3,"fabric":"Acetate","uploaded":False},
            ]
    else:  # is_bottom
        if is_formal:
            outfit += [
                {"icon":"👔","type":"Top",       "name":f"{p1} Fitted Silk Blouse", "color":p1,"fabric":"Silk","uploaded":False},
                {"icon":"👠","type":"Footwear",  "name":"Court Heels",              "color":"Nude","fabric":"Leather","uploaded":False},
                {"icon":"⌚","type":"Accessory", "name":"Delicate Gold Watch",      "color":"Gold","fabric":"Metal","uploaded":False},
            ]
        else:
            outfit += [
                {"icon":"👕","type":"Top",       "name":f"{p1} Fitted Tee",         "color":p1,"fabric":"Cotton","uploaded":False},
                {"icon":"👟","type":"Footwear",  "name":f"{p3} Clean Sneakers",     "color":p3,"fabric":"Canvas","uploaded":False},
                {"icon":"👜","type":"Bag",       "name":f"{p2} Crossbody Bag",      "color":p2,"fabric":"Leather","uploaded":False},
            ]
    return outfit


def evaluate(color_info, occasion, style, season):
    seed = sum(ord(c) for c in color_info["name"] + occasion + style)
    random.seed(seed)
    cs = random.randint(72, 96)
    os = random.randint(68, 95)
    ss = random.randint(65, 92)
    seas = random.randint(70, 95)
    overall = round(0.30*cs + 0.30*os + 0.25*ss + 0.15*seas, 1)
    verdict = "Good" if overall >= 72 else "Mixed" if overall >= 52 else "Needs Work"
    pairs = get_pairings(color_info)
    checks = [
        {"aspect":"Color Harmony",    "pass":cs>=65,"score":cs,
         "detail":f"{color_info['name']} pairs well with {pairs['pairs'][0]} & {pairs['pairs'][1]}."},
        {"aspect":"Occasion Match",   "pass":os>=65,"score":os,
         "detail":f"Outfit appropriately suited for {occasion}."},
        {"aspect":"Style Consistency","pass":ss>=65,"score":ss,
         "detail":f"Cohesive {style} aesthetic throughout."},
        {"aspect":"Season Fit",       "pass":seas>=65,"score":seas,
         "detail":f"Fabric choices are appropriate for {season}."},
    ]
    explanation = (
        f"Your **{color_info['name']}** piece is the star of this look! "
        f"We built the outfit around it using **{pairs['pairs'][0]}** and **{pairs['pairs'][1]}** "
        f"as complementary tones — these work because they "
        f"{'balance each other with contrast' if color_info['family'] == 'cool' else 'create a warm, cohesive harmony'}. "
        f"For **{occasion}**, this combination is both stylish and appropriate. "
        f"Avoid pairing your item with **{pairs['avoid'][0]}** as it would create a visual clash."
    )
    alternatives = [
        {"replace":"Bottom / Trousers",
         "with":f"{pairs['pairs'][2] if len(pairs['pairs'])>2 else pairs['pairs'][0]} wide-leg trousers + nude heels",
         "reason":f"A {pairs['pairs'][2] if len(pairs['pairs'])>2 else pairs['pairs'][0]} bottom elevates your {color_info['name']} piece for a more polished look."},
        {"replace":"Bag",
         "with":f"A {pairs['pairs'][1]} mini shoulder bag",
         "reason":f"A {pairs['pairs'][1]} bag adds a complementary color touch while letting your top remain the focal point."},
        {"replace":"Footwear",
         "with":"White sneakers (casual) or nude block heels (dressed up)",
         "reason":"Neutral footwear keeps the outfit balanced and lets your uploaded piece shine."},
    ]
    return {"overall":overall,"verdict":verdict,"checks":checks,
            "explanation":explanation,"alternatives":alternatives,
            "scores":{"Color Harmony":cs,"Occasion Match":os,"Style Consistency":ss,"Season Fit":seas}}

def score_color(s):
    return "#C4688A" if s >= 75 else "#E8A86A" if s >= 55 else "#E87070"


# ══════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>✨ ELVA</h1>
    <p>Where Ease meets Elegance</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.2], gap="large")

# ── LEFT ──────────────────────────────────────────────────
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📸 Upload Your Clothing Item</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a top, skirt, dress or any clothing piece",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
    )

    color_info   = None
    garment_type = None

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True, caption="Your item")

        dom_rgb      = get_dominant_color(img)
        color_info   = rgb_to_color_info(dom_rgb)
        garment_type = detect_garment_type(img)
        pairs        = get_pairings(color_info)

        st.markdown(f"""
        <div class="color-detected">
          <div class="swatch" style="background:{color_info['hex']}"></div>
          <div>
            <div class="color-name">🎨 Detected color: {color_info['name']}</div>
            <div class="color-sub">Garment type: {garment_type}</div>
          </div>
        </div>
        <div style="margin-top:0.75rem">
          <div style="font-size:0.8rem;font-weight:700;color:#8B4A6B;margin-bottom:0.35rem">
            ✅ Best colors to pair with this:
          </div>
          {''.join(f'<span class="pairing-badge">{p}</span>' for p in pairs['pairs'])}
        </div>
        <div style="margin-top:0.5rem">
          <div style="font-size:0.8rem;font-weight:700;color:#C4688A;margin-bottom:0.35rem">
            ❌ Avoid pairing with:
          </div>
          {''.join(f'<span class="pairing-badge" style="background:#FFE8E8;border-color:#F0C0C0;color:#A03030">{a}</span>' for a in pairs['avoid'])}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem;color:#C0A0B0;font-size:0.88rem">
          📷 Upload a photo to detect color and get smart pairings
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌸 Occasion</div>', unsafe_allow_html=True)
    occasion = st.selectbox("Occasion",
        ["College / Campus","Party / Night Out","Formal / Office",
         "Casual / Everyday","Date Night","Wedding / Festival"],
        label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💫 Style Vibe</div>', unsafe_allow_html=True)
    style = st.selectbox("Style",
        ["Minimalist","Casual","Classic","Trendy","Bohemian","Smart Casual"],
        label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌤️ Season</div>', unsafe_allow_html=True)
    season = st.radio("Season",
        ["Spring 🌸","Summer ☀️","Autumn 🍂","Winter ❄️"],
        horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌿 Your Skin Tone (Optional)</div>', unsafe_allow_html=True)
    skin_tone = st.selectbox("Skin tone",
        ["Prefer not to say", "Fair / Light", "Light Medium", "Medium / Wheatish", "Medium Dark / Olive", "Dark / Deep"],
        label_visibility="collapsed")
    notes = st.text_area(
        "Anything else?",
        placeholder="e.g. I prefer loose fits, I already own blue jeans, I avoid bright colors...",
        height=80,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button("✨ Build My Outfit")


# ── RIGHT ─────────────────────────────────────────────────
with right:
    if not analyze:
        st.markdown("""
        <div style="height:520px;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;text-align:center;color:#C0A0B0">
          <div style="font-size:4rem;margin-bottom:1rem">👗</div>
          <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                      color:#C4688A;margin-bottom:0.5rem">
            Upload your item & hit Build My Outfit
          </div>
          <div style="font-size:0.9rem;max-width:300px;line-height:1.7">
            We detect your clothing's color and build a complete outfit
            around it — with scores and alternatives ✨
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        season_clean = season.split(" ")[0]

        if not color_info:
            # No image — use generic neutral
            color_info   = {"name":"Neutral","family":"neutral","hex":"#D4C0CC"}
            garment_type = "top / shirt"

        outfit = build_outfit(color_info, garment_type, occasion, style, season_clean)
        result = evaluate(color_info, occasion, style, season_clean)

        vc = {"Good":"verdict-good","Mixed":"verdict-mixed","Needs Work":"verdict-bad"}[result["verdict"]]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap">
          <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#8B4A6B">
            Outfit for your {color_info['name']} piece
          </div>
          <span class="{vc}">{result['verdict']}</span>
          <span style="margin-left:auto;font-size:2rem;font-weight:700;color:#C4688A">
            {result['overall']}<span style="font-size:1rem;color:#C0A0B0">/100</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

        # Outfit items
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">👗 Complete Outfit</div>', unsafe_allow_html=True)
        for item in outfit:
            if item["uploaded"]:
                st.markdown(f"""
                <div class="uploaded-item">
                  <span style="font-size:1.3rem">{item['icon']}</span>
                  <div>
                    <div style="font-weight:700;color:#8B4A6B;font-size:0.9rem">{item['name']}</div>
                    <div style="font-size:0.72rem;color:#C4688A">{item['type']}</div>
                  </div>
                  <span style="font-size:0.75rem;color:#C4688A;font-weight:700;margin-left:auto">YOUR PIECE</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="outfit-item">
                  <span style="font-size:1.3rem">{item['icon']}</span>
                  <div>
                    <div style="font-weight:600;color:#5A2040;font-size:0.9rem">{item['name']}</div>
                    <div style="font-size:0.72rem;color:#B080A0">{item['type']}</div>
                  </div>
                  <span style="font-size:0.75rem;color:#B080A0;margin-left:auto">{item['color']} · {item['fabric']}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Scores
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
        for label, score in result["scores"].items():
            st.markdown(f"""
            <div class="score-row">
              <span class="score-label">{label}</span>
              <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{score}%;background:{score_color(score)}"></div>
              </div>
              <span class="score-val">{score}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Checks
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

        # Explanation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💬 Why This Works</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#5A3050;line-height:1.75;font-size:0.92rem">{result["explanation"]}</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Alternatives
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Want to Switch It Up?</div>', unsafe_allow_html=True)
        for alt in result["alternatives"]:
            st.markdown(f"""
            <div class="alt-card">
              <div class="alt-title">Swap <em>{alt['replace']}</em> → <strong>{alt['with']}</strong></div>
              <div class="alt-reason">{alt['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Made with 💗 · ELVA · Where Ease meets Elegance</div>',
            unsafe_allow_html=True)
