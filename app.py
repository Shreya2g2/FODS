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
html, body {
    background-color: #FAF7F2;
    font-family: 'sans-serif';
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #F2EDE4, #E6DCD3, ##A98BBF);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 { color: #5E5A54; }
.hero p { color: #7A746D; }

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #E6DCD3;
    margin-bottom: 1rem;
}

.card-title {
    font-weight: bold;
    color: #5E5A54;
    margin-bottom: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #A3B18A, #6B9080);
    color: white;
    border-radius: 20px;
    border: none;
}

.footer {
    text-align:center;
    color:#A89F94;
    margin-top:2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Functions ─────────────────────────────────────────────
def get_dominant_color(pil_img):
    img = pil_img.convert("RGB").resize((50, 50))
    pixels = np.array(img).reshape(-1, 3)
    return tuple(pixels.mean(axis=0).astype(int))

def rgb_to_color_info(rgb):
    return {"name": "Custom Color", "hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"}

def build_outfit(color_info):
    return [
        {"type": "Top", "name": f"{color_info['name']} Top"},
        {"type": "Bottom", "name": "Neutral Pants"},
        {"type": "Shoes", "name": "White Sneakers"},
    ]

# ── UI ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>✨ ELVA</h1>
  <p>Where Ease meets Elegance</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2)

# LEFT SIDE
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Upload Item</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg","png"])

    color_info = None

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)

        rgb = get_dominant_color(img)
        color_info = rgb_to_color_info(rgb)

        st.write(f"Detected color: {color_info['hex']}")

    st.markdown('</div>', unsafe_allow_html=True)

    occasion = st.selectbox("Occasion", ["Casual","Party","Formal"])

    analyze = st.button("Build Outfit")

# RIGHT SIDE
with right:
    if analyze:
        if not color_info:
            color_info = {"name":"Neutral","hex":"#CCCCCC"}

        outfit = build_outfit(color_info)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Your Outfit</div>', unsafe_allow_html=True)

        for item in outfit:
            st.write(f"{item['type']}: {item['name']}")

        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Made with 💗 · ELVA</div>', unsafe_allow_html=True)
