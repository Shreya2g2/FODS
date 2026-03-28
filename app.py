# ── Custom CSS ────────────────────────────────────────────
import streamlit as st
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #FAF7F2;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #F2EDE4 0%, #E6DCD3 50%, #DDE5DC 100%);
    border-radius: 20px;
    padding: 2.5rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #E6DCD3;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #5E5A54;
    margin-bottom: 0.4rem;
}

.hero p {
    font-size: 1rem;
    color: #7A746D;
    font-weight: 300;
    margin: 0;
}

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.6rem;
    border: 1px solid #E6DCD3;
    box-shadow: 0 2px 16px rgba(150,130,100,0.08);
    margin-bottom: 1.4rem;
}

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #5E5A54;
    margin-bottom: 1rem;
}

.color-detected {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #F5F2EC;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    border: 1px solid #E6DCD3;
    margin-top: 0.75rem;
}

.swatch {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid #D6CFC7;
    flex-shrink: 0;
}

.color-name { font-weight: 700; color: #5E5A54; font-size: 0.95rem; }
.color-sub  { font-size: 0.75rem; color: #8A837C; margin-top: 0.1rem; }

.pairing-badge {
    display: inline-block;
    background: #EFE8E0;
    color: #6B665F;
    border: 1.5px solid #D6CFC7;
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0.2rem;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #A3B18A, #6B9080);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 0.65rem 2.5rem;
    font-size: 1rem;
    font-weight: 700;
    width: 100%;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stFileUploader"] label {
    color: #5E5A54 !important;
    font-weight: 600;
}

.score-bar-bg {
    background: #E6DCD3;
}

.verdict-good  { background:#E3F0E8; color:#2F5D50; }
.verdict-mixed { background:#F5EBDD; color:#7A5C2E; }
.verdict-bad   { background:#F8E4E4; color:#8B3A3A; }

.outfit-item {
    background:#F5F2EC;
    border:1px solid #E6DCD3;
}

.uploaded-item {
    background: linear-gradient(135deg,#EFE8E0,#E6DCD3);
    border:2px solid #A3B18A;
}

.alt-card {
    background:#F5F2EC;
    border-left:4px solid #A3B18A;
}

.footer {
    color:#A89F94;
}
</style>
""", unsafe_allow_html=True)
