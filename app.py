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
.hero p { font-size: 1rem; color: #7A746D; font-weight: 300; margin: 0; }

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
    width: 36px; height: 36px;
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
    font-family: 'Lato', sans-serif;
    width: 100%;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(100,130,100,0.3);
}

div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stFileUploader"] label {
    color: #5E5A54 !important;
    font-weight: 600;
}

.score-row { display:flex; align-items:center; gap:1rem; margin-bottom:0.75rem; }
.score-label { font-size:0.85rem; font-weight:600; color:#5E5A54; width:160px; flex-shrink:0; }
.score-bar-bg { flex:1; height:8px; background:#E6DCD3; border-radius:10px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:10px; }
.score-val { font-size:0.8rem; font-weight:700; width:36px; text-align:right; color:#6B9080; }

.verdict-good  { background:#E3F0E8; color:#2F5D50; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }
.verdict-mixed { background:#F5EBDD; color:#7A5C2E; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }
.verdict-bad   { background:#F8E4E4; color:#8B3A3A; padding:0.3rem 1rem; border-radius:50px; font-weight:700; font-size:0.9rem; display:inline-block; }

.outfit-item {
    display:flex; align-items:center; gap:0.75rem;
    background:#F5F2EC;
    border-radius:10px;
    padding:0.6rem 1rem; margin-bottom:0.5rem;
    border:1px solid #E6DCD3;
}
.uploaded-item {
    display:flex; align-items:center; gap:0.75rem;
    background: linear-gradient(135deg,#EFE8E0,#E6DCD3);
    border-radius:10px; padding:0.6rem 1rem;
    margin-bottom:0.5rem;
    border:2px solid #A3B18A;
}

.alt-card {
    background:#F5F2EC; border-left:4px solid #A3B18A;
    border-radius:0 10px 10px 0;
    padding:0.75rem 1rem; margin-bottom:0.6rem;
}
.alt-title { font-weight:700; color:#5E5A54; font-size:0.88rem; }
.alt-reason { font-size:0.78rem; color:#8A837C; margin-top:0.2rem; }

.check-row { display:flex; gap:0.6rem; align-items:flex-start; margin-bottom:0.6rem; }
.check-detail { color:#8A837C; font-size:0.78rem; margin-top:0.15rem; }

.footer { text-align:center; color:#A89F94; font-size:0.78rem; margin-top:3rem; padding:1rem; }
</style>
