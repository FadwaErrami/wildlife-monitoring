import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from models.detr import load_detr, predict_detr
from models.faster_rcnn import load_faster_rcnn, predict_faster_rcnn
from models.yolo import load_yolo, predict_yolo


@st.cache_resource
def get_model_yolo():
    return load_yolo()


@st.cache_resource
def get_model_faster_rcnn():
    return load_faster_rcnn()


@st.cache_resource
def get_model_detr():
    return load_detr()


MODEL_CONFIG = [
    {"name": "YOLO", "loader": get_model_yolo, "predictor": predict_yolo},
    {"name": "Faster R-CNN", "loader": get_model_faster_rcnn, "predictor": predict_faster_rcnn},
    {"name": "DETR", "loader": get_model_detr, "predictor": predict_detr},
]


def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def run_inference(predictor, model, image_np, threshold: float = 0.5):
    start = time.time()
    result = predictor(model, image_np, threshold)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        annotated, metadata = result
    else:
        annotated, metadata = result, {}
    latency = (time.time() - start) * 1000
    fps = 1000.0 / latency if latency > 0 else 0.0
    return annotated, latency, fps, metadata


def build_performance_label(latency_ms):
    latency_s = latency_ms / 1000
    if latency_s < 0.8:
        return "⚡ Rapide"
    if latency_s < 1.7:
        return "⏱ Moyen"
    return "🐢 Lent"


def display_model_tab(name, loader_fn, predictor, image_np, threshold):
    st.markdown(f"<h3 class='tab-title'>{name}</h3>", unsafe_allow_html=True)
    try:
        model = loader_fn()
    except Exception as exc:
        error_msg = str(exc)
        if "DETR is not available" in error_msg:
            st.error(f"❌ {name} n'est pas disponible dans votre version de torchvision.")
            st.info("💡 DETR nécessite une version plus récente de PyTorch/torchvision.")
        else:
            st.error(f"Impossible de charger {name} : {exc}")
        return None

    if hasattr(model, "_weight_source"):
        st.info(f"Poids chargés : {model._weight_source}")

    if st.button(f"▶  Lancer {name}", key=f"btn_{name}"):
        with st.spinner(f"Analyse en cours avec {name}…"):
            annotated, latency, fps, metadata = run_inference(predictor, model, image_np, threshold)

        col_img, col_stats = st.columns([3, 2], gap="large")

        with col_img:
            st.image(annotated, caption=f"Résultat — {name}", use_container_width=True)

        with col_stats:
            n = metadata.get("num_detections", 0)
            perf = build_performance_label(latency)
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-row'>
                    <span class='stat-label'>INFÉRENCE</span>
                    <span class='stat-value accent'>{latency:.0f} ms</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>VITESSE</span>
                    <span class='stat-value'>{fps:.1f} FPS</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>DÉTECTIONS</span>
                    <span class='stat-value accent'>{n}</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>PERFORMANCE</span>
                    <span class='stat-value'>{perf}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if n > 0:
                labels = metadata.get("raw_label_names", [])
                scores = metadata.get("raw_scores", [])
                preds_html = "<div class='pred-list'><p class='pred-title'>PRÉDICTIONS</p>"
                for label, score in zip(labels, scores):
                    if score >= threshold:
                        pct = int(score * 100)
                        preds_html += f"""
                        <div class='pred-row'>
                            <span class='pred-name'>{label}</span>
                            <div class='pred-bar-wrap'>
                                <div class='pred-bar' style='width:{pct}%'></div>
                            </div>
                            <span class='pred-score'>{score:.2f}</span>
                        </div>"""
                preds_html += "</div>"
                st.markdown(preds_html, unsafe_allow_html=True)
                st.success(f"✅ {n} détection{'s' if n > 1 else ''} trouvée{'s' if n > 1 else ''}.")
            else:
                st.warning("Aucune détection au-dessus du seuil — essayez de le réduire.")
                st.caption(f"Score max détecté : {metadata.get('max_score', 0.0):.2f}")

        return {"model": name, "latency": latency, "fps": fps}
    return None


def display_compare_tab(image_np, threshold):
    st.markdown("<h3 class='tab-title'>Comparaison des modèles</h3>", unsafe_allow_html=True)
    results = []
    if st.button("▶  Lancer TOUS les modèles", key="btn_compare"):
        cols = st.columns(3, gap="medium")
        for idx, config in enumerate(MODEL_CONFIG):
            with cols[idx]:
                st.markdown(f"<p class='model-badge'>{config['name']}</p>", unsafe_allow_html=True)
                try:
                    model = config["loader"]()
                    annotated, latency, fps, metadata = run_inference(
                        config["predictor"], model, image_np, threshold
                    )
                except Exception as exc:
                    st.error(f"{config['name']} : {exc}")
                    annotated, latency, fps, metadata = np.array(Image.fromarray(image_np)), None, None, {}

                st.image(annotated, use_container_width=True)
                if latency is not None:
                    n = metadata.get("num_detections", 0)
                    st.markdown(
                        f"<div class='mini-stat'>{latency:.0f} ms &nbsp;·&nbsp; {fps:.1f} FPS &nbsp;·&nbsp; {n} det.</div>",
                        unsafe_allow_html=True,
                    )
                    results.append({"model": config["name"], "latency_ms": latency, "fps": fps})

        if results:
            results = sorted(results, key=lambda item: item["latency_ms"])
            medals = ["🥇", "🥈", "🥉"]
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<p class='section-label'>CLASSEMENT</p>", unsafe_allow_html=True)
            table_html = """
            <table class='compare-table'>
                <thead><tr>
                    <th>Rang</th><th>Modèle</th><th>Latence</th><th>FPS</th><th>Performance</th>
                </tr></thead><tbody>"""
            for rank, row in enumerate(results):
                table_html += f"""<tr>
                    <td>{medals[rank]}</td>
                    <td><strong>{row['model']}</strong></td>
                    <td class='accent'>{row['latency_ms']:.0f} ms</td>
                    <td>{row['fps']:.1f}</td>
                    <td>{build_performance_label(row['latency_ms'])}</td>
                </tr>"""
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='color:#6b9e78; font-style:italic;'>Chargez une image puis cliquez sur « Lancer TOUS les modèles ».</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────
#  CSS THÈME : Forêt nocturne — élégance maximale
# ─────────────────────────────────────────────────────────
DARK_FOREST_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Crimson+Pro:wght@300;400;500&display=swap');

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: 'Crimson Pro', Georgia, serif;
    color: #c8ddc8;
}

/* ── Fond principal ── */
.stApp {
    background: #0b1610;
    background-image:
        radial-gradient(ellipse 80% 60% at 70% -10%, rgba(30,80,40,0.25) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 10% 90%, rgba(10,40,20,0.3) 0%, transparent 60%);
}
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Titre principal ── */
h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #d4edda !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.2rem !important;
}
.app-subtitle {
    font-family: 'Crimson Pro', serif;
    font-size: 0.85rem;
    letter-spacing: 0.22em;
    color: #5fc97e;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #070f0a !important;
    border-right: 1px solid #1a3022 !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #d4edda !important;
    font-size: 1rem !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8aaa90 !important;
    font-size: 0.9rem;
    line-height: 1.6;
}
[data-testid="stSidebar"] label {
    color: #6b9e78 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr {
    border-color: #1a3022 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #5fc97e !important;
    border-color: #5fc97e !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #d4edda !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f2218 !important;
    border: 1.5px dashed #2a5a38 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #5fc97e !important;
}
[data-testid="stFileUploader"] label {
    color: #8aaa90 !important;
}

/* ── Onglets (tabs) ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a3022 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Crimson Pro', serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    color: #5a7060 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.5rem !important;
    border-radius: 0 !important;
    transition: color 0.2s, border-color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #d4edda !important;
    border-bottom: 2px solid #5fc97e !important;
    background: transparent !important;
}

/* ── Boutons principaux ── */
.stButton > button {
    font-family: 'Crimson Pro', serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    background: transparent !important;
    color: #5fc97e !important;
    border: 1px solid #2a5a38 !important;
    border-radius: 50px !important;
    padding: 0.55rem 2rem !important;
    transition: all 0.22s ease !important;
}
.stButton > button:hover {
    background: #5fc97e !important;
    color: #0b1610 !important;
    border-color: #5fc97e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(95,201,126,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Titre des onglets ── */
.tab-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.4rem !important;
    color: #d4edda !important;
    margin-bottom: 1.2rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #1a3022 !important;
}

/* ── Carte de statistiques ── */
.stat-card {
    background: #0f2218;
    border: 0.5px solid #1e3a28;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.35rem 0;
    border-bottom: 0.5px solid #152a1e;
}
.stat-row:last-child { border-bottom: none; }
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #5a7a60;
    text-transform: uppercase;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #c8ddc8;
}
.stat-value.accent { color: #5fc97e; }

/* ── Liste de prédictions ── */
.pred-list {
    background: #0f2218;
    border: 0.5px solid #1e3a28;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.pred-title {
    font-size: 0.7rem !important;
    letter-spacing: 0.18em;
    color: #5a7a60;
    text-transform: uppercase;
    margin-bottom: 0.7rem !important;
}
.pred-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.5rem;
}
.pred-name {
    font-size: 0.88rem;
    color: #aacaaa;
    min-width: 68px;
    text-transform: capitalize;
}
.pred-bar-wrap {
    flex: 1;
    height: 6px;
    background: #163022;
    border-radius: 3px;
    overflow: hidden;
}
.pred-bar {
    height: 100%;
    background: linear-gradient(90deg, #2d7a44, #5fc97e);
    border-radius: 3px;
    transition: width 0.4s ease;
}
.pred-score {
    font-size: 0.8rem;
    color: #5fc97e;
    min-width: 32px;
    text-align: right;
    font-family: 'Crimson Pro', serif;
}

/* ── Mini stat (compare tab) ── */
.mini-stat {
    font-size: 0.82rem;
    color: #6b9e78;
    text-align: center;
    padding: 0.35rem 0;
    letter-spacing: 0.05em;
}

/* ── Badge modèle ── */
.model-badge {
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    color: #d4edda !important;
    font-weight: 700 !important;
    text-align: center;
    margin-bottom: 0.5rem !important;
    letter-spacing: 0.05em;
}

/* ── Séparateur ── */
.divider {
    border: none !important;
    border-top: 1px solid #1a3022 !important;
    margin: 1.5rem 0 !important;
}
.section-label {
    font-size: 0.72rem !important;
    letter-spacing: 0.2em;
    color: #5a7a60 !important;
    text-transform: uppercase;
    margin-bottom: 0.8rem !important;
}

/* ── Tableau comparatif ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Crimson Pro', serif;
    font-size: 0.95rem;
}
.compare-table thead tr {
    background: #0f2218;
    border-bottom: 1px solid #1e3a28;
}
.compare-table th {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a7a60 !important;
    padding: 0.7rem 1rem;
    text-align: left;
    background: transparent !important;
}
.compare-table td {
    padding: 0.65rem 1rem;
    color: #aacaaa;
    border-bottom: 0.5px solid #152a1e;
}
.compare-table td.accent { color: #5fc97e; }
.compare-table tbody tr:hover { background: #0f2218; }

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 0.5px solid #1e3a28;
}

/* ── Alerts / Info / Success / Warning ── */
.stAlert {
    background: #0f2218 !important;
    border-radius: 8px !important;
    border-left-color: #2a5a38 !important;
    color: #8aaa90 !important;
}
div[data-baseweb="notification"] {
    background: #0f2218 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding: 2rem 1rem 1.5rem;
    border-top: 1px solid #1a3022;
}
.footer-team {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #d4edda;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.footer-sub {
    font-size: 0.8rem;
    color: #4a6e50;
    letter-spacing: 0.05em;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="Wildlife Monitoring Camera Trap",
        page_icon="C:\\Users\\hp\\object-detection-benchmark\\wild-removebg-preview.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(DARK_FOREST_CSS, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────
    st.markdown(
        """
        <div style='margin-bottom:0.2rem'>
            <span style='font-size:2.6rem; vertical-align:middle; margin-right:12px;'>🌿</span>
            <span style='font-family:"Playfair Display",Georgia,serif;
                         font-size:2.2rem; font-weight:700;
                         color:#d4edda; vertical-align:middle;
                         letter-spacing:-0.02em;'>
                Wildlife Monitoring
            </span>
        </div>
        <p class='app-subtitle'>Camera Trap · Object Detection Benchmark</p>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "Comparez YOLO, Faster R‑CNN et DETR pour la détection d'animaux sauvages sur vos images de pièges photographiques."
    )

    # ── Sidebar ─────────────────────────────────────────
    st.sidebar.image("C:\\Users\\hp\\object-detection-benchmark\\animals-removebg-preview.png")
    threshold = st.sidebar.slider("Seuil de confiance", 0.10, 0.90, 0.50, 0.05)
    st.sidebar.caption(f"Seuil actuel : **{threshold:.2f}**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## À propos")
    st.sidebar.write(
        "Cette application compare trois architectures de détection d'objets "
        "pour le suivi de la faune sauvage par caméra-piège."
    )
    st.sidebar.write("Chargez une image et observez les résultats en temps réel.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size:0.75rem; color:#4a6e50; text-align:center; letter-spacing:0.1em;'>"
        "YOLO · FASTER R-CNN · DETR</p>",
        unsafe_allow_html=True,
    )

    # ── Upload ───────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Chargez une image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is None:
        st.warning("Veuillez charger une image pour lancer les modèles.")
        return

    image_np = load_image(uploaded_file)

    # ── Tabs ─────────────────────────────────────────────
    tabs = st.tabs(["YOLO", "Faster R-CNN", "DETR", "Comparaison"])

    for tab, config in zip(tabs[:3], MODEL_CONFIG):
        with tab:
            display_model_tab(
                config["name"], config["loader"], config["predictor"], image_np, threshold
            )

    with tabs[3]:
        display_compare_tab(image_np, threshold)

    # ── Footer ───────────────────────────────────────────
    st.markdown(
        """
        <div class='footer'>
            <p class='footer-team'>✦ Fadwa Errami &nbsp;·&nbsp; Khadija Mellak &nbsp;·&nbsp; Khajida Oufkir ✦</p>
            <p class='footer-sub'>Projet de suivi de la faune par caméra‑piège &nbsp;·&nbsp; Wildlife Monitoring System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()