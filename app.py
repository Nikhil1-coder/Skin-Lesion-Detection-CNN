import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import textwrap
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DermaAI | Skin Lesion Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HTML HELPER
# ============================================================

def html(content):
    content = textwrap.dedent(content)

    # Remove blank lines so Streamlit does not
    # interpret HTML as Markdown code
    content = "\n".join(
        line for line in content.splitlines()
        if line.strip()
    )

    st.markdown(
        content,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(37, 99, 235, 0.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(6, 182, 212, 0.08),
            transparent 25%
        ),
        #070b14;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* ============================================================
   HIDE DEFAULT STREAMLIT ELEMENTS
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
   BRAND
   ============================================================ */

.brand-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 10px 0 18px;

    border-bottom:
        1px solid rgba(148,163,184,0.12);
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon {
    width: 52px;
    height: 52px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );

    font-size: 27px;

    box-shadow:
        0 10px 30px
        rgba(37,99,235,0.30);
}

.brand-name {
    color: #f8fafc;

    font-size: 26px;

    font-weight: 800;
}

.brand-subtitle {
    color: #64748b;

    font-size: 12px;

    margin-top: 2px;
}

.online-badge {
    padding: 7px 12px;

    border-radius: 20px;

    background:
        rgba(34,197,94,0.10);

    border:
        1px solid rgba(34,197,94,0.25);

    color: #86efac;

    font-size: 12px;

    font-weight: 700;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    margin-top: 25px;
    margin-bottom: 25px;

    padding: 38px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,0.96),
            rgba(10,25,43,0.80)
        );

    border:
        1px solid rgba(96,165,250,0.16);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.20);
}

.hero-title {
    color: #f8fafc;

    font-size: 42px;

    font-weight: 850;

    line-height: 1.15;

    margin-bottom: 12px;
}

.hero-highlight {
    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #22d3ee
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 850px;

    color: #94a3b8;

    font-size: 16px;

    line-height: 1.7;
}

.badges {
    display: flex;

    gap: 9px;

    flex-wrap: wrap;

    margin-top: 22px;
}

.badge {
    padding: 7px 12px;

    border-radius: 20px;

    background:
        rgba(30,41,59,0.75);

    border:
        1px solid rgba(148,163,184,0.12);

    color: #cbd5e1;

    font-size: 12px;

    font-weight: 650;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-heading {
    color: #f8fafc;

    font-size: 23px;

    font-weight: 800;

    margin-top: 20px;

    margin-bottom: 6px;
}

.section-description {
    color: #64748b;

    font-size: 13px;

    margin-bottom: 18px;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    min-height: 120px;

    padding: 22px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(15,23,42,0.90),
            rgba(15,23,42,0.70)
        );

    border:
        1px solid rgba(148,163,184,0.12);

    text-align: center;

    transition:
        transform 0.2s ease,
        border 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);

    border:
        1px solid rgba(96,165,250,0.35);
}

.metric-icon {
    font-size: 20px;

    margin-bottom: 4px;
}

.metric-value {
    color: #f8fafc;

    font-size: 27px;

    font-weight: 850;
}

.metric-label {
    color: #94a3b8;

    font-size: 12px;

    margin-top: 4px;
}


/* ============================================================
   ANALYSIS CARD
   ============================================================ */

.analysis-card {
    padding: 24px;

    border-radius: 20px;

    background:
        rgba(15,23,42,0.82);

    border:
        1px solid rgba(148,163,184,0.13);

    margin-bottom: 20px;
}


/* ============================================================
   RESULT
   ============================================================ */

.result-card {
    padding: 30px;

    border-radius: 22px;

    text-align: center;

    margin-top: 18px;

    margin-bottom: 20px;

    animation:
        resultAppear 0.35s ease;
}

@keyframes resultAppear {

    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

.result-benign {
    background:
        linear-gradient(
            145deg,
            rgba(6,78,59,0.55),
            rgba(15,23,42,0.85)
        );

    border:
        1px solid rgba(52,211,153,0.40);
}

.result-malignant {
    background:
        linear-gradient(
            145deg,
            rgba(127,29,29,0.55),
            rgba(15,23,42,0.85)
        );

    border:
        1px solid rgba(248,113,113,0.45);
}

.result-icon {
    font-size: 45px;

    margin-bottom: 7px;
}

.result-title {
    color: #f8fafc;

    font-size: 31px;

    font-weight: 850;
}

.result-description {
    color: #cbd5e1;

    font-size: 14px;

    margin-top: 8px;
}


/* ============================================================
   CONFIDENCE
   ============================================================ */

.confidence-card {
    padding: 24px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(148,163,184,0.12);

    margin-bottom: 20px;
}

.confidence-header {
    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 10px;
}

.confidence-title {
    color: #cbd5e1;

    font-size: 14px;

    font-weight: 650;
}

.confidence-number {
    color: #f8fafc;

    font-size: 24px;

    font-weight: 850;
}

.confidence-track {
    height: 11px;

    background: #1e293b;

    border-radius: 20px;

    overflow: hidden;
}

.confidence-fill {
    height: 100%;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #06b6d4
        );

    border-radius: 20px;
}


/* ============================================================
   PROBABILITY
   ============================================================ */

.probability-card {
    padding: 22px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.78);

    border:
        1px solid rgba(148,163,184,0.12);
}

.probability-row {
    margin-bottom: 18px;
}

.probability-row:last-child {
    margin-bottom: 0;
}

.probability-label {
    display: flex;

    justify-content: space-between;

    margin-bottom: 7px;

    color: #cbd5e1;

    font-size: 13px;
}

.probability-track {
    height: 9px;

    background: #1e293b;

    border-radius: 20px;

    overflow: hidden;
}

.probability-fill {
    height: 100%;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #06b6d4
        );
}


/* ============================================================
   IMAGE CARD
   ============================================================ */

.image-card {
    padding: 12px;

    border-radius: 20px;

    background:
        rgba(15,23,42,0.82);

    border:
        1px solid rgba(148,163,184,0.13);
}

.image-card img {
    border-radius: 14px;
}


/* ============================================================
   INFO CARD
   ============================================================ */

.info-card {
    padding: 20px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.78);

    border:
        1px solid rgba(148,163,184,0.12);
}

.info-row {
    padding: 13px 0;

    border-bottom:
        1px solid rgba(148,163,184,0.08);
}

.info-row:last-child {
    border-bottom: none;
}

.info-label {
    color: #64748b;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.7px;
}

.info-value {
    color: #e2e8f0;

    font-size: 14px;

    font-weight: 650;

    margin-top: 4px;
}


/* ============================================================
   PROCESS STEPS
   ============================================================ */

.step-card {
    min-height: 155px;

    padding: 22px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.78);

    border:
        1px solid rgba(148,163,184,0.12);
}

.step-number {
    width: 38px;
    height: 38px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 11px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );

    color: white;

    font-size: 13px;

    font-weight: 850;

    margin-bottom: 14px;
}

.step-title {
    color: #f8fafc;

    font-size: 16px;

    font-weight: 750;

    margin-bottom: 6px;
}

.step-text {
    color: #94a3b8;

    font-size: 13px;

    line-height: 1.6;
}


/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer {
    margin-top: 30px;

    padding: 18px 20px;

    border-left:
        4px solid #f59e0b;

    border-radius: 10px;

    background:
        rgba(120,53,15,0.15);

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.7;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    margin-top: 40px;

    padding-top: 25px;

    border-top:
        1px solid rgba(148,163,184,0.08);

    color: #64748b;

    font-size: 12px;

    line-height: 1.8;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 48px;

    border-radius: 12px;

    font-weight: 700;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(0,0,0,0.20);
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    padding: 10px;

    border-radius: 16px;

    background:
        rgba(15,23,42,0.70);

    border:
        1px solid rgba(148,163,184,0.12);
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background:
        #090f1c;
}


/* ============================================================
   PROFESSIONAL UI OVERRIDES
   ============================================================ */

:root {
    --ink: #f4f7fb;
    --muted: #9aa9bb;
    --subtle: #66768a;
    --line: rgba(148, 163, 184, 0.16);
    --panel: rgba(13, 22, 37, 0.86);
    --panel-strong: #101d30;
    --accent: #56c7d9;
    --accent-strong: #2e8cff;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}

.stApp {
    background:
        linear-gradient(180deg, #07111f 0%, #091321 48%, #07101c 100%);
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--line);
    background: #08111e;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--ink);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.brand-wrapper {
    padding: 8px 0 20px;
    border-bottom: 1px solid var(--line);
}

.brand-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(145deg, var(--accent-strong), var(--accent));
    box-shadow: 0 10px 28px rgba(46, 140, 255, 0.22);
}

.brand-name {
    font-size: 24px;
    letter-spacing: -0.02em;
}

.brand-subtitle,
.section-description,
.hero-description,
.step-text {
    color: var(--muted);
}

.online-badge {
    border-radius: 8px;
    padding: 6px 10px;
    letter-spacing: 0.06em;
    font-size: 10px;
}

.hero {
    position: relative;
    overflow: hidden;
    margin-top: 28px;
    padding: 42px 44px;
    border-radius: 18px;
    background:
        linear-gradient(120deg, rgba(17, 39, 66, 0.96), rgba(11, 25, 43, 0.88));
    border: 1px solid rgba(86, 199, 217, 0.22);
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
}

.hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -100px;
    top: -120px;
    border: 1px solid rgba(86, 199, 217, 0.16);
    border-radius: 50%;
}

.hero-title {
    position: relative;
    z-index: 1;
    max-width: 760px;
    font-size: clamp(2rem, 4vw, 3.25rem);
    letter-spacing: -0.04em;
}

.hero-highlight {
    color: #8be7ee;
    background: none;
    -webkit-text-fill-color: initial;
    background-clip: initial;
}

.badge {
    border-radius: 8px;
    background: rgba(7, 17, 31, 0.46);
    border-color: rgba(139, 231, 238, 0.18);
    color: #c9d8e8;
}

.section-heading {
    margin-top: 28px;
    font-size: 20px;
    letter-spacing: -0.01em;
}

.metric-card,
.analysis-card,
.confidence-card,
.probability-card,
.info-card,
.step-card,
.image-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
}

.metric-card {
    min-height: 126px;
    padding: 20px;
}

.metric-value,
.confidence-number {
    color: var(--ink);
}

.metric-label,
.info-label {
    color: var(--subtle);
}

.step-card {
    min-height: 164px;
}

.step-number {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: var(--panel-strong);
    border: 1px solid rgba(86, 199, 217, 0.4);
    color: var(--accent);
}

[data-testid="stFileUploader"] {
    padding: 16px;
    border: 1px dashed rgba(86, 199, 217, 0.4);
    border-radius: 12px;
    background: rgba(10, 28, 46, 0.72);
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

.stButton > button {
    min-height: 46px;
    border-radius: 9px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    letter-spacing: 0.01em;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #247cff, #28b8cb);
    border: 0;
    box-shadow: 0 9px 24px rgba(36, 124, 255, 0.2);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 14px;
    color: var(--muted);
    font-size: 13px;
}

.stTabs [aria-selected="true"] {
    color: var(--ink);
}

.disclaimer {
    background: rgba(120, 78, 18, 0.14);
    border-left-color: #e9ae4a;
    border-radius: 8px;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 3rem;
    }

    .hero {
        padding: 28px 24px;
    }

    .hero-title {
        font-size: 2.15rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0 8px;
        font-size: 11px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "model/skin_lesion_model.keras"


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

    model_status = True

except Exception:

    model = None

    model_status = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html("""
    <div style="
        text-align:center;
        padding:15px 0 20px;
    ">

        <div style="
            font-size:38px;
        ">
            🩺
        </div>

        <div style="
            color:#f8fafc;
            font-size:22px;
            font-weight:800;
        ">
            DermaAI
        </div>

        <div style="
            color:#64748b;
            font-size:12px;
            margin-top:4px;
        ">
            CNN Skin Lesion Detection
        </div>

    </div>
    """)

    st.divider()

    st.markdown("### 🧠 Model")

    html(f"""
    <div class="info-card">

        <div class="info-row">

            <div class="info-label">
                Architecture
            </div>

            <div class="info-value">
                Custom CNN
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Input
            </div>

            <div class="info-value">
                224 × 224 × 3
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Classes
            </div>

            <div class="info-value">
                2
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Validation Accuracy
            </div>

            <div class="info-value">
                82.25%
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Status
            </div>

            <div class="info-value">
                {"🟢 Online" if model_status else "🔴 Error"}
            </div>

        </div>

    </div>
    """)

    st.divider()

    st.markdown("### 📂 Dataset")

    html("""
    <div class="info-card">

        <div class="info-row">

            <div class="info-label">
                Benign Images
            </div>

            <div class="info-value">
                1,800
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Malignant Images
            </div>

            <div class="info-value">
                1,497
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Total Images
            </div>

            <div class="info-value">
                3,297
            </div>

        </div>

    </div>
    """)

    st.divider()

    st.caption(
        "Academic AI/ML Project"
    )


# ============================================================
# TOP BRAND
# ============================================================

html(f"""
<div class="brand-wrapper">

    <div class="brand-left">

        <div class="brand-icon">
            🩺
        </div>

        <div>

            <div class="brand-name">
                DermaAI
            </div>

            <div class="brand-subtitle">
                AI-powered skin lesion classification
            </div>

        </div>

    </div>

    <div class="online-badge">
        {"● MODEL ONLINE" if model_status else "● MODEL ERROR"}
    </div>

</div>
""")


# ============================================================
# HERO
# ============================================================

html("""
<div class="hero">

    <div class="hero-title">
        AI-Powered
        <span class="hero-highlight">
            Skin Lesion Analysis
        </span>
    </div>

    <div class="hero-description">

        An academic deep-learning application that uses a Convolutional Neural Network to classify dermoscopic images into <b>Benign</b> and <b>Malignant</b> categories.

    </div>

    <div class="badges">

        <span class="badge">
            🧠 Custom CNN
        </span>

        <span class="badge">
            📐 224 × 224 Input
        </span>

        <span class="badge">
            🎯 Binary Classification
        </span>

        <span class="badge">
            ⚡ TensorFlow
        </span>

        <span class="badge">
            🚀 Streamlit
        </span>

    </div>

</div>
""")


# ============================================================
# NAVIGATION TABS
# ============================================================

tab_dashboard, tab_analyze, tab_model, tab_dataset, tab_about = st.tabs(
    [
        "🏠 Dashboard",
        "🔬 Analyze",
        "🧠 Model",
        "📊 Dataset",
        "ℹ️ About"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    st.markdown(
        '<div class="section-heading">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Key information about the trained skin lesion classification system.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🎯
            </div>

            <div class="metric-value">
                82.25%
            </div>

            <div class="metric-label">
                Validation Accuracy
            </div>

        </div>
        """)

    with c2:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🗂️
            </div>

            <div class="metric-value">
                3,297
            </div>

            <div class="metric-label">
                Dataset Images
            </div>

        </div>
        """)

    with c3:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🎯
            </div>

            <div class="metric-value">
                2
            </div>

            <div class="metric-label">
                Classes
            </div>

        </div>
        """)

    with c4:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🖼️
            </div>

            <div class="metric-value">
                224²
            </div>

            <div class="metric-label">
                Model Input
            </div>

        </div>
        """)


    st.markdown(
        '<div class="section-heading">🔬 How the System Works</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        html("""
        <div class="step-card">

            <div class="step-number">
                01
            </div>

            <div class="step-title">
                Upload
            </div>

            <div class="step-text">
                Upload a dermoscopic skin lesion
                image in JPG, JPEG or PNG format.
            </div>

        </div>
        """)

    with s2:

        html("""
        <div class="step-card">

            <div class="step-number">
                02
            </div>

            <div class="step-title">
                Preprocess
            </div>

            <div class="step-text">
                The image is resized to 224 × 224
                pixels and normalized.
            </div>

        </div>
        """)

    with s3:

        html("""
        <div class="step-card">

            <div class="step-number">
                03
            </div>

            <div class="step-title">
                CNN Analysis
            </div>

            <div class="step-text">
                The trained CNN extracts image
                features and generates probabilities.
            </div>

        </div>
        """)

    with s4:

        html("""
        <div class="step-card">

            <div class="step-number">
                04
            </div>

            <div class="step-title">
                Result
            </div>

            <div class="step-text">
                The system displays the predicted
                class and confidence score.
            </div>

        </div>
        """)


# ============================================================
# ANALYZE TAB
# ============================================================

with tab_analyze:

    st.markdown(
        '<div class="section-heading">🔬 Analyze Skin Lesion</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload an image and run the trained CNN model.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload skin lesion image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )


    if uploaded_file is None:

        st.info(
            "📤 Upload an image above to start the analysis."
        )


    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        image_col, analysis_col = st.columns(
            [1.1, 0.9],
            gap="large"
        )


        # ====================================================
        # IMAGE
        # ====================================================

        with image_col:

            st.markdown(
                '<div class="section-heading">'
                '🖼️ Uploaded Image'
                '</div>',
                unsafe_allow_html=True
            )

            html("""
            <div class="image-card">
            """)

            st.image(
                image,
                use_container_width=True
            )

            html("""
            </div>
            """)


            html(f"""
            <div class="info-card">

                <div class="info-row">

                    <div class="info-label">
                        File Name
                    </div>

                    <div class="info-value">
                        {uploaded_file.name}
                    </div>

                </div>

                <div class="info-row">

                    <div class="info-label">
                        Original Resolution
                    </div>

                    <div class="info-value">
                        {image.width} × {image.height} px
                    </div>

                </div>

                <div class="info-row">

                    <div class="info-label">
                        Processing Resolution
                    </div>

                    <div class="info-value">
                        224 × 224 px
                    </div>

                </div>

                <div class="info-row">

                    <div class="info-label">
                        Image Type
                    </div>

                    <div class="info-value">
                        RGB
                    </div>

                </div>

            </div>
            """)


        # ====================================================
        # ANALYSIS
        # ====================================================

        with analysis_col:

            st.markdown(
                '<div class="section-heading">'
                '🤖 AI Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            analyze_button = st.button(
                "🔬 Analyze Image",
                type="primary",
                use_container_width=True
            )


            clear_button = st.button(
                "↻ Clear / Reset",
                use_container_width=True
            )


            if clear_button:

                st.rerun()


            if analyze_button:

                if model is None:

                    st.error(
                        "❌ Model could not be loaded. "
                        "Check model/skin_lesion_model.keras"
                    )

                else:

                    with st.spinner(
                        "🧠 CNN is analyzing the image..."
                    ):

                        # ------------------------------------
                        # RESIZE
                        # ------------------------------------

                        processed_image = image.resize(
                            (224, 224)
                        )


                        # ------------------------------------
                        # NUMPY
                        # ------------------------------------

                        img_array = np.array(
                            processed_image
                        ).astype(
                            "float32"
                        ) / 255.0


                        # ------------------------------------
                        # ADD BATCH
                        # ------------------------------------

                        img_array = np.expand_dims(
                            img_array,
                            axis=0
                        )


                        # ------------------------------------
                        # PREDICT
                        # ------------------------------------

                        prediction = model.predict(
                            img_array,
                            verbose=0
                        )[0][0]


                        malignant_probability = float(
                            prediction
                        )

                        benign_probability = float(
                            1 - prediction
                        )


                        # ------------------------------------
                        # RESULT
                        # ------------------------------------

                        if malignant_probability >= 0.5:

                            result = "Malignant"

                            confidence = (
                                malignant_probability * 100
                            )

                            result_class = (
                                "result-malignant"
                            )

                            icon = "🔴"

                            description = (
                                "The model assigns the higher "
                                "probability to the malignant class."
                            )

                        else:

                            result = "Benign"

                            confidence = (
                                benign_probability * 100
                            )

                            result_class = (
                                "result-benign"
                            )

                            icon = "🟢"

                            description = (
                                "The model assigns the higher "
                                "probability to the benign class."
                            )


                    # ========================================
                    # SAVE HISTORY
                    # ========================================

                    st.session_state.prediction_history.append(
                        {
                            "time": datetime.now().strftime(
                                "%H:%M:%S"
                            ),
                            "file": uploaded_file.name,
                            "result": result,
                            "confidence": confidence
                        }
                    )


                    # ========================================
                    # RESULT CARD
                    # ========================================

                    html(f"""
                    <div class="result-card {result_class}">

                        <div class="result-icon">
                            {icon}
                        </div>

                        <div class="result-title">
                            {result}
                        </div>

                        <div class="result-description">
                            {description}
                        </div>

                    </div>
                    """)


                    # ========================================
                    # CONFIDENCE
                    # ========================================

                    html(f"""
                    <div class="confidence-card">

                        <div class="confidence-header">

                            <div class="confidence-title">
                                🎯 Prediction Confidence
                            </div>

                            <div class="confidence-number">
                                {confidence:.2f}%
                            </div>

                        </div>

                        <div class="confidence-track">

                            <div
                                class="confidence-fill"
                                style="width:{confidence:.2f}%;">
                            </div>

                        </div>

                    </div>
                    """)


                    # ========================================
                    # PROBABILITIES
                    # ========================================

                    st.markdown(
                        '<div class="section-heading">'
                        '📈 Class Probabilities'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    benign_percent = (
                        benign_probability * 100
                    )

                    malignant_percent = (
                        malignant_probability * 100
                    )


                    html(f"""
                    <div class="probability-card">

                        <div class="probability-row">

                            <div class="probability-label">

                                <span>
                                    🟢 Benign
                                </span>

                                <span>
                                    {benign_percent:.2f}%
                                </span>

                            </div>

                            <div class="probability-track">

                                <div
                                    class="probability-fill"
                                    style="
                                    width:{benign_percent:.2f}%;
                                    ">
                                </div>

                            </div>

                        </div>


                        <div class="probability-row">

                            <div class="probability-label">

                                <span>
                                    🔴 Malignant
                                </span>

                                <span>
                                    {malignant_percent:.2f}%
                                </span>

                            </div>

                            <div class="probability-track">

                                <div
                                    class="probability-fill"
                                    style="
                                    width:{malignant_percent:.2f}%;
                                    ">
                                </div>

                            </div>

                        </div>

                    </div>
                    """)


# ============================================================
# MODEL TAB
# ============================================================

with tab_model:

    st.markdown(
        '<div class="section-heading">'
        '🧠 CNN Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Technical information about the trained classification model.'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)


    with c1:

        html("""
        <div class="info-card">

            <div class="info-row">

                <div class="info-label">
                    Model Architecture
                </div>

                <div class="info-value">
                    Custom Convolutional Neural Network
                </div>

            </div>

            <div class="info-row">

                <div class="info-label">
                    Input Shape
                </div>

                <div class="info-value">
                    224 × 224 × 3
                </div>

            </div>

            <div class="info-row">

                <div class="info-label">
                    Classification Type
                </div>

                <div class="info-value">
                    Binary Classification
                </div>

            </div>

            <div class="info-row">

                <div class="info-label">
                    Classes
                </div>

                <div class="info-value">
                    Benign / Malignant
                </div>

            </div>

        </div>
        """)


    with c2:

        html("""
        <div class="info-card">

            <div class="info-row">

                <div class="info-label">
                    Framework
                </div>

                <div class="info-value">
                    TensorFlow / Keras
                </div>

            </div>

            <div class="info-row">

                <div class="info-label">
                    Application
                </div>

                <div class="info-value">
                    Streamlit
                </div>

            </div>

            <div class="info-row">

                <div class="info-label">
                    Model Input
                </div>

                <div class="info-value">
                    Normalized RGB Image
                </div>

            </div>

            <div class="info-row">

                <div class="info-label">
                    Validation Accuracy
                </div>

                <div class="info-value">
                    82.25%
                </div>

            </div>

        </div>
        """)


    st.markdown(
        '<div class="section-heading">'
        '⚙️ Prediction Pipeline'
        '</div>',
        unsafe_allow_html=True
    )


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        html("""
        <div class="step-card">

            <div class="step-number">
                01
            </div>

            <div class="step-title">
                Input
            </div>

            <div class="step-text">
                RGB skin lesion image.
            </div>

        </div>
        """)


    with p2:

        html("""
        <div class="step-card">

            <div class="step-number">
                02
            </div>

            <div class="step-title">
                Resize
            </div>

            <div class="step-text">
                Image converted to
                224 × 224 pixels.
            </div>

        </div>
        """)


    with p3:

        html("""
        <div class="step-card">

            <div class="step-number">
                03
            </div>

            <div class="step-title">
                CNN
            </div>

            <div class="step-text">
                Image features are processed
                by the trained network.
            </div>

        </div>
        """)


    with p4:

        html("""
        <div class="step-card">

            <div class="step-number">
                04
            </div>

            <div class="step-title">
                Output
            </div>

            <div class="step-text">
                Probability and predicted
                class are generated.
            </div>

        </div>
        """)


# ============================================================
# DATASET TAB
# ============================================================

with tab_dataset:

    st.markdown(
        '<div class="section-heading">'
        '📊 Dataset Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Dataset statistics used for the skin lesion classification project.'
        '</div>',
        unsafe_allow_html=True
    )


    d1, d2, d3 = st.columns(3)


    with d1:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🟢
            </div>

            <div class="metric-value">
                1,800
            </div>

            <div class="metric-label">
                Benign Images
            </div>

        </div>
        """)


    with d2:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🔴
            </div>

            <div class="metric-value">
                1,497
            </div>

            <div class="metric-label">
                Malignant Images
            </div>

        </div>
        """)


    with d3:

        html("""
        <div class="metric-card">

            <div class="metric-icon">
                🗂️
            </div>

            <div class="metric-value">
                3,297
            </div>

            <div class="metric-label">
                Total Images
            </div>

        </div>
        """)


    st.markdown(
        '<div class="section-heading">'
        '📌 Dataset Classes'
        '</div>',
        unsafe_allow_html=True
    )


    html("""
    <div class="info-card">

        <div class="info-row">

            <div class="info-label">
                Class 1
            </div>

            <div class="info-value">
                🟢 Benign — 1,800 images
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Class 2
            </div>

            <div class="info-value">
                🔴 Malignant — 1,497 images
            </div>

        </div>

        <div class="info-row">

            <div class="info-label">
                Classification
            </div>

            <div class="info-value">
                Binary Classification
            </div>

        </div>

    </div>
    """)


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.markdown(
        '<div class="section-heading">'
        'ℹ️ About DermaAI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Academic project information.'
        '</div>',
        unsafe_allow_html=True
    )


    html("""
    <div class="analysis-card">

        <div class="section-title">
            🩺 Project Objective
        </div>

        <div class="hero-description">

            DermaAI is an academic deep-learning project
            designed to demonstrate how Convolutional Neural
            Networks can be applied to skin lesion image
            classification.

            <br><br>

            The system processes a dermoscopic image and
            produces a binary prediction between Benign
            and Malignant classes together with the model's
            prediction confidence.

        </div>

    </div>
    """)


    html("""
    <div class="disclaimer">

        <b>⚠️ Medical Disclaimer</b>

        <br><br>

        This application is developed for educational,
        academic and demonstration purposes only.

        <br><br>

        The prediction generated by this system must not
        be considered a medical diagnosis and should not
        replace examination, consultation or advice from
        a qualified healthcare professional.

    </div>
    """)

# ============================================================
# PREDICTION HISTORY
# ============================================================

if len(st.session_state.prediction_history) > 0:

    st.markdown(
        '<div class="section-heading">'
        '🕘 Recent Predictions'
        '</div>',
        unsafe_allow_html=True
    )

    for item in reversed(
        st.session_state.prediction_history[-5:]
    ):

        icon = (
            "🟢"
            if item["result"] == "Benign"
            else "🔴"
        )

        html(f"""
        <div class="info-card"
             style="margin-bottom:10px;">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
            ">

                <div>

                    <div class="info-value">
                        {icon} {item["result"]}
                    </div>

                    <div class="info-label">
                        {item["file"]}
                        • {item["time"]}
                    </div>

                </div>

                <div style="
                    color:#f8fafc;
                    font-size:18px;
                    font-weight:800;
                ">
                    {item["confidence"]:.2f}%
                </div>

            </div>

        </div>
        """)


# ============================================================
# DISCLAIMER
# ============================================================

html("""
<div class="disclaimer">

    <b>⚠️ Important Notice</b>

    <br><br>

    This AI system is an academic demonstration of
    computer vision and deep learning. Predictions are
    generated by a trained machine-learning model and
    are not intended for clinical diagnosis.

</div>
""")


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="footer">

    🩺 <b>DermaAI</b>

    <br>

    Skin Lesion Detection using Convolutional Neural Network

    <br>

    Python • TensorFlow • Keras • Streamlit • Computer Vision

    <br><br>

    Academic AI/ML Project

</div>
""")