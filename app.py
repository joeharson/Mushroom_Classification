import streamlit as st
import pandas as pd

# Page View for Home & Overview
def show_overview():
    # Custom High-End Styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .hero-banner {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #10b981 100%);
            padding: 2.4rem 2.6rem;
            border-radius: 22px;
            color: #ffffff !important;
            margin-bottom: 2rem;
            box-shadow: 0 12px 30px -8px rgba(13, 148, 136, 0.35);
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.22);
            backdrop-filter: blur(8px);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.7px;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #ffffff !important;
        }

        .hero-banner h1 {
            font-weight: 800;
            font-size: 2.4rem;
            margin: 0 0 0.5rem 0;
            color: #ffffff !important;
            letter-spacing: -0.5px;
        }

        .hero-banner p {
            font-size: 1.08rem;
            opacity: 0.96;
            margin: 0;
            max-width: 860px;
            line-height: 1.6;
            color: #f0fdf4 !important;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.1rem;
            margin-bottom: 2rem;
        }

        .metric-chip {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-chip:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        }

        .metric-chip-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #64748b !important;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }

        .metric-chip-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a !important;
        }

        .info-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1.6rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.03);
            margin-bottom: 1.3rem;
        }

        .info-card h3 {
            font-weight: 800;
            font-size: 1.2rem;
            color: #0f172a !important;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .cta-banner {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1.5px solid #86efac;
            border-radius: 18px;
            padding: 1.8rem;
            margin-top: 1.5rem;
            text-align: center;
            box-shadow: 0 6px 20px -5px rgba(34, 197, 94, 0.15);
        }

        .pipeline-step {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
        }

        .pipeline-step b {
            color: #0f172a !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Hero Header
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🍄 FungiGuard AI • Mycology Platform</div>
        <h1>Mushroom Edibility & Toxicity AI Suite</h1>
        <p>A supervised machine learning system powered by an ensemble Random Forest classifier to identify wild mushroom edibility and evaluate toxicological risks across 22 morphological indicators.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Grid
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-chip">
            <div class="metric-chip-label">Verified Specimens</div>
            <div class="metric-chip-value">8,124 Samples</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Diagnostic Markers</div>
            <div class="metric-chip-value">22 Attributes</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Algorithm Architecture</div>
            <div class="metric-chip-value">Random Forest</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Classifier Accuracy</div>
            <div class="metric-chip-value" style="color: #0d9488 !important;">99.1% Test Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    with col_left:
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Project Overview & Objectives</h3>
            <p style="color: #334155; line-height: 1.6; margin: 0;">
                Foraging wild fungi is a popular activity, but dangerous lookalikes present severe toxicological risks. 
                This application uses ensemble decision-tree machine learning to recognize subtle morphological patterns across cap shape, gill spacing, tissue bruising, and scent profiles, providing instant edibility confidence scores.
            </p>
        </div>
        <div class="info-card">
            <h3>⚙️ Machine Learning Pipeline Architecture</h3>
            <div class="pipeline-step">
                <b>1. Data Ingestion:</b> UCI Mushroom dataset containing 8,124 verified fungal records across 22 categorical attributes.
            </div>
            <div class="pipeline-step">
                <b>2. Categorical Preprocessing:</b> High-fidelity LabelEncoding mapped to pure botanical descriptions.
            </div>
            <div class="pipeline-step">
                <b>3. Model Training:</b> Random Forest ensemble with 100 estimators, maximum depth pruning, and random state optimization.
            </div>
            <div class="pipeline-step">
                <b>4. Real-Time Inference:</b> Instant sub-15ms probability estimation with dual edibility vs. toxicity hazard breakdown.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Dataset Class Balance</h3>
            <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0.8rem;">
                The baseline dataset has balanced representation across both classes:
            </p>
            <div style="background: #f1f5f9; padding: 0.85rem 1rem; border-radius: 12px; margin-bottom: 0.6rem;">
                <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 0.95rem; color: #166534;">
                    <span>🥗 Edible (e)</span>
                    <span>4,208 samples (51.8%)</span>
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 0.85rem 1rem; border-radius: 12px;">
                <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 0.95rem; color: #991b1b;">
                    <span>☠️ Poisonous (p)</span>
                    <span>3,916 samples (48.2%)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cta-banner">
            <h4 style="margin: 0 0 0.4rem 0; color: #166534; font-weight: 800; font-size: 1.15rem;">🚀 Test a Wild Mushroom Specimen</h4>
            <p style="margin: 0 0 1rem 0; color: #15803d; font-size: 0.92rem;">
                Launch the interactive diagnostic classifier to evaluate morphological features and view live risk scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🍄 Open Mushroom Classifier", type="primary", width="stretch"):
            try:
                st.switch_page("pages/app1.py")
            except Exception:
                pass

    # Dataset Preview
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔎 Interactive Dataset Preview (UCI Mushroom Benchmark)"):
        try:
            df_preview = pd.read_csv('mushrooms.csv')
            st.dataframe(df_preview.head(15), width="stretch")
        except Exception as e:
            st.error(f"Could not load mushrooms.csv: {e}")

# Page Configuration
st.set_page_config(
    page_title="FungiGuard AI • Home & Overview",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Multi-Page Navigation with Custom UI Names (keeping filenames app.py and pages/app1.py)
overview_page = st.Page(show_overview, title="Home & Overview", icon="🏠", default=True)
classifier_page = st.Page("pages/app1.py", title="Mushroom Classifier", icon="🍄")

pg = st.navigation({
    "AI Mycology Suite": [overview_page, classifier_page]
})

# Sidebar branding
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.6rem 0 0.9rem 0;">
        <span style="font-size: 2.2rem;">🍄</span>
        <h3 style="margin: 0.3rem 0 0.1rem 0; font-weight: 800; font-size: 1.2rem; color: #0f172a;">FungiGuard AI</h3>
        <p style="margin: 0; font-size: 0.8rem; color: #64748b; font-weight: 600;">Specimen Safety Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

pg.run()
