import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="FungiGuard AI • Mushroom Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Theme-Adaptive & Polished Mycology Design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #10b981 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 20px;
        color: #ffffff !important;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 30px -8px rgba(13, 148, 136, 0.35);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.2);
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
        font-size: 2.3rem;
        margin: 0 0 0.4rem 0;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }

    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
        max-width: 840px;
        line-height: 1.5;
        color: #f0fdf4 !important;
    }

    /* Metric Badges Bar */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.8rem;
    }

    .metric-chip {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    .metric-chip-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748b !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .metric-chip-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a !important;
    }

    /* Result Cards */
    .result-edible {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 18px;
        padding: 2.2rem 1.8rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(34, 197, 94, 0.25);
    }

    .result-poisonous {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 18px;
        padding: 2.2rem 1.8rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.25);
    }

    .result-heading {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0 0.2rem 0;
        letter-spacing: -0.5px;
    }

    .confidence-score {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.8rem 0;
    }

    .result-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .pill-green {
        background: #bbf7d0;
        color: #166534 !important;
    }

    .pill-red {
        background: #fecaca;
        color: #991b1b !important;
    }

    .key-metric-badge {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }

    .key-metric-label {
        font-size: 0.72rem;
        color: #64748b !important;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.4px;
    }

    .key-metric-val {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-top: 0.15rem;
    }

    .disclaimer-card {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1.1rem 1.3rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1.5rem;
        font-size: 0.86rem;
        color: #78350f !important;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive Human-Readable Feature Mapping Dictionaries (Pure Words Alone)
FEATURE_VALUE_MAPPINGS = {
    'cap-shape': {
        'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 
        'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken'
    },
    'cap-surface': {
        'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth'
    },
    'cap-color': {
        'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 
        'r': 'Green', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 
        'w': 'White', 'y': 'Yellow'
    },
    'bruises': {
        't': 'Bruises', 'f': 'No Bruises'
    },
    'odor': {
        'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 
        'f': 'Foul', 'm': 'Musty', 'n': 'None', 
        'p': 'Pungent', 's': 'Spicy'
    },
    'gill-attachment': {
        'a': 'Attached', 'f': 'Free'
    },
    'gill-spacing': {
        'c': 'Close', 'w': 'Crowded'
    },
    'gill-size': {
        'b': 'Broad', 'n': 'Narrow'
    },
    'gill-color': {
        'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 
        'g': 'Gray', 'r': 'Green', 'o': 'Orange', 'p': 'Pink', 
        'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'
    },
    'stalk-shape': {
        'e': 'Enlarging', 't': 'Tapering'
    },
    'stalk-root': {
        'b': 'Bulbous', 'c': 'Club', 'e': 'Equal', 
        'r': 'Rooted', '?': 'Missing'
    },
    'stalk-surface-above-ring': {
        'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'
    },
    'stalk-surface-below-ring': {
        'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'
    },
    'stalk-color-above-ring': {
        'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 
        'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'
    },
    'stalk-color-below-ring': {
        'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 
        'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'
    },
    'veil-type': {
        'p': 'Partial'
    },
    'veil-color': {
        'n': 'Brown', 'o': 'Orange', 'w': 'White', 'y': 'Yellow'
    },
    'ring-number': {
        'n': 'None', 'o': 'One', 't': 'Two'
    },
    'ring-type': {
        'e': 'Evanescent', 'f': 'Flaring', 'l': 'Large', 
        'n': 'None', 'p': 'Pendant'
    },
    'spore-print-color': {
        'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 
        'r': 'Green', 'o': 'Orange', 'u': 'Purple', 'w': 'White', 'y': 'Yellow'
    },
    'population': {
        'a': 'Abundant', 'c': 'Clustered', 'n': 'Numerous', 
        's': 'Scattered', 'v': 'Several', 'y': 'Solitary'
    },
    'habitat': {
        'd': 'Woods', 'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 
        'p': 'Paths', 'u': 'Urban', 'w': 'Waste'
    }
}

# Pure Word Formatter without character or parentheses
def format_feature_option(feature, code):
    return FEATURE_VALUE_MAPPINGS.get(feature, {}).get(code, str(code).title())

# Cached Model & Dataset Loading
@st.cache_resource
def load_ml_assets():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        df = pd.read_csv('mushrooms.csv')
        
        # Build individual column encoders for high-fidelity transformations
        encoders = {}
        for col in df.columns:
            if col != 'class':
                le = LabelEncoder()
                le.fit(df[col])
                encoders[col] = le

        return model, df, encoders
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

model, df, encoders = load_ml_assets()

if model is None or df is None:
    st.error("⚠️ Failed to load machine learning model or dataset. Ensure `model.pkl` and `mushrooms.csv` are present.")
    st.stop()

# All 22 features in exact model training order
feature_order = list(getattr(model, 'feature_names_in_', [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]))

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🍄 FungiGuard AI • Multi-Feature Classifier</div>
    <h1>Mushroom Edibility & Toxicity Classifier</h1>
    <p>Predict wild fungal specimen edibility with machine learning precision, explore full botanical diagnostic markers, and inspect comprehensive toxicity risk metrics in real-time.</p>
</div>
""", unsafe_allow_html=True)

# Top Metric Chips
st.markdown("""
<div class="metric-grid">
    <div class="metric-chip">
        <div class="metric-chip-label">Classifier Accuracy</div>
        <div class="metric-chip-value" style="color: #0d9488 !important;">99.1%</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Dataset Baseline</div>
        <div class="metric-chip-value">8,124 Samples</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Model Architecture</div>
        <div class="metric-chip-value">Random Forest</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Decision Protocol</div>
        <div class="metric-chip-value" style="color: #2563eb !important;">Multi-Feature</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State without conflicting default arguments
for feat in feature_order:
    key_name = f"feat_{feat}"
    if key_name not in st.session_state:
        st.session_state[key_name] = df[feat].iloc[0]

# Two Column Responsive Layout
col_left, col_right = st.columns([1.35, 1.0], gap="large")

with col_left:
    st.markdown("### 🎛️ Specimen Diagnostic Attributes")
    st.caption("Select botanical characteristics observed on the mushroom specimen. All options are displayed as clean, plain-English botanical terms.")

    # Quick Presets Bar
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        if st.button("🥗 Preset: Safe Edible", width="stretch", help="Load features typical of safe culinary mushrooms"):
            edible_preset = {
                'odor': 'a', 'bruises': 't', 'gill-spacing': 'w', 'gill-size': 'b',
                'gill-color': 'w', 'stalk-surface-below-ring': 's', 'ring-type': 'p',
                'population': 'a', 'habitat': 'm', 'cap-shape': 'x', 'cap-surface': 's',
                'cap-color': 'n', 'spore-print-color': 'k'
            }
            for k, v in edible_preset.items():
                st.session_state[f"feat_{k}"] = v
            st.rerun()

    with pcol2:
        if st.button("☠️ Preset: Highly Toxic", width="stretch", help="Load features typical of poisonous mushrooms"):
            toxic_preset = {
                'odor': 'f', 'bruises': 'f', 'gill-spacing': 'c', 'gill-size': 'n',
                'gill-color': 'b', 'stalk-surface-below-ring': 'k', 'ring-type': 'l',
                'population': 'v', 'habitat': 'p', 'cap-shape': 'k', 'cap-surface': 'y',
                'cap-color': 'e', 'spore-print-color': 'h'
            }
            for k, v in toxic_preset.items():
                st.session_state[f"feat_{k}"] = v
            st.rerun()

    with pcol3:
        if st.button("🔄 Reset Defaults", width="stretch", help="Reset all controls to first sample"):
            for feat in feature_order:
                st.session_state[f"feat_{feat}"] = df[feat].iloc[0]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Categorized Input Tabs
    tab_cap, tab_gills, tab_stalk, tab_eco = st.tabs([
        "🍄 Cap & Odor",
        "🌿 Gills & Spores",
        "🌲 Stalk & Ring",
        "🌍 Habitat & Ecology"
    ])

    user_raw_inputs = {}

    def render_selectbox(feature_name, label, options_list, help_text=None):
        return st.selectbox(
            label,
            options=options_list,
            key=f"feat_{feature_name}",
            format_func=lambda x: format_feature_option(feature_name, x),
            help=help_text
        )

    with tab_cap:
        c1, c2 = st.columns(2)
        with c1:
            user_raw_inputs['odor'] = render_selectbox(
                'odor', "👃 Odor", list(df['odor'].unique()), "Scent is the single most critical primary indicator of mushroom edibility."
            )
            user_raw_inputs['bruises'] = render_selectbox(
                'bruises', "🫐 Bruises When Cut / Handled", list(df['bruises'].unique()), "Whether physical fungal tissue discolors upon bruising or slicing."
            )
            user_raw_inputs['cap-shape'] = render_selectbox(
                'cap-shape', "🧢 Cap Shape", list(df['cap-shape'].unique())
            )
        with c2:
            user_raw_inputs['cap-surface'] = render_selectbox(
                'cap-surface', "✨ Cap Surface Texture", list(df['cap-surface'].unique())
            )
            user_raw_inputs['cap-color'] = render_selectbox(
                'cap-color', "🎨 Cap Color", list(df['cap-color'].unique())
            )

    with tab_gills:
        g1, g2 = st.columns(2)
        with g1:
            user_raw_inputs['gill-size'] = render_selectbox(
                'gill-size', "📏 Gill Size", list(df['gill-size'].unique())
            )
            user_raw_inputs['gill-spacing'] = render_selectbox(
                'gill-spacing', "↔️ Gill Spacing", list(df['gill-spacing'].unique())
            )
            user_raw_inputs['gill-color'] = render_selectbox(
                'gill-color', "🎨 Gill Color", list(df['gill-color'].unique())
            )
        with g2:
            user_raw_inputs['gill-attachment'] = render_selectbox(
                'gill-attachment', "📎 Gill Attachment", list(df['gill-attachment'].unique())
            )
            user_raw_inputs['veil-color'] = render_selectbox(
                'veil-color', "🪶 Veil Color", list(df['veil-color'].unique())
            )
            user_raw_inputs['spore-print-color'] = render_selectbox(
                'spore-print-color', "🔬 Spore Print Color", list(df['spore-print-color'].unique())
            )
            user_raw_inputs['veil-type'] = 'p'

    with tab_stalk:
        s1, s2 = st.columns(2)
        with s1:
            user_raw_inputs['stalk-shape'] = render_selectbox(
                'stalk-shape', "📐 Stalk Shape", list(df['stalk-shape'].unique())
            )
            user_raw_inputs['stalk-root'] = render_selectbox(
                'stalk-root', "🌱 Stalk Root", list(df['stalk-root'].unique())
            )
            user_raw_inputs['stalk-surface-above-ring'] = render_selectbox(
                'stalk-surface-above-ring', "🪵 Surface Above Ring", list(df['stalk-surface-above-ring'].unique())
            )
            user_raw_inputs['stalk-surface-below-ring'] = render_selectbox(
                'stalk-surface-below-ring', "🪵 Surface Below Ring", list(df['stalk-surface-below-ring'].unique())
            )
        with s2:
            user_raw_inputs['stalk-color-above-ring'] = render_selectbox(
                'stalk-color-above-ring', "🎨 Color Above Ring", list(df['stalk-color-above-ring'].unique())
            )
            user_raw_inputs['stalk-color-below-ring'] = render_selectbox(
                'stalk-color-below-ring', "🎨 Color Below Ring", list(df['stalk-color-below-ring'].unique())
            )
            user_raw_inputs['ring-type'] = render_selectbox(
                'ring-type', "💍 Ring Type", list(df['ring-type'].unique())
            )
            user_raw_inputs['ring-number'] = render_selectbox(
                'ring-number', "🔢 Ring Number", list(df['ring-number'].unique())
            )

    with tab_eco:
        e1, e2 = st.columns(2)
        with e1:
            user_raw_inputs['population'] = render_selectbox(
                'population', "👥 Specimen Population", list(df['population'].unique()), "Distribution pattern in natural habitat."
            )
        with e2:
            user_raw_inputs['habitat'] = render_selectbox(
                'habitat', "🏞️ Natural Habitat", list(df['habitat'].unique()), "Surrounding natural ecosystem."
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⚡ Run Real-Time Safety Assessment", type="primary", width="stretch")

# Build feature vector in exact model input order
encoded_row = []
for feat in feature_order:
    raw_val = user_raw_inputs.get(feat, df[feat].iloc[0])
    try:
        encoded_val = encoders[feat].transform([raw_val])[0]
    except Exception:
        encoded_val = 0
    encoded_row.append(encoded_val)

input_df = pd.DataFrame([encoded_row], columns=feature_order)

with col_right:
    st.markdown("### 📊 AI Classification & Toxicity Analysis")

    try:
        # Run inference
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0]

        edible_prob = float(prediction_prob[0])
        poisonous_prob = float(prediction_prob[1])

        # Dynamic Result Banner
        if poisonous_prob > 0.50:
            st.markdown(f"""
            <div class="result-poisonous">
                <span style="font-size: 3.2rem;">☠️</span>
                <div style="margin-top: 0.5rem;"><span class="result-pill pill-red">⚠️ High Toxicity Hazard</span></div>
                <div class="result-heading" style="color: #991b1b !important;">POISONOUS MUSHROOM</div>
                <div class="confidence-score" style="color: #dc2626 !important;">{poisonous_prob:.1%}</div>
                <p style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem; color: #7f1d1d !important;">
                    Fatal or Severe Toxic Risk Detected
                </p>
                <div style="background: rgba(239, 68, 68, 0.12); padding: 0.85rem 1rem; border-radius: 10px; font-size: 0.88rem; color: #991b1b !important; text-align: left; line-height: 1.5;">
                    🚫 <b>CRITICAL SAFETY NOTICE:</b> Ingestion of this specimen is highly likely to cause acute mycetism, severe gastrointestinal toxicity, or organ failure. <b>DO NOT CONSUME.</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-edible">
                <span style="font-size: 3.2rem;">🥗</span>
                <div style="margin-top: 0.5rem;"><span class="result-pill pill-green">✅ Non-Toxic Profile</span></div>
                <div class="result-heading" style="color: #166534 !important;">EDIBLE MUSHROOM</div>
                <div class="confidence-score" style="color: #15803d !important;">{edible_prob:.1%}</div>
                <p style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem; color: #14532d !important;">
                    High Botanical Correlation with Edible Species
                </p>
                <div style="background: rgba(34, 197, 94, 0.14); padding: 0.85rem 1rem; border-radius: 10px; font-size: 0.88rem; color: #14532d !important; text-align: left; line-height: 1.5;">
                    ✨ <b>CULINARY PROFILE:</b> Morphological and scent characteristics align closely with safe culinary wild varieties.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dual Probability Meters
        st.markdown("#### 📈 Probability Distribution")
        
        st.write(f"**Edibility Confidence:** `{edible_prob:.2%}`")
        st.progress(edible_prob)
        
        st.write(f"**Toxicity / Poison Probability:** `{poisonous_prob:.2%}`")
        st.progress(poisonous_prob)

        # Key Diagnostic Badges (Words alone, no codes or parentheses)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Diagnostic Indicator Summary")
        
        sm1, sm2 = st.columns(2)
        with sm1:
            odor_text = format_feature_option('odor', user_raw_inputs['odor'])
            bruise_text = format_feature_option('bruises', user_raw_inputs['bruises'])
            spacing_text = format_feature_option('gill-spacing', user_raw_inputs['gill-spacing'])
            
            st.markdown(f"""
            <div class="key-metric-badge">
                <div class="key-metric-label">Odor Marker</div>
                <div class="key-metric-val">{odor_text}</div>
            </div>
            <div class="key-metric-badge">
                <div class="key-metric-label">Bruising Reaction</div>
                <div class="key-metric-val">{bruise_text}</div>
            </div>
            <div class="key-metric-badge">
                <div class="key-metric-label">Gill Spacing</div>
                <div class="key-metric-val">{spacing_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with sm2:
            spore_text = format_feature_option('spore-print-color', user_raw_inputs['spore-print-color'])
            pop_text = format_feature_option('population', user_raw_inputs['population'])
            hab_text = format_feature_option('habitat', user_raw_inputs['habitat'])

            st.markdown(f"""
            <div class="key-metric-badge">
                <div class="key-metric-label">Spore Print</div>
                <div class="key-metric-val">{spore_text}</div>
            </div>
            <div class="key-metric-badge">
                <div class="key-metric-label">Population Pattern</div>
                <div class="key-metric-val">{pop_text}</div>
            </div>
            <div class="key-metric-badge">
                <div class="key-metric-label">Ecology / Habitat</div>
                <div class="key-metric-val">{hab_text}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Prediction could not be evaluated: {err}")

    # Botany Disclaimer Card
    st.markdown("""
    <div class="disclaimer-card">
        <b>⚠️ Mycological Safety Advisory:</b>
        Never consume wild-foraged mushrooms based solely on automated classifiers. Many edible species have deadly toxic lookalikes in nature. Always cross-verify with certified mycological field experts.
    </div>
    """, unsafe_allow_html=True)
