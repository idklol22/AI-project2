"""
Streamlit Frontend — Predictive Maintenance Dashboard
======================================================
A premium, light-themed dashboard that provides:
  1. Manual feature input form (15 vibration features + metadata)
  2. Sliders below each input for quick adjustment
  3. Randomize button to populate all fields with random values
  4. CSV batch upload for bulk predictions
  5. ONNX model inference with real-time results
  6. Confidence gauges, severity indicators, and fault distribution charts
  7. Email alert configuration & one-click sending
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import onnxruntime as ort
from datetime import datetime

# ─── page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Caterpillar Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS — LIGHT THEME ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #f8f9fc 0%, #eef1f8 50%, #f0f2f8 100%);
}

/* header card */
.hero-card {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #6366f1 100%);
    border: none;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.25);
}
.hero-card h1 {
    color: #fff;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-card p {
    color: rgba(255,255,255,0.75);
    font-size: 14px;
    margin: 8px 0 0 0;
}

/* metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-card .label {
    color: #6b7280;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-card .value {
    color: #1f2937;
    font-size: 28px;
    font-weight: 700;
}

/* severity badges */
.badge-high { background: #ef4444; color: #fff; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; display: inline-block; }
.badge-early { background: #f59e0b; color: #1a1a2e; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; display: inline-block; }
.badge-normal { background: #10b981; color: #fff; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; display: inline-block; }

/* feature input card */
.feature-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.feature-card .feat-label {
    color: #374151;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1f2937 !important;
}

/* input labels */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* buttons — primary */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* randomize button styling via key */
div[data-testid="stButton"][class*="randomize"] > button {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
}

/* section headers */
.section-header {
    color: #1f2937;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}
.section-sub {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ─── load model & metadata ─────────────────────────────────────────────────

MODELS_DIR = "models"
ONNX_PATH = os.path.join(MODELS_DIR, "fault_detector.onnx")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

FEATURE_COLS = [
    "rms", "peak_to_peak", "kurtosis", "skewness", "crest_factor",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "dominant_frequency", "frequency_rms", "entropy", "impulse_factor",
    "clearance_factor", "band_energy_1_5kHz", "snr_estimated",
]

# (label, slider_min, slider_max, default, step)
FEATURE_DESCRIPTIONS = {
    "rms":                 ("RMS Amplitude",            0.0, 10.0,    0.5,     0.01),
    "peak_to_peak":        ("Peak-to-Peak",             0.0, 30.0,    1.5,     0.01),
    "kurtosis":            ("Kurtosis",               -10.0, 50.0,    3.0,     0.1),
    "skewness":            ("Skewness",               -10.0, 10.0,    0.0,     0.01),
    "crest_factor":        ("Crest Factor",             0.0, 20.0,    3.0,     0.01),
    "spectral_centroid":   ("Spectral Centroid (Hz)",   0.0, 10000.0, 1500.0,  1.0),
    "spectral_bandwidth":  ("Spectral Bandwidth (Hz)",  0.0, 8000.0,  800.0,   1.0),
    "spectral_rolloff":    ("Spectral Roll-off (Hz)",   0.0, 10000.0, 2500.0,  1.0),
    "dominant_frequency":  ("Dominant Frequency (Hz)",  0.0, 10000.0, 500.0,   1.0),
    "frequency_rms":       ("Frequency-Domain RMS",     0.0, 500.0,   10.0,    0.1),
    "entropy":             ("Signal Entropy",           0.0, 10.0,    3.0,     0.01),
    "impulse_factor":      ("Impulse Factor",           0.0, 30.0,    3.0,     0.01),
    "clearance_factor":    ("Clearance Factor",         0.0, 50.0,    4.0,     0.01),
    "band_energy_1_5kHz":  ("Band Energy 1-5 kHz",     0.0, 200000.0, 5000.0, 10.0),
    "snr_estimated":       ("Estimated SNR (dB)",     -30.0, 80.0,    15.0,    0.1),
}


@st.cache_resource
def load_onnx_session():
    if not os.path.exists(ONNX_PATH):
        return None
    return ort.InferenceSession(ONNX_PATH)


@st.cache_resource
def load_metadata():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH, "r") as f:
        return json.load(f)


def predict(session, features: np.ndarray, meta: dict) -> dict:
    """Run ONNX inference and return structured result."""
    # standardise using saved scaler params
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scale = np.array(meta["scaler_scale"], dtype=np.float32)
    x = ((features - mean) / scale).reshape(1, -1).astype(np.float32)

    cls_logits, ef_logit = session.run(None, {"features": x})

    # softmax
    exp = np.exp(cls_logits[0] - np.max(cls_logits[0]))
    probs = exp / exp.sum()

    pred_idx = int(np.argmax(probs))
    pred_class = meta["label_classes"][pred_idx]
    confidence = float(probs[pred_idx])

    # early-fault sigmoid
    ef_prob = float(1 / (1 + np.exp(-ef_logit[0][0])))
    is_early = ef_prob > 0.5

    return {
        "fault_class": pred_class,
        "confidence": confidence,
        "probabilities": {meta["label_classes"][i]: float(probs[i]) for i in range(len(probs))},
        "early_fault_probability": ef_prob,
        "is_early_fault": is_early,
    }


# ─── session state init for feature values ─────────────────────────────────
# Use widget keys directly as the source of truth.
# Initialize them once so both number_input and slider start in sync.

for _feat in FEATURE_COLS:
    _default = FEATURE_DESCRIPTIONS[_feat][3]
    if f"num_{_feat}" not in st.session_state:
        st.session_state[f"num_{_feat}"] = _default
    if f"slider_{_feat}" not in st.session_state:
        st.session_state[f"slider_{_feat}"] = _default


def _sync_from_number(feat_key):
    """Callback: when number_input changes, push value into the slider key."""
    val = st.session_state[f"num_{feat_key}"]
    desc, s_min, s_max, _, _ = FEATURE_DESCRIPTIONS[feat_key]
    # clamp for the slider (slider has range limits), but keep number unrestricted
    st.session_state[f"slider_{feat_key}"] = float(np.clip(val, s_min, s_max))


def _sync_from_slider(feat_key):
    """Callback: when slider changes, push value into the number_input key."""
    st.session_state[f"num_{feat_key}"] = st.session_state[f"slider_{feat_key}"]


def randomize_features():
    """Randomize all feature values — sets BOTH widget keys directly."""
    rng = np.random.default_rng()
    random_ranges = {
        "rms":                (0.1,  4.0),
        "peak_to_peak":       (0.5,  20.0),
        "kurtosis":           (-2.0, 30.0),
        "skewness":           (-2.0, 2.0),
        "crest_factor":       (1.5,  8.0),
        "spectral_centroid":  (200.0, 4500.0),
        "spectral_bandwidth": (100.0, 3500.0),
        "spectral_rolloff":   (500.0, 5000.0),
        "dominant_frequency": (10.0,  5000.0),
        "frequency_rms":      (1.0,   80.0),
        "entropy":            (1.5,  5.0),
        "impulse_factor":     (1.5,  12.0),
        "clearance_factor":   (1.5,  15.0),
        "band_energy_1_5kHz": (100.0, 80000.0),
        "snr_estimated":      (-5.0,  40.0),
    }
    for feat in FEATURE_COLS:
        lo, hi = random_ranges[feat]
        val = round(float(rng.uniform(lo, hi)), 4)
        st.session_state[f"num_{feat}"] = val
        desc, s_min, s_max, _, _ = FEATURE_DESCRIPTIONS[feat]
        st.session_state[f"slider_{feat}"] = float(np.clip(val, s_min, s_max))


# ─── sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    mode = st.radio("Input Mode", ["Manual Input", "CSV Upload"], index=0)

    st.markdown("---")
    st.markdown("### 📧 Email Settings")
    email_enabled = st.toggle("Enable Email Alerts", value=False)
    if email_enabled:
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
        smtp_port = st.number_input("Port", value=587, step=1)
        email_user = st.text_input("Username / Email")
        email_pass = st.text_input("Password", type="password")
        recipients = st.text_area("Recipients (one per line)", "maintainer@example.com")
        from_addr = st.text_input("From Address", "predictive.maintenance@caterpillar.com")

    st.markdown("---")
    st.markdown("### 🏭 Machine Context")
    machine_id = st.selectbox("Machine ID", [
        "CAT-EX-001", "CAT-EX-002", "CAT-TB-001", "CAT-TB-002", "CAT-DZ-001",
    ])
    sensor_loc = st.selectbox("Sensor Location", [
        "bearing_left", "bearing_right", "gearbox",
        "turbine_blade", "shaft_coupling", "motor_housing",
    ])


# ─── header ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-card">
    <h1>⚙️ Caterpillar Predictive Maintenance</h1>
    <p>CNN-LSTM Deep Learning Model &bull; Real-Time Fault Detection &bull; Automated Alerts</p>
</div>
""", unsafe_allow_html=True)


# ─── main content ──────────────────────────────────────────────────────────

session = load_onnx_session()
meta = load_metadata()

if session is None or meta is None:
    st.warning(
        "Model files not found. Please run the training pipeline first:\n\n"
        "```bash\n"
        "python src/generate_dataset.py\n"
        "python src/train.py\n"
        "python src/export_onnx.py\n"
        "```"
    )
    st.stop()


# ── Manual Input Mode ──

if mode == "Manual Input":
    st.markdown('<div class="section-header">📊 Enter Vibration Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Provide the 15 engineered features from the vibration sensor reading. Use the sliders or type any value directly.</div>', unsafe_allow_html=True)

    # ── Randomize button ──
    col_rand, col_spacer = st.columns([1, 3])
    with col_rand:
        if st.button("🎲 Randomize All Inputs", use_container_width=True, key="randomize_btn"):
            randomize_features()
            st.rerun()

    st.markdown("")

    # ── Feature input grid: 3 columns ──
    cols = st.columns(3)
    for i, feat in enumerate(FEATURE_COLS):
        desc, slider_min, slider_max, default, step_val = FEATURE_DESCRIPTIONS[feat]

        with cols[i % 3]:
            st.markdown(f'<div class="feature-card"><div class="feat-label">{desc}</div></div>', unsafe_allow_html=True)

            # number input — NO min/max range barrier
            # The key IS the source of truth; Streamlit reads the value from session_state[key]
            st.number_input(
                desc,
                value=float(default),  # only used on very first render
                step=step_val,
                format="%.4f",
                key=f"num_{feat}",
                on_change=_sync_from_number,
                args=(feat,),
                label_visibility="collapsed",
            )

            # slider below — synced bidirectionally
            st.slider(
                f"{desc} slider",
                min_value=float(slider_min),
                max_value=float(slider_max),
                value=float(default),  # only used on very first render
                step=step_val,
                key=f"slider_{feat}",
                on_change=_sync_from_slider,
                args=(feat,),
                label_visibility="collapsed",
            )

    st.markdown("---")

    if st.button("🔍 Run Prediction", use_container_width=True):
        # read values from the number_input widget keys (unrestricted source of truth)
        features = np.array([st.session_state[f"num_{f}"] for f in FEATURE_COLS], dtype=np.float32)
        result = predict(session, features, meta)

        # ── results display ──
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            fault_display = result["fault_class"].replace("_", " ").title()
            badge_class = "badge-high" if "severe" in result["fault_class"] or "combined" in result["fault_class"] \
                else ("badge-early" if "early" in result["fault_class"] else "badge-normal")
            sev_text = "SEVERE" if "severe" in result["fault_class"] or "combined" in result["fault_class"] \
                else ("EARLY" if "early" in result["fault_class"] else "NORMAL")
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Detected Fault</div>
                <div class="value" style="font-size: 18px;">{fault_display}</div>
                <div style="margin-top: 8px;"><span class="{badge_class}">{sev_text}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            conf_color = "#10b981" if result["confidence"] > 0.8 else "#f59e0b"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value" style="color: {conf_color};">{result['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            ef_color = "#ef4444" if result["early_fault_probability"] > 0.5 else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Early Fault Prob.</div>
                <div class="value" style="color: {ef_color};">{result['early_fault_probability']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Machine</div>
                <div class="value" style="font-size: 16px;">{machine_id}<br/><span style="font-size: 12px; color: #6b7280;">{sensor_loc.replace('_',' ').title()}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── confidence gauge ──
        st.markdown("#### Confidence Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["confidence"] * 100,
            number={"suffix": "%", "font": {"color": "#1f2937"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                "bar": {"color": "#6366f1"},
                "bgcolor": "#f3f4f6",
                "bordercolor": "#d1d5db",
                "steps": [
                    {"range": [0, 50], "color": "#fef2f2"},
                    {"range": [50, 80], "color": "#fefce8"},
                    {"range": [80, 100], "color": "#f0fdf4"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
            title={"text": "Model Confidence", "font": {"color": "#6b7280", "size": 14}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=40, b=20, l=40, r=40),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── class probabilities bar chart ──
        st.markdown("#### Class Probability Distribution")
        prob_df = pd.DataFrame(
            list(result["probabilities"].items()),
            columns=["Fault Class", "Probability"],
        ).sort_values("Probability", ascending=True)

        fig_bar = px.bar(
            prob_df, x="Probability", y="Fault Class", orientation="h",
            color="Probability",
            color_continuous_scale=["#e0e7ff", "#6366f1", "#ef4444"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#374151",
            height=350,
            margin=dict(t=20, b=20),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── email alert ──
        if result["fault_class"] != "normal":
            st.markdown("---")
            st.markdown("### 📧 Send Maintenance Alert")

            if email_enabled:
                if st.button("📨 Send Alert Email Now", use_container_width=True):
                    from src.notifier import send_alert
                    email_cfg = {
                        "smtp_server": smtp_server,
                        "smtp_port": smtp_port,
                        "use_tls": True,
                        "username": email_user,
                        "password": email_pass,
                        "from_address": from_addr,
                        "recipients": [r.strip() for r in recipients.strip().split("\n") if r.strip()],
                        "subject_prefix": "[FAULT ALERT]",
                    }
                    success = send_alert(
                        fault_class=result["fault_class"],
                        confidence=result["confidence"],
                        severity=result["early_fault_probability"],
                        machine_id=machine_id,
                        sensor_location=sensor_loc,
                        email_cfg=email_cfg,
                    )
                    if success:
                        st.success("Alert email sent successfully!")
                    else:
                        st.error("Failed to send email. Check your SMTP settings.")
            else:
                st.info("Enable Email Alerts in the sidebar to send notifications.")


# ── CSV Upload Mode ──

elif mode == "CSV Upload":
    st.markdown('<div class="section-header">📁 Upload Sensor Data CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV file with the 15 vibration feature columns for batch prediction.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Rows loaded:** {len(df)}")

        # check columns
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        st.dataframe(df.head(10), use_container_width=True)

        if st.button("🔍 Run Batch Prediction", use_container_width=True):
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                features = np.array([row[f] for f in FEATURE_COLS], dtype=np.float32)
                res = predict(session, features, meta)
                res["row_index"] = i
                results.append(res)
                progress.progress((i + 1) / len(df))

            res_df = pd.DataFrame(results)
            st.markdown("### 📊 Batch Results")

            # summary metrics
            c1, c2, c3 = st.columns(3)
            fault_count = (res_df["fault_class"] != "normal").sum()
            early_count = res_df["is_early_fault"].sum()
            with c1:
                st.metric("Total Samples", len(res_df))
            with c2:
                st.metric("Faults Detected", int(fault_count))
            with c3:
                st.metric("Early-Stage Faults", int(early_count))

            # distribution pie
            class_counts = res_df["fault_class"].value_counts()
            fig_pie = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#374151",
                height=400,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # results table
            display_df = res_df[["row_index", "fault_class", "confidence", "early_fault_probability", "is_early_fault"]]
            st.dataframe(display_df, use_container_width=True)

            # download
            csv = display_df.to_csv(index=False)
            st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")

            # email alert for faults
            if fault_count > 0 and email_enabled:
                st.markdown("---")
                if st.button("📨 Send Batch Alert Email", use_container_width=True):
                    from src.notifier import send_alert
                    email_cfg = {
                        "smtp_server": smtp_server,
                        "smtp_port": smtp_port,
                        "use_tls": True,
                        "username": email_user,
                        "password": email_pass,
                        "from_address": from_addr,
                        "recipients": [r.strip() for r in recipients.strip().split("\n") if r.strip()],
                        "subject_prefix": "[FAULT ALERT - BATCH]",
                    }
                    # send one alert for the most severe fault
                    worst = res_df.loc[res_df["confidence"].idxmax()]
                    send_alert(
                        fault_class=worst["fault_class"],
                        confidence=worst["confidence"],
                        severity=worst["early_fault_probability"],
                        machine_id=machine_id,
                        sensor_location=sensor_loc,
                        email_cfg=email_cfg,
                    )
                    st.success(f"Alert sent for {int(fault_count)} detected faults.")


# ─── footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9ca3af; font-size: 12px;'>"
    "Caterpillar Inc. - Predictive Maintenance Health Monitoring System | "
    f"Built with Streamlit | {datetime.now().year}"
    "</p>",
    unsafe_allow_html=True,
)
