"""
Streamlit Frontend — Predictive Maintenance Dashboard
======================================================
A premium, dark-themed dashboard that provides:
  1. Manual feature input form (15 vibration features + metadata)
  2. CSV batch upload for bulk predictions
  3. ONNX model inference with real-time results
  4. Confidence gauges, severity indicators, and fault distribution charts
  5. Email alert configuration & one-click sending
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

# ─── custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #0a0a1a 0%, #111128 50%, #0d0d20 100%);
}

/* header card */
.hero-card {
    background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 50%, #1a1a3e 100%);
    border: 1px solid rgba(120, 80, 255, 0.2);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(80, 40, 200, 0.15);
}
.hero-card h1 {
    color: #fff;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-card p {
    color: #a0a0d0;
    font-size: 14px;
    margin: 8px 0 0 0;
}

/* metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card .label {
    color: #888;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-card .value {
    color: #fff;
    font-size: 28px;
    font-weight: 700;
}

/* severity badges */
.badge-high { background: #ff3333; color: #fff; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; }
.badge-early { background: #ffa500; color: #1a1a2e; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; }
.badge-normal { background: #33cc66; color: #1a1a2e; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d1f;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* input labels */
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #c0c0e0 !important;
    font-weight: 500 !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c3fe0, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7c4ff0, #9b6cf6) !important;
    box-shadow: 0 4px 20px rgba(108, 63, 224, 0.4) !important;
    transform: translateY(-1px) !important;
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

FEATURE_DESCRIPTIONS = {
    "rms": ("RMS Amplitude", 0.0, 5.0, 0.5),
    "peak_to_peak": ("Peak-to-Peak", 0.0, 15.0, 1.5),
    "kurtosis": ("Kurtosis", -3.0, 20.0, 3.0),
    "skewness": ("Skewness", -3.0, 3.0, 0.0),
    "crest_factor": ("Crest Factor", 1.0, 10.0, 3.0),
    "spectral_centroid": ("Spectral Centroid (Hz)", 0.0, 5000.0, 1500.0),
    "spectral_bandwidth": ("Spectral Bandwidth (Hz)", 0.0, 3000.0, 800.0),
    "spectral_rolloff": ("Spectral Roll-off (Hz)", 0.0, 5000.0, 2500.0),
    "dominant_frequency": ("Dominant Frequency (Hz)", 0.0, 5000.0, 500.0),
    "frequency_rms": ("Frequency-Domain RMS", 0.0, 100.0, 10.0),
    "entropy": ("Signal Entropy", 0.0, 6.0, 3.0),
    "impulse_factor": ("Impulse Factor", 1.0, 15.0, 3.0),
    "clearance_factor": ("Clearance Factor", 1.0, 20.0, 4.0),
    "band_energy_1_5kHz": ("Band Energy 1-5 kHz", 0.0, 50000.0, 5000.0),
    "snr_estimated": ("Estimated SNR (dB)", -10.0, 50.0, 15.0),
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
    <p>CNN-LSTM Deep Learning Model • Real-Time Fault Detection • Automated Alerts</p>
</div>
""", unsafe_allow_html=True)


# ─── main content ──────────────────────────────────────────────────────────

session = load_onnx_session()
meta = load_metadata()

if session is None or meta is None:
    st.warning(
        "⚠️ Model files not found. Please run the training pipeline first:\n\n"
        "```bash\n"
        "python src/generate_dataset.py\n"
        "python src/train.py\n"
        "python src/export_onnx.py\n"
        "```"
    )
    st.stop()


# ── Manual Input Mode ──

if mode == "Manual Input":
    st.markdown("### 📊 Enter Vibration Features")
    st.markdown("Provide the 15 engineered features from the vibration sensor reading.")

    cols = st.columns(3)
    feature_values = {}
    for i, feat in enumerate(FEATURE_COLS):
        desc, min_v, max_v, default = FEATURE_DESCRIPTIONS[feat]
        with cols[i % 3]:
            feature_values[feat] = st.number_input(
                desc, min_value=min_v, max_value=max_v, value=default,
                step=(max_v - min_v) / 100, format="%.4f", key=feat,
            )

    st.markdown("---")

    if st.button("🔍 Run Prediction", use_container_width=True):
        features = np.array([feature_values[f] for f in FEATURE_COLS], dtype=np.float32)
        result = predict(session, features, meta)

        # ── results display ──
        st.markdown("### 🎯 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            fault_display = result["fault_class"].replace("_", " ").title()
            badge_class = "badge-high" if "severe" in result["fault_class"] or "combined" in result["fault_class"] \
                else ("badge-early" if "early" in result["fault_class"] else "badge-normal")
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Detected Fault</div>
                <div class="value" style="font-size: 18px;">{fault_display}</div>
                <div style="margin-top: 8px;"><span class="{badge_class}">
                    {"SEVERE" if "severe" in result["fault_class"] or "combined" in result["fault_class"]
                     else ("EARLY" if "early" in result["fault_class"] else "NORMAL")}
                </span></div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value" style="color: {'#33cc66' if result['confidence'] > 0.8 else '#ffa500'};">
                    {result['confidence']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Early Fault Prob.</div>
                <div class="value" style="color: {'#ff3333' if result['early_fault_probability'] > 0.5 else '#33cc66'};">
                    {result['early_fault_probability']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Machine</div>
                <div class="value" style="font-size: 16px;">{machine_id}<br/><span style="font-size: 12px; color: #888;">{sensor_loc.replace('_',' ').title()}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── confidence gauge ──
        st.markdown("#### Confidence Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["confidence"] * 100,
            number={"suffix": "%", "font": {"color": "#fff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#555"},
                "bar": {"color": "#8b5cf6"},
                "bgcolor": "#1a1a3e",
                "bordercolor": "#333",
                "steps": [
                    {"range": [0, 50], "color": "#2a1a3e"},
                    {"range": [50, 80], "color": "#3a2a4e"},
                    {"range": [80, 100], "color": "#4a3a5e"},
                ],
                "threshold": {
                    "line": {"color": "#ff3333", "width": 3},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
            title={"text": "Model Confidence", "font": {"color": "#a0a0d0", "size": 14}},
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
            color_continuous_scale=["#1a1a3e", "#6c3fe0", "#ff3333"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c0c0e0",
            height=350,
            margin=dict(t=20, b=20),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
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
                        st.success("✅ Alert email sent successfully!")
                    else:
                        st.error("❌ Failed to send email. Check your SMTP settings.")
            else:
                st.info("💡 Enable Email Alerts in the sidebar to send notifications.")


# ── CSV Upload Mode ──

elif mode == "CSV Upload":
    st.markdown("### 📁 Upload Sensor Data CSV")
    st.markdown("Upload a CSV file with the 15 vibration feature columns for batch prediction.")

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
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c0c0e0",
                height=400,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # results table
            display_df = res_df[["row_index", "fault_class", "confidence", "early_fault_probability", "is_early_fault"]]
            st.dataframe(display_df, use_container_width=True)

            # download
            csv = display_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv, "predictions.csv", "text/csv")

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
                        "subject_prefix": "[FAULT ALERT — BATCH]",
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
                    st.success(f"✅ Alert sent for {int(fault_count)} detected faults.")


# ─── footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #555; font-size: 12px;'>"
    "Caterpillar Inc. — Predictive Maintenance Health Monitoring System • "
    f"Built with Streamlit • {datetime.now().year}"
    "</p>",
    unsafe_allow_html=True,
)
