import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import smtplib
import urllib.parse

from src.models import (
    load_model, 
    FEATURE_COLS, 
    FAULT_LABELS, 
    ALL_LABELS
)

# Constants
MODEL_DIR = "artifacts_hierarchical"
MODEL_PATH = os.path.join(MODEL_DIR, "hierarchical_cnn_lstm.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_model(MODEL_PATH, device=device)
    scaler = joblib.load(SCALER_PATH)
    return model, checkpoint, scaler

model, checkpoint, scaler = load_assets()

if model is None:
    st.error("Model artifacts not found! Please ensure 'a.py' has been run to train and save the model.")
    st.stop()

st.title("Predictive Maintenance Diagnostic Dashboard")

# Feature ranges strictly within the min/max bounds present in synthetic_vibration_data.csv
RANGES = {
    "machine_id_encoded": (0.0, 4.0),
    "operating_speed_rpm": (574.4, 3008.2),
    "load_percentage": (15.0, 100.0),
    "temperature_celsius": (35.0, 137.4),
    "rms": (0.1798, 2.7114),
    "peak_to_peak": (0.9347, 22.3976),
    "kurtosis": (-1.6356, 5.4025),
    "skewness": (-1.5817, 1.6560),
    "crest_factor": (1.5722, 11.3677),
    "spectral_centroid": (790.0052, 2798.1028),
    "spectral_bandwidth": (1311.8097, 1983.0100),
    "spectral_rolloff": (12.2070, 4890.1367),
    "dominant_frequency": (9.7656, 4699.7070),
    "frequency_rms": (11.5074, 173.5098),
    "entropy": (2.6264, 4.0976),
    "impulse_factor": (1.7240, 15.5052),
    "clearance_factor": (1.8631, 19.3171),
    "band_energy_1_5kHz": (3110.0379, 6267250.7292),
    "snr_estimated": (-27.9704, -7.3438),
}

HARDCODED_NORMAL = {
    "machine_id_encoded": 1.0,
    "operating_speed_rpm": 1794.9,
    "load_percentage": 48.5,
    "temperature_celsius": 86.8,
    "rms": 0.39956,
    "peak_to_peak": 2.472235,
    "kurtosis": 0.145224,
    "skewness": -0.618266,
    "crest_factor": 3.755717,
    "spectral_centroid": 1362.514,
    "spectral_bandwidth": 1579.7778,
    "spectral_rolloff": 119.6289,
    "dominant_frequency": 29.2969,
    "frequency_rms": 25.565756,
    "entropy": 3.666511,
    "impulse_factor": 4.849179,
    "clearance_factor": 5.841148,
    "band_energy_1_5kHz": 21809.5097,
    "snr_estimated": -18.1239
}

# Ensure hardcoded normal values stay within ranges
for k, v in HARDCODED_NORMAL.items():
    min_v, max_v = RANGES[k]
    HARDCODED_NORMAL[k] = max(min_v, min(max_v, v))

if "features" not in st.session_state:
    st.session_state.features = {k: float(RANGES[k][0]) for k in FEATURE_COLS}

@st.cache_data
def load_sample_data():
    if os.path.exists("synthetic_vibration_data.csv"):
        df = pd.read_csv("synthetic_vibration_data.csv")
        normal_df = df[df['fault_present'] == 0]
        faulty_df = df[df['fault_present'] == 1]
        return normal_df, faulty_df
    return None, None

normal_df, faulty_df = load_sample_data()

def randomize_features(mode="any"):
    if mode == "normal":
        for k in FEATURE_COLS:
            st.session_state.features[k] = float(HARDCODED_NORMAL[k])
    elif mode == "faulty" and faulty_df is not None and not faulty_df.empty:
        row = faulty_df.sample(1).iloc[0]
        for k in FEATURE_COLS:
            st.session_state.features[k] = float(row[k])
    else:
        for k in FEATURE_COLS:
            min_v, max_v = RANGES[k]
            st.session_state.features[k] = float(np.random.uniform(min_v, max_v))

def send_email(subject, body, to_email, sender_email, sender_password):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    if not sender_email or not sender_password:
        st.error("Please configure the SMTP Sender Email and Password in the sidebar.")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        st.error(f"Failed to send email. Verify your credentials. Error: {e}")
        return False

with st.sidebar:
    st.header("Email Configuration (Gmail)")
    st.session_state.sender_email = st.text_input("Sender Email", value=st.session_state.get("sender_email", ""))
    st.session_state.sender_password = st.text_input("App Password", value=st.session_state.get("sender_password", ""), type="password", help="Use a Gmail App Password, not your standard account password.")
    st.divider()
    
    st.header("Input Features")
    # Quick fill buttons
    c1, c2 = st.columns(2)
    with c1:
        st.button("Random Any", on_click=randomize_features, args=("any",), use_container_width=True)
        st.button("Set Faulty", on_click=randomize_features, args=("faulty",), use_container_width=True)
    with c2:
        st.button("Set Normal", on_click=randomize_features, args=("normal",), use_container_width=True)

    st.divider()

    for k in FEATURE_COLS:
        min_v, max_v = RANGES[k]
        current_val = float(st.session_state.features.get(k, min_v))
        val = st.number_input(
            label=k.replace('_', ' ').title(),
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(current_val),
            format="%.4f"
        )
        st.session_state.features[k] = val


# Run Inference
st.subheader("Model Inference")

if st.button("Predict", type="primary"):
    st.session_state.show_prediction = True

if st.session_state.get("show_prediction", False):
    # Sequence length that was used during training
    seq_len = checkpoint.get("seq_len", 24)
    feat_vector = np.array([st.session_state.features[k] for k in FEATURE_COLS], dtype=np.float32)
    
    # Broadcast to (seq_len, num_features) to simulate a steady window
    x_win = np.tile(feat_vector, (seq_len, 1))
    
    # Validate scaler features
    expected_features = scaler.n_features_in_
    if expected_features != len(FEATURE_COLS):
        st.warning(f"Scaler expects {expected_features} features, but {len(FEATURE_COLS)} provided. This might cause shape errors if dataset feature cols changed.")
    
    try:
        # Scale
        x_win_scaled = scaler.transform(x_win)
        
        # Batch dimension
        x_tensor = torch.tensor(x_win_scaled, dtype=torch.float32).unsqueeze(0) # [1, 24, 19]
        
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        with torch.no_grad():
            bin_logit, multi_logit = model(x_tensor)
            bin_prob = 1.0 / (1.0 + np.exp(-bin_logit.cpu().numpy()))[0]
            multi_logits_arr = multi_logit.cpu().numpy()[0]
            multi_probs = np.exp(multi_logits_arr) / np.sum(np.exp(multi_logits_arr))
            
        threshold = checkpoint.get("threshold", 0.5)
        pred_class = FAULT_LABELS[np.argmax(multi_probs)]
        human_readable_class = pred_class.replace('_', ' ').title()
        
        # Display Results
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("### Structural Integrity Assessment")
            is_fault = bin_prob >= threshold
            if is_fault:
                st.error(f"ALERT: Structural Anomaly Detected\nProbability: {bin_prob:.4f} (Threshold: {threshold:.4f})")
                
                st.markdown("---")
                st.markdown("#### Automated Alert Dispatch")
                with st.form("email_dispatch_form"):
                    recipient = st.text_input("Maintainer Email", value="wadhwasanjam@gmail.com")
                    alert_body = f"System Check Failure:\nA predictive maintenance alert has been triggered for machine index {st.session_state.features.get('machine_id_encoded')}.\n\nAnomaly Probability: {bin_prob:.4f}\nPrimary Suspect Parameter: {human_readable_class}\n\nImmediate physical inspection required."
                    body = st.text_area("Alert Context", value=alert_body, height=170)
                    submitted = st.form_submit_button("Dispatch Security Alert")
                    if submitted:
                        if send_email("Alert: Potential Machine Fault Detected", body, recipient, st.session_state.sender_email, st.session_state.sender_password):
                            st.success(f"Alert successfully dispatched to {recipient}.")
                            
            else:
                st.success(f"SYSTEM NOMINAL\nProbability: {bin_prob:.4f} (Threshold: {threshold:.4f})")
                
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = bin_prob,
                title = {'text': "Anomaly Confidence"},
                delta = {'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred" if is_fault else "darkgreen"},
                    'steps': [
                        {'range': [0, threshold], 'color': "lightgreen"},
                        {'range': [threshold, 1], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            st.markdown("### Probabilistic Class Breakdown")
            
            fault_df = pd.DataFrame({
                "Fault Classification": [lbl.replace('_', ' ').title() for lbl in FAULT_LABELS],
                "Confidence": multi_probs
            })
            fault_df = fault_df.sort_values(by="Confidence", ascending=True)
            
            fig_bar = px.bar(fault_df, x="Confidence", y="Fault Classification", orientation='h')
            fig_bar.update_traces(marker_color='#4A90E2')
            fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            if is_fault:
                st.warning(f"Primary Suspect Parameter: {human_readable_class}")
            else:
                st.info("System operations are stable. Highlighted categorizations represent environmental artifact weighting.")
                
    except Exception as e:
        st.error(f"Inference execution failed: {e}")
