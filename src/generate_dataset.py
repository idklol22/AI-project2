"""
Synthetic Vibration Dataset Generator
=====================================
Generates realistic high-frequency vibration sensor data for heavy machinery
(turbines, excavators, dozers). Each sample is a raw time-domain signal from
which 15 engineered features are extracted. The final output is a single CSV
with metadata, features, and detailed fault labels.

Fault Classes (8):
  normal, imbalance_early, imbalance_severe, misalignment_early,
  misalignment_severe, bearing_wear_early, bearing_wear_severe, combined_fault
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew, entropy as sp_entropy
import yaml
from datetime import datetime, timedelta


# ─── helpers ────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── raw signal generators ─────────────────────────────────────────────────

def _base_signal(n: int, fs: int, rpm: float, rng: np.random.Generator) -> np.ndarray:
    """Healthy baseline: sum of harmonics of the shaft rotation frequency."""
    f_rot = rpm / 60.0  # rotation frequency in Hz
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for h in range(1, 5):
        amp = rng.uniform(0.3, 1.0) / h
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * h * f_rot * t + phase)
    # small broadband component
    sig += 0.05 * rng.normal(size=n)
    return sig


def _inject_imbalance(sig: np.ndarray, fs: int, rpm: float, severity: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Add a strong 1× rotational frequency component."""
    t = np.arange(len(sig)) / fs
    f_rot = rpm / 60.0
    amp = severity * rng.uniform(1.5, 3.0)
    sig += amp * np.sin(2 * np.pi * f_rot * t + rng.uniform(0, 2 * np.pi))
    return sig


def _inject_misalignment(sig: np.ndarray, fs: int, rpm: float, severity: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Add 2× and 3× harmonics (axial vibration pattern)."""
    t = np.arange(len(sig)) / fs
    f_rot = rpm / 60.0
    for h in [2, 3]:
        amp = severity * rng.uniform(1.0, 2.5) / h
        sig += amp * np.sin(2 * np.pi * h * f_rot * t + rng.uniform(0, 2 * np.pi))
    return sig


def _inject_bearing_wear(sig: np.ndarray, fs: int, rpm: float, severity: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Add high-frequency periodic bursts (simulating roller-element defects)."""
    n = len(sig)
    t = np.arange(n) / fs
    # Bearing defect frequency ≈ 3–8× shaft speed
    f_def = rng.uniform(3, 8) * (rpm / 60.0)
    burst_env = 0.5 * (1 + np.sin(2 * np.pi * f_def * t))
    hf_carrier = rng.uniform(2000, 4500)  # Hz
    burst = severity * rng.uniform(1.0, 2.0) * burst_env * np.sin(2 * np.pi * hf_carrier * t)
    sig += burst
    return sig


def _add_noise(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian white noise + occasional impulse noise."""
    # white noise
    power_sig = np.mean(sig ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(power_noise), size=len(sig))
    # impulse noise (sparse)
    impulse_mask = rng.random(size=len(sig)) < 0.001
    impulse = impulse_mask * rng.normal(0, np.sqrt(power_sig) * 3, size=len(sig))
    return sig + noise + impulse


# ─── feature extraction ────────────────────────────────────────────────────

def extract_features(sig: np.ndarray, fs: int) -> dict:
    """Extract 15 engineered features from a 1-D vibration signal."""
    n = len(sig)

    # time-domain
    rms = np.sqrt(np.mean(sig ** 2))
    peak_to_peak = np.max(sig) - np.min(sig)
    kurt = float(kurtosis(sig, fisher=True))
    skewness = float(skew(sig))
    crest_factor = np.max(np.abs(sig)) / (rms + 1e-12)
    impulse_factor = np.max(np.abs(sig)) / (np.mean(np.abs(sig)) + 1e-12)
    mean_abs = np.mean(np.sqrt(np.abs(sig))) ** 2
    clearance_factor = np.max(np.abs(sig)) / (mean_abs + 1e-12)

    # frequency-domain
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(sig))
    mag_norm = mag / (np.sum(mag) + 1e-12)

    spectral_centroid = float(np.sum(freqs * mag_norm))
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag_norm)))
    cum_energy = np.cumsum(mag ** 2)
    total_energy = cum_energy[-1] + 1e-12
    spectral_rolloff = float(freqs[np.searchsorted(cum_energy, 0.85 * total_energy)])
    dominant_frequency = float(freqs[np.argmax(mag)])
    frequency_rms = float(np.sqrt(np.mean(mag ** 2)))

    # entropy
    sig_hist, _ = np.histogram(sig, bins=64, density=True)
    sig_hist = sig_hist[sig_hist > 0]
    ent = float(sp_entropy(sig_hist))

    # band energy (1 kHz – 5 kHz)
    band_mask = (freqs >= 1000) & (freqs <= 5000)
    band_energy = float(np.sum(mag[band_mask] ** 2))

    # SNR estimate
    signal_power = np.mean(sig ** 2)
    noise_est = np.median(mag) ** 2
    snr_est = float(10 * np.log10(signal_power / (noise_est + 1e-12) + 1e-12))

    return {
        "rms": round(rms, 6),
        "peak_to_peak": round(peak_to_peak, 6),
        "kurtosis": round(kurt, 6),
        "skewness": round(skewness, 6),
        "crest_factor": round(crest_factor, 6),
        "spectral_centroid": round(spectral_centroid, 4),
        "spectral_bandwidth": round(spectral_bandwidth, 4),
        "spectral_rolloff": round(spectral_rolloff, 4),
        "dominant_frequency": round(dominant_frequency, 4),
        "frequency_rms": round(frequency_rms, 6),
        "entropy": round(ent, 6),
        "impulse_factor": round(impulse_factor, 6),
        "clearance_factor": round(clearance_factor, 6),
        "band_energy_1_5kHz": round(band_energy, 4),
        "snr_estimated": round(snr_est, 4),
    }


# ─── sample generation ─────────────────────────────────────────────────────

FAULT_CLASSES = [
    "normal",
    "imbalance_early",
    "imbalance_severe",
    "misalignment_early",
    "misalignment_severe",
    "bearing_wear_early",
    "bearing_wear_severe",
    "combined_fault",
]

# class weights for balanced generation
CLASS_WEIGHTS = [0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20]


def generate_single_sample(
    fault_class: str,
    n: int,
    fs: int,
    rpm: float,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, bool]:
    """Generate a single raw signal with injected fault. Returns (signal, severity, is_early)."""
    sig = _base_signal(n, fs, rpm, rng)
    severity = 0.0
    is_early = False

    if fault_class == "normal":
        severity = 0.0
    elif fault_class == "imbalance_early":
        severity = rng.uniform(0.1, 0.35)
        is_early = True
        sig = _inject_imbalance(sig, fs, rpm, severity, rng)
    elif fault_class == "imbalance_severe":
        severity = rng.uniform(0.6, 1.0)
        sig = _inject_imbalance(sig, fs, rpm, severity, rng)
    elif fault_class == "misalignment_early":
        severity = rng.uniform(0.1, 0.35)
        is_early = True
        sig = _inject_misalignment(sig, fs, rpm, severity, rng)
    elif fault_class == "misalignment_severe":
        severity = rng.uniform(0.6, 1.0)
        sig = _inject_misalignment(sig, fs, rpm, severity, rng)
    elif fault_class == "bearing_wear_early":
        severity = rng.uniform(0.1, 0.35)
        is_early = True
        sig = _inject_bearing_wear(sig, fs, rpm, severity, rng)
    elif fault_class == "bearing_wear_severe":
        severity = rng.uniform(0.6, 1.0)
        sig = _inject_bearing_wear(sig, fs, rpm, severity, rng)
    elif fault_class == "combined_fault":
        severity = rng.uniform(0.4, 1.0)
        sig = _inject_imbalance(sig, fs, rpm, severity * rng.uniform(0.5, 1.0), rng)
        sig = _inject_misalignment(sig, fs, rpm, severity * rng.uniform(0.5, 1.0), rng)
        sig = _inject_bearing_wear(sig, fs, rpm, severity * rng.uniform(0.3, 0.8), rng)

    sig = _add_noise(sig, snr_db, rng)
    return sig, severity, is_early


def generate_dataset(cfg: dict) -> pd.DataFrame:
    """Generate the full synthetic dataset and return as a DataFrame."""
    data_cfg = cfg["data"]
    n_samples = data_cfg["num_samples"]
    n = data_cfg["sample_length"]
    fs = data_cfg["sampling_rate"]
    snr_lo, snr_hi = data_cfg["snr_range"]
    machines = data_cfg["machines"]
    locations = data_cfg["sensor_locations"]
    seed = data_cfg["random_seed"]

    rng = np.random.default_rng(seed)

    # assign classes with approximate weights
    probs = np.array(CLASS_WEIGHTS, dtype=float)
    probs /= probs.sum()
    class_indices = rng.choice(len(FAULT_CLASSES), size=n_samples, p=probs)

    rows = []
    base_time = datetime(2025, 1, 1, 0, 0, 0)

    print(f"Generating {n_samples} samples ...")
    for i in range(n_samples):
        fault_class = FAULT_CLASSES[class_indices[i]]
        rpm = rng.uniform(500, 3000)
        load = rng.uniform(20, 100)
        temp = rng.uniform(40, 120)
        snr_db = rng.uniform(snr_lo, snr_hi)
        machine = rng.choice(machines)
        location = rng.choice(locations)
        ts = base_time + timedelta(minutes=int(i * rng.uniform(1, 15)))

        sig, severity, is_early = generate_single_sample(fault_class, n, fs, rpm, snr_db, rng)
        feats = extract_features(sig, fs)

        row = {
            "machine_id": machine,
            "sensor_location": location,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "operating_speed_rpm": round(rpm, 1),
            "load_percentage": round(load, 1),
            "temperature_celsius": round(temp, 1),
            **feats,
            "fault_class": fault_class,
            "is_early_fault": int(is_early),
            "fault_severity": round(severity, 4),
        }
        rows.append(row)

        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{n_samples}] generated")

    df = pd.DataFrame(rows)
    print(f"Dataset shape: {df.shape}")
    return df


def main():
    cfg = load_config()
    df = generate_dataset(cfg)

    out_path = cfg["data"]["output_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # quick stats
    print("\nClass distribution:")
    print(df["fault_class"].value_counts().to_string())
    print(f"\nEarly-fault samples: {df['is_early_fault'].sum()}")
    print(f"Feature columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
