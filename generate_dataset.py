import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy as sp_entropy
import yaml
from datetime import datetime, timedelta


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


MACHINE_ID_MAP = {
    "CAT-EX-001": 0,
    "CAT-EX-002": 1,
    "CAT-TB-001": 2,
    "CAT-TB-002": 3,
    "CAT-DZ-001": 4,
}

# (base_rpm_offset, base_snr_offset, base_amplitude_scale)
MACHINE_PROFILES = {
    "CAT-EX-001": (0.0,    0.0, 1.00),
    "CAT-EX-002": (60.0,  -1.2, 1.08),
    "CAT-TB-001": (-110.0, 2.2, 0.82),
    "CAT-TB-002": (-85.0,  1.0, 0.88),
    "CAT-DZ-001": (130.0, -2.8, 1.22),
}

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

FAULT_ONLY_CLASSES = [c for c in FAULT_CLASSES if c != "normal"]


def _effective_sample_count(n_samples: int, multiplier: float = 1.5) -> int:
    total = int(round(n_samples * multiplier))
    if total % 2 != 0:
        total += 1
    return total


def _exact_class_counts(total_samples: int) -> dict:
    normal_count = total_samples // 2
    faulty_total = total_samples - normal_count

    per_fault = faulty_total // len(FAULT_ONLY_CLASSES)
    remainder = faulty_total % len(FAULT_ONLY_CLASSES)

    counts = {"normal": normal_count}
    for i, cls in enumerate(FAULT_ONLY_CLASSES):
        counts[cls] = per_fault + (1 if i < remainder else 0)
    return counts


def _split_into_runs(count: int, min_len: int, max_len: int, rng: np.random.Generator):
    runs = []
    remaining = count
    while remaining > 0:
        if remaining <= max_len:
            runs.append(remaining)
            break
        run = int(rng.integers(min_len, max_len + 1))
        if remaining - run < min_len:
            run = remaining
        runs.append(run)
        remaining -= run
    return runs


def _build_episode_plan(class_counts: dict, min_len: int, max_len: int, rng: np.random.Generator):
    normal_runs = [("normal", r) for r in _split_into_runs(class_counts["normal"], min_len, max_len, rng)]

    fault_runs = []
    for cls in FAULT_ONLY_CLASSES:
        runs = _split_into_runs(class_counts[cls], min_len, max_len, rng)
        fault_runs.extend([(cls, r) for r in runs])

    rng.shuffle(normal_runs)
    rng.shuffle(fault_runs)

    plan = []
    i, j = 0, 0
    use_normal = True

    while i < len(normal_runs) or j < len(fault_runs):
        if use_normal and i < len(normal_runs):
            plan.append(normal_runs[i])
            i += 1
        elif (not use_normal) and j < len(fault_runs):
            plan.append(fault_runs[j])
            j += 1
        elif i < len(normal_runs):
            plan.append(normal_runs[i])
            i += 1
        elif j < len(fault_runs):
            plan.append(fault_runs[j])
            j += 1
        use_normal = not use_normal

    return plan


def _severity_bounds(fault_class: str):
    bounds = {
        "normal": (0.0, 0.0),
        "imbalance_early": (0.18, 0.30),
        "imbalance_severe": (0.62, 0.95),
        "misalignment_early": (0.16, 0.28),
        "misalignment_severe": (0.60, 0.92),
        "bearing_wear_early": (0.18, 0.32),
        "bearing_wear_severe": (0.62, 0.96),
        "combined_fault": (0.58, 0.96),
    }
    return bounds[fault_class]


def _episode_severity_curve(fault_class: str, length: int, rng: np.random.Generator):
    lo, hi = _severity_bounds(fault_class)
    if fault_class == "normal":
        return np.zeros(length, dtype=np.float32)

    start = rng.uniform(lo, min(lo + 0.05, hi))
    end = rng.uniform(max(start + 0.04, hi - 0.06), hi)
    curve = np.linspace(start, end, length)
    jitter = rng.normal(0, 0.01 + 0.015 * (hi - lo), size=length)
    curve = np.clip(curve + jitter, lo, hi)
    return curve.astype(np.float32)


def _base_signal(n: int, fs: int, rpm: float, amp_scale: float,
                 load: float, temp: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    f_rot = rpm / 60.0

    load_gain = 0.76 + 0.0070 * load
    temp_gain = 0.95 + 0.0022 * max(temp - 40.0, 0.0)

    sig = np.zeros(n)
    am = 1.0 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.20, 0.85) * t + rng.uniform(0, 2 * np.pi))

    for h in range(1, 8):
        amp = amp_scale * load_gain * rng.uniform(0.22, 0.82) / (h ** 1.10)
        phase = rng.uniform(0, 2 * np.pi)
        jitter = 1.0 + rng.uniform(-0.007, 0.007)
        sig += am * amp * np.sin(2 * np.pi * h * f_rot * jitter * t + phase)

    structural_f = rng.uniform(140, 520)
    sig += 0.10 * amp_scale * temp_gain * np.sin(
        2 * np.pi * structural_f * t + rng.uniform(0, 2 * np.pi)
    )
    sig += 0.025 * amp_scale * rng.normal(size=n)
    return sig


def _inject_imbalance(sig: np.ndarray, fs: int, rpm: float, severity: float,
                      load: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(sig)) / fs
    f_rot = rpm / 60.0

    mod = 1.0 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.18, 0.65) * t + rng.uniform(0, 2 * np.pi))
    main_amp = severity * (1.45 + 0.010 * load) * rng.uniform(1.05, 1.30)
    weak_2x = severity * rng.uniform(0.06, 0.16)
    low_sideband = severity * rng.uniform(0.03, 0.08)

    sig += mod * main_amp * np.sin(2 * np.pi * f_rot * t + rng.uniform(0, 2 * np.pi))
    sig += weak_2x * np.sin(2 * np.pi * 2.0 * f_rot * t + rng.uniform(0, 2 * np.pi))
    sig += low_sideband * np.sin(2 * np.pi * 0.5 * f_rot * t + rng.uniform(0, 2 * np.pi))
    return sig


def _inject_misalignment(sig: np.ndarray, fs: int, rpm: float, severity: float,
                         load: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(sig)) / fs
    f_rot = rpm / 60.0
    gain = 1.0 + 0.006 * load

    amps = {
        2: severity * gain * rng.uniform(1.25, 1.85) / (2 ** 0.40),
        3: severity * gain * rng.uniform(0.85, 1.40) / (3 ** 0.45),
        4: severity * gain * rng.uniform(0.25, 0.55) / (4 ** 0.55),
    }

    for h, amp in amps.items():
        sig += amp * np.sin(2 * np.pi * h * f_rot * t + rng.uniform(0, 2 * np.pi))

    sig += 0.12 * severity * np.sin(2 * np.pi * 1.5 * f_rot * t + rng.uniform(0, 2 * np.pi))
    sig += 0.05 * severity * np.sin(2 * np.pi * 0.5 * f_rot * t + rng.uniform(0, 2 * np.pi))
    return sig


def _inject_bearing_wear(sig: np.ndarray, fs: int, rpm: float, severity: float,
                         temp: float, rng: np.random.Generator) -> np.ndarray:
    n = len(sig)
    t = np.arange(n) / fs
    f_rot = rpm / 60.0

    defect_mult = rng.uniform(3.8, 7.2)
    f_def = defect_mult * f_rot
    hf_carrier = rng.uniform(2200, min(4700, fs / 2 - 200))
    env = (0.35 + 0.65 * np.sin(2 * np.pi * f_def * t + rng.uniform(0, 2 * np.pi))) ** 2

    temp_gain = 0.90 + 0.0032 * max(temp - 45.0, 0.0)
    burst_amp = severity * temp_gain * rng.uniform(1.20, 1.90)
    sig += burst_amp * env * np.sin(2 * np.pi * hf_carrier * t + rng.uniform(0, 2 * np.pi))

    impulse_rate = 0.0012 + 0.0030 * severity
    impulse_mask = rng.random(n) < impulse_rate
    impulse_mag = rng.normal(0, burst_amp * 1.10, size=n)
    sig += impulse_mask * impulse_mag
    return sig


def _add_noise(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power_sig = np.mean(sig ** 2) + 1e-12
    power_noise = power_sig / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(power_noise), size=len(sig))

    impulse_mask = rng.random(size=len(sig)) < rng.uniform(0.0005, 0.0015)
    impulse = impulse_mask * rng.normal(
        0, np.sqrt(power_sig) * rng.uniform(1.4, 2.4), size=len(sig)
    )
    return sig + noise + impulse


def extract_features(sig: np.ndarray, fs: int) -> dict:
    n = len(sig)

    rms = np.sqrt(np.mean(sig ** 2))
    peak_to_peak = np.max(sig) - np.min(sig)
    kurt = float(kurtosis(sig, fisher=True))
    skewness = float(skew(sig))
    crest_factor = np.max(np.abs(sig)) / (rms + 1e-12)
    impulse_factor = np.max(np.abs(sig)) / (np.mean(np.abs(sig)) + 1e-12)
    mean_abs = np.mean(np.sqrt(np.abs(sig))) ** 2
    clearance_factor = np.max(np.abs(sig)) / (mean_abs + 1e-12)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(sig))
    mag_norm = mag / (np.sum(mag) + 1e-12)

    spectral_centroid = float(np.sum(freqs * mag_norm))
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag_norm)))
    cum_energy = np.cumsum(mag ** 2)
    total_energy = cum_energy[-1] + 1e-12
    idx = min(np.searchsorted(cum_energy, 0.85 * total_energy), len(freqs) - 1)
    spectral_rolloff = float(freqs[idx])
    dominant_frequency = float(freqs[np.argmax(mag)])
    frequency_rms = float(np.sqrt(np.mean(mag ** 2)))

    sig_hist, _ = np.histogram(sig, bins=64, density=True)
    sig_hist = sig_hist[sig_hist > 0]
    ent = float(sp_entropy(sig_hist))

    band_mask = (freqs >= 1000) & (freqs <= 5000)
    band_energy = float(np.sum(mag[band_mask] ** 2))

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


def _sample_operating_conditions(fault_class: str, machine: str, rng: np.random.Generator):
    rpm_offset, snr_offset, amp_scale = MACHINE_PROFILES[machine]

    if fault_class == "normal":
        rpm = rng.uniform(800, 2500)
        load = rng.uniform(20, 80)
        temp = rng.uniform(40, 85)
        snr_db = rng.uniform(18, 30)
    elif "imbalance" in fault_class:
        rpm = rng.uniform(1100, 2900)
        load = rng.uniform(30, 88)
        temp = rng.uniform(45, 95)
        snr_db = rng.uniform(15, 27)
    elif "misalignment" in fault_class:
        rpm = rng.uniform(700, 2200)
        load = rng.uniform(55, 100)
        temp = rng.uniform(55, 115)
        snr_db = rng.uniform(14, 25)
    elif "bearing_wear" in fault_class:
        rpm = rng.uniform(650, 2100)
        load = rng.uniform(40, 95)
        temp = rng.uniform(65, 125)
        snr_db = rng.uniform(13, 24)
    else:  # combined_fault
        rpm = rng.uniform(900, 2500)
        load = rng.uniform(60, 100)
        temp = rng.uniform(75, 130)
        snr_db = rng.uniform(11, 22)

    rpm = max(250.0, rpm + rpm_offset + rng.normal(0, 25))
    snr_db = max(4.0, snr_db + snr_offset + rng.normal(0, 0.8))
    return rpm, load, temp, snr_db, amp_scale


def generate_single_sample(
    fault_class: str,
    n: int,
    fs: int,
    rpm: float,
    snr_db: float,
    amp_scale: float,
    load: float,
    temp: float,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sig = _base_signal(n, fs, rpm, amp_scale, load, temp, rng)

    if fault_class != "normal":
        # keep some hard samples, but not so many that early classes collapse together
        difficulty = rng.beta(1.8, 3.8)
        effective_snr = max(5.0, snr_db - 1.2 * difficulty)

        if "imbalance" in fault_class:
            sig = _inject_imbalance(sig, fs, rpm, severity, load, rng)
        elif "misalignment" in fault_class:
            sig = _inject_misalignment(sig, fs, rpm, severity, load, rng)
        elif "bearing_wear" in fault_class:
            sig = _inject_bearing_wear(sig, fs, rpm, severity, temp, rng)
        elif fault_class == "combined_fault":
            sig = _inject_imbalance(sig, fs, rpm, severity * rng.uniform(0.50, 0.85), load, rng)
            sig = _inject_misalignment(sig, fs, rpm, severity * rng.uniform(0.55, 0.90), load, rng)
            sig = _inject_bearing_wear(sig, fs, rpm, severity * rng.uniform(0.45, 0.80), temp, rng)
    else:
        effective_snr = snr_db

    sig = _add_noise(sig, effective_snr, rng)
    return sig


def _smooth_step(prev_val: float, target_val: float, drift_std: float, rng: np.random.Generator):
    return 0.82 * prev_val + 0.18 * target_val + rng.normal(0, drift_std)


def generate_dataset(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    base_samples = int(data_cfg["num_samples"])
    n_samples = _effective_sample_count(base_samples, data_cfg.get("size_multiplier", 1.5))
    n = data_cfg["sample_length"]
    fs = data_cfg["sampling_rate"]
    machines = data_cfg["machines"]
    seed = data_cfg["random_seed"]
    min_ep, max_ep = data_cfg.get("episode_len_range", [14, 32])

    rng = np.random.default_rng(seed)
    class_counts = _exact_class_counts(n_samples)
    episode_plan = _build_episode_plan(class_counts, min_ep, max_ep, rng)

    rows = []

    machine_time = {
        m: datetime(2025, 1, 1, 0, 0, 0) + timedelta(minutes=int(12 * i))
        for i, m in enumerate(machines)
    }

    machine_state = {
        m: {
            "rpm": None,
            "load": None,
            "temp": None,
            "snr_db": None,
        }
        for m in machines
    }

    machine_episode_counts = {m: 0 for m in machines}

    generated = 0
    print(f"Generating {n_samples} samples ...")

    for fault_class, ep_len in episode_plan:
        machine = min(machine_episode_counts, key=machine_episode_counts.get)
        machine_episode_counts[machine] += 1

        target_rpm, target_load, target_temp, target_snr_db, amp_scale = _sample_operating_conditions(
            fault_class, machine, rng
        )

        sev_curve = _episode_severity_curve(fault_class, ep_len, rng)

        state = machine_state[machine]
        if state["rpm"] is None:
            state["rpm"] = target_rpm
            state["load"] = target_load
            state["temp"] = target_temp
            state["snr_db"] = target_snr_db

        for j in range(ep_len):
            if generated >= n_samples:
                break

            state["rpm"] = max(250.0, _smooth_step(state["rpm"], target_rpm, 14.0, rng))
            state["load"] = np.clip(_smooth_step(state["load"], target_load, 2.0, rng), 15, 100)
            state["temp"] = np.clip(_smooth_step(state["temp"], target_temp, 1.8, rng), 35, 140)
            state["snr_db"] = max(4.0, _smooth_step(state["snr_db"], target_snr_db, 0.45, rng))

            severity = float(sev_curve[j])
            sample_snr = max(4.0, state["snr_db"] - 0.9 * severity + rng.normal(0, 0.30))

            machine_time[machine] += timedelta(minutes=int(rng.integers(1, 5)))

            sig = generate_single_sample(
                fault_class=fault_class,
                n=n,
                fs=fs,
                rpm=state["rpm"],
                snr_db=sample_snr,
                amp_scale=amp_scale,
                load=state["load"],
                temp=state["temp"],
                severity=severity,
                rng=rng,
            )

            feats = extract_features(sig, fs)

            row = {
                "machine_id": machine,
                "machine_id_encoded": MACHINE_ID_MAP[machine],
                "timestamp": machine_time[machine].strftime("%Y-%m-%d %H:%M:%S"),
                "operating_speed_rpm": round(state["rpm"], 1),
                "load_percentage": round(state["load"], 1),
                "temperature_celsius": round(state["temp"], 1),
                **feats,
                "fault_class": fault_class,
                "fault_present": 0 if fault_class == "normal" else 1,
            }
            rows.append(row)
            generated += 1

            if generated % 2000 == 0:
                print(f"  [{generated}/{n_samples}] generated")

        if generated >= n_samples:
            break

    df = pd.DataFrame(rows)

    # exact balancing safety pass
    target_counts = _exact_class_counts(n_samples)
    parts = []
    for cls in FAULT_CLASSES:
        cls_df = df[df["fault_class"] == cls]
        if len(cls_df) >= target_counts[cls]:
            parts.append(cls_df.iloc[:target_counts[cls]])
        else:
            needed = target_counts[cls] - len(cls_df)
            pad = cls_df.sample(n=needed, replace=True, random_state=seed) if len(cls_df) > 0 else pd.DataFrame()
            parts.append(pd.concat([cls_df, pad], ignore_index=True))

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    return df


def main():
    cfg = load_config()
    df = generate_dataset(cfg)

    out_path = cfg["data"]["output_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")
    print("\nClass distribution:")
    print(df["fault_class"].value_counts().sort_index().to_string())
    print(f"\nFault-present samples: {df['fault_present'].sum()}")
    print(f"Normal samples: {(df['fault_class'] == 'normal').sum()}")
    print(f"\nMachine distribution:\n{df['machine_id'].value_counts().to_string()}")
    print(f"\nColumns:\n{list(df.columns)}")


if __name__ == "__main__":
    main()