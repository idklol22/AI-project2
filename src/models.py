import torch
import torch.nn as nn
import numpy as np

# Labels
FAULT_LABELS = [
    "imbalance_early",
    "imbalance_severe",
    "misalignment_early",
    "misalignment_severe",
    "bearing_wear_early",
    "bearing_wear_severe",
    "combined_fault",
]

ALL_LABELS = ["normal"] + FAULT_LABELS
FAULT_TO_IDX = {k: i for i, k in enumerate(FAULT_LABELS)}
FULL_TO_IDX = {k: i for i, k in enumerate(ALL_LABELS)}

FEATURE_COLS = [
    "machine_id_encoded",
    "operating_speed_rpm",
    "load_percentage",
    "temperature_celsius",
    "rms",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "crest_factor",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "dominant_frequency",
    "frequency_rms",
    "entropy",
    "impulse_factor",
    "clearance_factor",
    "band_energy_1_5kHz",
    "snr_estimated",
]

class CNNLSTMHierarchical(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, conv_dim1: int, conv_dim2: int, dropout: float):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim1, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim1),
            nn.GELU(),
            nn.Conv1d(conv_dim1, conv_dim2, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_dim2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        rep_dim = hidden_dim * 4

        self.binary_head = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.multi_head = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(FAULT_LABELS)),
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)           # [B, F, T]
        x = self.cnn(x)                 # [B, C, T]
        x = x.transpose(1, 2)           # [B, T, C]

        out, _ = self.lstm(x)           # [B, T, 2H]
        mean_pool = out.mean(dim=1)     # [B, 2H]
        last_step = out[:, -1, :]       # [B, 2H]
        rep = torch.cat([mean_pool, last_step], dim=1)

        bin_logit = self.binary_head(rep).squeeze(1)
        multi_logit = self.multi_head(rep)
        return bin_logit, multi_logit

def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    
    model = CNNLSTMHierarchical(
        input_dim=len(checkpoint["feature_cols"]),
        hidden_dim=cfg["hidden_dim"],
        conv_dim1=cfg["conv_dim1"],
        conv_dim2=cfg["conv_dim2"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
