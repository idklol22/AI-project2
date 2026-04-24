"""
CNN-LSTM Hybrid Model for Vibration-Based Fault Detection
==========================================================
Architecture:
  - 1D CNN feature extractor (3 conv blocks) operating on the 15 engineered features
    reshaped as a 1-channel 1D sequence.
  - Bidirectional LSTM for temporal-pattern modelling.
  - Dual-head output:
      • Classification head → 8 fault classes (softmax)
      • Early-fault head → binary (sigmoid)

The model accepts a tensor of shape (batch, 15) — the 15 engineered vibration
features — and produces class logits + early-fault probability.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → Dropout"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class FaultDetectorCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for fault classification + early-fault detection.

    Input:  (batch, num_features)   — e.g. (B, 15)
    Output: (class_logits (B, C), early_fault_prob (B, 1))
    """

    def __init__(
        self,
        num_features: int = 15,
        cnn_channels: list[int] = None,
        cnn_kernel_sizes: list[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 8,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [7, 5, 3]

        self.num_features = num_features

        # ── CNN feature extractor ──
        cnn_layers = []
        in_ch = 1
        for out_ch, ks in zip(cnn_channels, cnn_kernel_sizes):
            cnn_layers.append(ConvBlock(in_ch, out_ch, ks, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # ── BiLSTM ──
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ── classification head ──
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # ── early-fault head ──
        self.early_fault_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, num_features)
        """
        # reshape to (batch, 1, num_features) for Conv1d
        x = x.unsqueeze(1)  # (B, 1, F)

        # CNN
        x = self.cnn(x)  # (B, C_last, F)

        # transpose for LSTM: (B, seq_len, features) where seq_len = F
        x = x.permute(0, 2, 1)  # (B, F, C_last)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, F, lstm_out_dim)

        # take the last time-step
        h = lstm_out[:, -1, :]  # (B, lstm_out_dim)

        class_logits = self.classifier(h)          # (B, num_classes)
        early_fault = self.early_fault_head(h)     # (B, 1)

        return class_logits, early_fault


def build_model(cfg: dict) -> FaultDetectorCNNLSTM:
    """Build model from config dict."""
    mcfg = cfg["model"]
    return FaultDetectorCNNLSTM(
        num_features=15,
        cnn_channels=mcfg["cnn_channels"],
        cnn_kernel_sizes=mcfg["cnn_kernel_sizes"],
        lstm_hidden=mcfg["lstm_hidden"],
        lstm_layers=mcfg["lstm_layers"],
        bidirectional=mcfg["bidirectional"],
        dropout=mcfg["dropout"],
        num_classes=mcfg["num_classes"],
    )


if __name__ == "__main__":
    # quick sanity check
    model = FaultDetectorCNNLSTM()
    dummy = torch.randn(4, 15)
    cls_out, ef_out = model(dummy)
    print(f"Class logits shape : {cls_out.shape}")   # (4, 8)
    print(f"Early-fault shape  : {ef_out.shape}")     # (4, 1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters   : {total_params:,}")
