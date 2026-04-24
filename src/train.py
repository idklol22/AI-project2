"""
Training Pipeline
=================
Loads the synthetic CSV, trains the CNN-LSTM model with:
  - Weighted cross-entropy for multi-class fault classification
  - Binary cross-entropy for early-fault detection
  - Cosine-annealing LR schedule
  - Early stopping on validation early-fault recall
  - Saves best checkpoint
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, classification_report
import yaml
import json

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model


# ─── Dataset class ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rms", "peak_to_peak", "kurtosis", "skewness", "crest_factor",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "dominant_frequency", "frequency_rms", "entropy", "impulse_factor",
    "clearance_factor", "band_energy_1_5kHz", "snr_estimated",
]


class VibrationDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, early_flags: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.early_flags = torch.tensor(early_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.early_flags[idx]


# ─── helpers ────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(cfg: dict):
    """Load CSV, encode labels, scale features, split."""
    csv_path = cfg["data"]["output_csv"]
    df = pd.read_csv(csv_path)

    # encode fault classes
    le = LabelEncoder()
    y = le.fit_transform(df["fault_class"])
    early = df["is_early_fault"].values.astype(np.float32)

    # scale features
    X = df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train / val split
    val_split = cfg["training"]["val_split"]
    X_train, X_val, y_train, y_val, ef_train, ef_val = train_test_split(
        X, y, early, test_size=val_split, random_state=42, stratify=y,
    )

    return X_train, X_val, y_train, y_val, ef_train, ef_val, le, scaler


# ─── training ──────────────────────────────────────────────────────────────

def train(cfg: dict):
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data
    X_train, X_val, y_train, y_val, ef_train, ef_val, le, scaler = prepare_data(cfg)
    train_ds = VibrationDataset(X_train, y_train, ef_train)
    val_ds = VibrationDataset(X_val, y_val, ef_val)

    # weighted sampler for class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    # model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # loss
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    ef_criterion = nn.BCEWithLogitsLoss()

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                                 weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    # early stopping
    best_val_recall = 0.0
    patience_counter = 0
    ckpt_dir = tcfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── training loop ──
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for X_b, y_b, ef_b in train_loader:
            X_b, y_b, ef_b = X_b.to(device), y_b.to(device), ef_b.to(device)
            cls_logits, ef_logits = model(X_b)

            loss_cls = cls_criterion(cls_logits, y_b)
            loss_ef = ef_criterion(ef_logits.squeeze(-1), ef_b)
            loss = loss_cls + 0.5 * loss_ef

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ── validation ──
        model.eval()
        all_preds, all_labels, all_ef_preds, all_ef_labels = [], [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for X_b, y_b, ef_b in val_loader:
                X_b, y_b, ef_b = X_b.to(device), y_b.to(device), ef_b.to(device)
                cls_logits, ef_logits = model(X_b)

                loss_cls = cls_criterion(cls_logits, y_b)
                loss_ef = ef_criterion(ef_logits.squeeze(-1), ef_b)
                val_loss += (loss_cls + 0.5 * loss_ef).item()

                preds = cls_logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_b.cpu().numpy())

                ef_preds = (torch.sigmoid(ef_logits.squeeze(-1)) > 0.5).int().cpu().numpy()
                all_ef_preds.extend(ef_preds)
                all_ef_labels.extend(ef_b.cpu().numpy().astype(int))

        avg_val_loss = val_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        ef_recall = recall_score(all_ef_labels, all_ef_preds, zero_division=0)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} | "
            f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
            f"F1 {f1:.4f} | EF-Recall {ef_recall:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
        )

        # checkpoint
        if ef_recall > best_val_recall:
            best_val_recall = ef_recall
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": f1,
                "val_ef_recall": ef_recall,
            }, os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  ✓ Best checkpoint saved (EF-Recall={ef_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= tcfg["early_stop_patience"]:
                print(f"  ✗ Early stopping at epoch {epoch}")
                break

    # final report
    print("\n" + "=" * 60)
    print("Final Validation Report")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds, target_names=le.classes_, zero_division=0,
    ))

    # save label encoder mapping & scaler params
    meta = {
        "label_classes": le.classes_.tolist(),
        "feature_columns": FEATURE_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {ckpt_dir}/metadata.json")

    return model, le, scaler


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
