# train_hierarchical_cnn_lstm.py

import os
import random
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
@dataclass
class CFG:
    csv_path: str = "synthetic_vibration_data.csv"
    model_dir: str = "artifacts_hierarchical"
    seq_len: int = 24
    train_stride: int = 1
    eval_stride: int = 1
    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 96
    conv_dim1: int = 64
    conv_dim2: int = 96
    dropout: float = 0.20
    num_workers: int = 0
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lambda_multi: float = 1.0
    early_stop_patience: int = 8
    label_smoothing: float = 0.05


cfg = CFG()

os.makedirs(cfg.model_dir, exist_ok=True)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# =========================
# Labels / features
# =========================
from src.models import (
    CNNLSTMHierarchical, 
    FAULT_LABELS, 
    ALL_LABELS, 
    FAULT_TO_IDX, 
    FULL_TO_IDX,
    FEATURE_COLS
)


# =========================
# Data loading / splitting
# =========================
def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    required = FEATURE_COLS + ["machine_id", "timestamp", "fault_present", "fault_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not set(df["fault_class"].unique()).issubset(set(ALL_LABELS)):
        raise ValueError("fault_class contains unexpected labels.")

    return df


def split_by_machine_time(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    train_parts, val_parts, test_parts = [], [], []

    for machine_id, g in df.groupby("machine_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if n_train < cfg.seq_len + 2 or n_val < cfg.seq_len + 2:
            raise ValueError(f"Not enough rows for machine {machine_id} after split.")

        train_parts.append(g.iloc[:n_train].copy())
        val_parts.append(g.iloc[n_train:n_train + n_val].copy())
        test_parts.append(g.iloc[n_train + n_val:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df


def create_windows(df: pd.DataFrame, seq_len: int, stride: int):
    X, y_bin, y_multi, y_full = [], [], [], []

    for machine_id, g in df.groupby("machine_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feat = g[FEATURE_COLS].values.astype(np.float32)
        yb = g["fault_present"].values.astype(np.int64)
        yc = g["fault_class"].values

        for end in range(seq_len - 1, len(g), stride):
            start = end - seq_len + 1
            x_win = feat[start:end + 1]
            yb_last = int(yb[end])
            yc_last = yc[end]

            X.append(x_win)
            y_bin.append(yb_last)

            if yb_last == 1:
                y_multi.append(FAULT_TO_IDX[yc_last])
            else:
                y_multi.append(-1)

            y_full.append(FULL_TO_IDX[yc_last])

    X = np.stack(X)
    y_bin = np.array(y_bin, dtype=np.int64)
    y_multi = np.array(y_multi, dtype=np.int64)
    y_full = np.array(y_full, dtype=np.int64)
    return X, y_bin, y_multi, y_full


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    B, T, F = X_train.shape
    scaler.fit(X_train.reshape(B * T, F))
    return scaler


def transform_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    B, T, F = X.shape
    X2 = scaler.transform(X.reshape(B * T, F)).reshape(B, T, F)
    return X2.astype(np.float32)


# =========================
# Dataset
# =========================
class SeqDataset(Dataset):
    def __init__(self, X, y_bin, y_multi, y_full):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_bin = torch.tensor(y_bin, dtype=torch.float32)
        self.y_multi = torch.tensor(y_multi, dtype=torch.long)
        self.y_full = torch.tensor(y_full, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_bin[idx], self.y_multi[idx], self.y_full[idx]


# =========================
# Noise augmentation
# =========================
def augment_batch(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return x

    noise = 0.02 * torch.randn_like(x)
    x = x + noise

    time_mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < 0.04).float()
    feat_mask = (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) < 0.03).float()
    point_mask = (torch.rand_like(x) < 0.01).float()

    x = x * (1.0 - time_mask)
    x = x * (1.0 - feat_mask)
    x = x * (1.0 - point_mask)
    return x


# =========================
# Model
# =========================
# Model class moved to src.models.py


# =========================
# Metrics / eval
# =========================
@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    all_bin_logits = []
    all_multi_logits = []
    all_y_bin = []
    all_y_multi = []
    all_y_full = []

    for x, yb, ym, yf in loader:
        x = x.to(device)
        bin_logit, multi_logit = model(x)

        all_bin_logits.append(bin_logit.cpu().numpy())
        all_multi_logits.append(multi_logit.cpu().numpy())
        all_y_bin.append(yb.numpy())
        all_y_multi.append(ym.numpy())
        all_y_full.append(yf.numpy())

    return (
        np.concatenate(all_bin_logits),
        np.concatenate(all_multi_logits),
        np.concatenate(all_y_bin).astype(int),
        np.concatenate(all_y_multi).astype(int),
        np.concatenate(all_y_full).astype(int),
    )


def hierarchical_predict(bin_probs, multi_logits, threshold):
    bin_pred = (bin_probs >= threshold).astype(int)
    multi_pred = multi_logits.argmax(axis=1)

    full_pred = np.zeros(len(bin_pred), dtype=int)
    for i in range(len(bin_pred)):
        if bin_pred[i] == 0:
            full_pred[i] = FULL_TO_IDX["normal"]
        else:
            full_pred[i] = FULL_TO_IDX[FAULT_LABELS[multi_pred[i]]]
    return bin_pred, multi_pred, full_pred


def tune_threshold(y_true_bin, bin_probs):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.20, 0.80, 61):
        pred = (bin_probs >= t).astype(int)
        f1 = f1_score(y_true_bin, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def evaluate_model(model, loader, device, threshold=None):
    bin_logits, multi_logits, y_bin, y_multi, y_full = collect_outputs(model, loader, device)
    bin_probs = 1.0 / (1.0 + np.exp(-bin_logits))

    if threshold is None:
        threshold, _ = tune_threshold(y_bin, bin_probs)

    bin_pred, multi_pred, full_pred = hierarchical_predict(bin_probs, multi_logits, threshold)

    binary_f1 = f1_score(y_bin, bin_pred, zero_division=0)
    full_macro_f1 = f1_score(y_full, full_pred, average="macro", zero_division=0)

    fault_mask = (y_bin == 1)
    if fault_mask.sum() > 0:
        fault_true_names = [ALL_LABELS[i] for i in y_full[fault_mask]]
        fault_pred_names = [ALL_LABELS[i] for i in full_pred[fault_mask]]
        fault_macro_f1 = f1_score(
            fault_true_names,
            fault_pred_names,
            average="macro",
            labels=FAULT_LABELS,
            zero_division=0,
        )
    else:
        fault_macro_f1 = 0.0

    return {
        "threshold": threshold,
        "binary_f1": float(binary_f1),
        "fault_macro_f1": float(fault_macro_f1),
        "full_macro_f1": float(full_macro_f1),
        "bin_probs": bin_probs,
        "multi_logits": multi_logits,
        "y_bin": y_bin,
        "y_multi": y_multi,
        "y_full": y_full,
        "bin_pred": bin_pred,
        "multi_pred": multi_pred,
        "full_pred": full_pred,
    }


# =========================
# Training
# =========================
def main():
    df = load_dataframe(cfg.csv_path)
    train_df, val_df, test_df = split_by_machine_time(df, cfg.train_frac, cfg.val_frac)

    X_train, yb_train, ym_train, yf_train = create_windows(train_df, cfg.seq_len, cfg.train_stride)
    X_val, yb_val, ym_val, yf_val = create_windows(val_df, cfg.seq_len, cfg.eval_stride)
    X_test, yb_test, ym_test, yf_test = create_windows(test_df, cfg.seq_len, cfg.eval_stride)

    scaler = fit_scaler(X_train)
    X_train = transform_windows(X_train, scaler)
    X_val = transform_windows(X_val, scaler)
    X_test = transform_windows(X_test, scaler)

    train_ds = SeqDataset(X_train, yb_train, ym_train, yf_train)
    val_ds = SeqDataset(X_val, yb_val, ym_val, yf_val)
    test_ds = SeqDataset(X_test, yb_test, ym_test, yf_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    device = cfg.device
    model = CNNLSTMHierarchical(
        input_dim=len(FEATURE_COLS),
        hidden_dim=cfg.hidden_dim,
        conv_dim1=cfg.conv_dim1,
        conv_dim2=cfg.conv_dim2,
        dropout=cfg.dropout,
    ).to(device)

    pos = max(yb_train.sum(), 1)
    neg = max(len(yb_train) - yb_train.sum(), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    fault_train = ym_train[ym_train >= 0]
    class_counts = np.bincount(fault_train, minlength=len(FAULT_LABELS)).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bin_loss = 0.0
        total_multi_loss = 0.0

        for x, yb, ym, _ in train_loader:
            x = x.to(device)
            yb = yb.to(device)
            ym = ym.to(device)

            x = augment_batch(x)

            optimizer.zero_grad()
            bin_logit, multi_logit = model(x)

            loss_bin = bce(bin_logit, yb)

            mask = (yb > 0.5)
            if mask.any():
                loss_multi = ce(multi_logit[mask], ym[mask])
            else:
                loss_multi = torch.tensor(0.0, device=device)

            loss = loss_bin + cfg.lambda_multi * loss_multi
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            total_bin_loss += loss_bin.item()
            total_multi_loss += float(loss_multi.item())

        val_metrics = evaluate_model(model, val_loader, device, threshold=None)
        composite = 0.50 * val_metrics["binary_f1"] + 0.50 * val_metrics["fault_macro_f1"]
        scheduler.step(composite)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={total_loss/len(train_loader):.4f} | "
            f"bin_loss={total_bin_loss/len(train_loader):.4f} | "
            f"multi_loss={total_multi_loss/len(train_loader):.4f} | "
            f"val_bin_f1={val_metrics['binary_f1']:.4f} | "
            f"val_fault_macro_f1={val_metrics['fault_macro_f1']:.4f} | "
            f"val_full_macro_f1={val_metrics['full_macro_f1']:.4f} | "
            f"thr={val_metrics['threshold']:.3f}"
        )

        if composite > best_score:
            best_score = composite
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_threshold = val_metrics["threshold"]
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    val_metrics = evaluate_model(model, val_loader, device, threshold=best_threshold)
    test_metrics = evaluate_model(model, test_loader, device, threshold=best_threshold)

    print("\nBest threshold:", best_threshold)
    print("Validation binary F1:", round(val_metrics["binary_f1"], 4))
    print("Validation fault macro F1:", round(val_metrics["fault_macro_f1"], 4))
    print("Validation full macro F1:", round(val_metrics["full_macro_f1"], 4))

    print("\nTest binary F1:", round(test_metrics["binary_f1"], 4))
    print("Test fault macro F1:", round(test_metrics["fault_macro_f1"], 4))
    print("Test full macro F1:", round(test_metrics["full_macro_f1"], 4))

    y_bin_true = test_metrics["y_bin"]
    y_bin_pred = test_metrics["bin_pred"]

    print("\nBinary report:")
    print(classification_report(y_bin_true, y_bin_pred, digits=4, zero_division=0))

    y_full_true_names = [ALL_LABELS[i] for i in test_metrics["y_full"]]
    y_full_pred_names = [ALL_LABELS[i] for i in test_metrics["full_pred"]]

    print("\nHierarchical 8-class report:")
    print(classification_report(
        y_full_true_names,
        y_full_pred_names,
        labels=ALL_LABELS,
        digits=4,
        zero_division=0
    ))

    fault_mask = test_metrics["y_bin"] == 1
    if fault_mask.sum() > 0:
        fault_true_names = [ALL_LABELS[i] for i in test_metrics["y_full"][fault_mask]]
        fault_pred_names = [ALL_LABELS[i] for i in test_metrics["full_pred"][fault_mask]]

        print("\nFault-only multiclass report:")
        print(classification_report(
            fault_true_names,
            fault_pred_names,
            labels=FAULT_LABELS,
            digits=4,
            zero_division=0
        ))

        cm_fault = confusion_matrix(fault_true_names, fault_pred_names, labels=FAULT_LABELS)
        print("\nFault-only confusion matrix labels:")
        print(FAULT_LABELS)
        print(cm_fault)

    cm_full = confusion_matrix(y_full_true_names, y_full_pred_names, labels=ALL_LABELS)
    print("\nFull confusion matrix labels:")
    print(ALL_LABELS)
    print(cm_full)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": FEATURE_COLS,
            "fault_labels": FAULT_LABELS,
            "all_labels": ALL_LABELS,
            "threshold": best_threshold,
            "seq_len": cfg.seq_len,
            "config": cfg.__dict__,
        },
        os.path.join(cfg.model_dir, "hierarchical_cnn_lstm.pt"),
    )

    joblib.dump(scaler, os.path.join(cfg.model_dir, "scaler.joblib"))

    with open(os.path.join(cfg.model_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "val_binary_f1": val_metrics["binary_f1"],
                "val_fault_macro_f1": val_metrics["fault_macro_f1"],
                "val_full_macro_f1": val_metrics["full_macro_f1"],
                "test_binary_f1": test_metrics["binary_f1"],
                "test_fault_macro_f1": test_metrics["fault_macro_f1"],
                "test_full_macro_f1": test_metrics["full_macro_f1"],
                "threshold": best_threshold,
                "seq_len": cfg.seq_len,
            },
            f,
            indent=2,
        )

    print(f"\nSaved model and scaler to: {cfg.model_dir}")


if __name__ == "__main__":
    main()