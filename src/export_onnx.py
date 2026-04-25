"""
ONNX Export
===========
Loads the best PyTorch checkpoint, exports the model to ONNX format
with dynamic batch size for flexible deployment.
"""

import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def export_to_onnx(cfg: dict):
    mcfg = cfg["model"]
    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
    onnx_path = cfg["export"]["onnx_path"]
    opset = cfg["export"]["opset_version"]

    # load model
    model = build_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # dummy input
    dummy = torch.randn(1, 16)

    # export
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["features"],
        output_names=["class_logits", "early_fault_logit"],
        dynamic_axes={
            "features": {0: "batch"},
            "class_logits": {0: "batch"},
            "early_fault_logit": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {onnx_path}")

    # verify
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed [OK]")

    # quick inference test
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 16).astype(np.float32)
    outputs = session.run(None, {"features": test_input})
    print(f"Test inference — class logits shape: {outputs[0].shape}, early-fault shape: {outputs[1].shape}")


if __name__ == "__main__":
    cfg = load_config()
    export_to_onnx(cfg)
