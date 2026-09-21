# src/checkpoints.py
import os
import torch


def save_checkpoint(model, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)


def load_best_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    Load model weights from checkpoint path into the given model.
    """
    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(ckpt_path, map_location=device)
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[warn] checkpoint not found: {ckpt_path}")
    return model
