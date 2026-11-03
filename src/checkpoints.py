# src/checkpoints.py
import os
import torch


def save_checkpoint(model, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)


def load_state_dict_checked(model: torch.nn.Module, state_dict: dict, source: str):
    """load_state_dict that fails on an architecture mismatch instead of silently skipping weights
    (strict=False alone would, e.g., drop an attention-pooling checkpoint's attention weights when
    loaded into an average-pooling model and evaluate a half-random model)."""
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {source} does not fit the model built from these arguments "
            f"(missing: {result.missing_keys[:5]}, unexpected: {result.unexpected_keys[:5]}). "
            "Pass the same --model_kwargs (e.g. pool) and --cls_loss the model was trained with.")
    return model


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
        load_state_dict_checked(model, state_dict, ckpt_path)
    else:
        print(f"[warn] checkpoint not found: {ckpt_path}")
    return model
