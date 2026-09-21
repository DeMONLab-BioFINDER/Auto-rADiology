# src/model_factory.py
import json
import inspect
import torch

from src.models import *


def _resolve_model_class(name: str):
    """Return a model class by name from models.py; raise a helpful error if missing."""
    try:
        return globals()[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{name}'. Ensure it's defined in models.py "
            f"and the name matches exactly (case-sensitive)."
        ) from e


def build_model_from_args(args, device=None, n_classes: int | None = None):
    """
    Dynamically instantiate a model by name from models.py using args.model.
    - Filters kwargs to what the class __init__ accepts.
    - Merges defaults from args with optional --model_kwargs (JSON or dict).
    - If args.resume is set, loads weights with strict=False.
    """
    ModelCls = _resolve_model_class(args.model)

    # Defaults from args (common across your models)
    defaults = {"in_channels": getattr(args, "in_channels", 1),
                "widths": tuple(getattr(args, "widths", (32, 64, 128, 256))),
                "dropout": getattr(args, "dropout", 0.3)}

    extra_dim = 0
    # scalar CL input
    if getattr(args, "input_cl", None) is not None:
        extra_dim += 1
        print('add extra input CL to the last FC layer')
    # global image-derived features
    if getattr(args, "extra_global_feats", None):
        # expect comma-separated string: "p95,std,frac_hi"
        feats = [f for f in args.extra_global_feats.split(",") if f.strip()]
        extra_dim += len(feats)
        print(f'add extra global input {args.extra_global_feats} to the last FC layer')
    defaults["extra_dim"] = extra_dim

    # JSON-only model kwargs
    extra = {}
    if hasattr(args, "model_kwargs") and args.model_kwargs:
        try:
            extra = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            raise ValueError(f"--model_kwargs must be valid JSON: {e}")
        if "widths" in extra and isinstance(extra["widths"], list):
            extra["widths"] = tuple(extra["widths"])

    # keep only params accepted by __init__
    allowed = set(inspect.signature(ModelCls.__init__).parameters) - {"self", "*args", "**kwargs"}
    params = {k: v for k, v in {**defaults, **extra}.items() if k in allowed}

    # auto-wire class count if provided by caller
    if n_classes is not None:
        if "num_classes" in allowed and "num_classes" not in params:
            params["num_classes"] = n_classes
        elif "out_channels" in allowed and "out_channels" not in params:
            params["out_channels"] = n_classes

    # Instantiate:  Build + move
    model = ModelCls(**params)
    if device is not None:
        model = model.to(device)

    # resume weights (Optional)
    if getattr(args, "resume", ""):
        state = torch.load(args.resume, map_location=device or "cpu")
        model.load_state_dict(state, strict=False)

    print(model)
    return model
