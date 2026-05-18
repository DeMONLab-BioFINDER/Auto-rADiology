# src/hyperparam_spaces.py
import json
from typing import Dict, Tuple, Optional


def suggest_common(trial, base_args) -> Dict:
    """
    Common hyperparams plus preprocessing/loss choices.
    User-overridable through optional args:
        tune_lr, tune_weight_decay, tune_batch_size, tune_dropout
        tune_spacing, tune_norm, tune_smoothing
        tune_cls_loss, tune_reg_loss, tune_loss_w_dataset, tune_cls_threshold
    """
    lr_choices = _float_list(base_args, "tune_lr", [3e-4, 1e-4])
    wd_choices = _float_list(base_args, "tune_weight_decay", [1e-5, 1e-4])
    bs_choices = _int_list(base_args, "tune_batch_size", [2, 4, 8])
    dropout_choices = _float_list(base_args, "tune_dropout", [0.0, 0.2, 0.3])

    spacing_choices = _bool_list(base_args, "tune_spacing", [True, False])
    norm_choices = _csv_list(base_args, "tune_norm", ["percentile_01", "zscore"])

    lds_choices = _float_list(base_args, "tune_loss_w_dataset", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]) #!!! fail if UNet3D

    common = {
        #"lr": trial.suggest_categorical("lr", lr_choices),
        #"weight_decay": trial.suggest_categorical("weight_decay", wd_choices),
        #"batch_size": trial.suggest_categorical("batch_size", bs_choices),
        #"dropout": trial.suggest_categorical("dropout", dropout_choices),

        # preprocessing
        "spacing": trial.suggest_categorical("spacing", spacing_choices),
        "intensity_norm": trial.suggest_categorical("intensity_norm", norm_choices),

        "loss_w_dataset": trial.suggest_categorical("loss_w_dataset", lds_choices)
    }

    if _is_vr_task(base_args):
        cls_loss_choices = _csv_list(base_args, "tune_cls_loss", ["bce", "weighted_bce"])
        cls_thr_choices = _float_list(base_args, "tune_cls_threshold", [0.3, 0.4, 0.5, 0.6, 0.7])
        
        common["cls_loss"] = trial.suggest_categorical("cls_loss", cls_loss_choices)
        common["cls_threshold"] = trial.suggest_categorical("cls_threshold", cls_thr_choices)

    if _is_cl_task(base_args):
        reg_loss_choices = _csv_list(base_args, "tune_reg_loss", ["mse", "huber", "weighted_mse", "weighted_huber"])
        common["reg_loss"] = trial.suggest_categorical("reg_loss", reg_loss_choices)

    return common


def suggest_CNN3D(trial, base_args, common: Dict) -> Tuple[str, str]:
    width_opts = _json_tuple_choices(base_args, "tune_widths", [(16, 32, 64, 128), (32, 64, 128, 256)])
    key = trial.suggest_categorical("cnn_widths", [json.dumps(o) for o in width_opts])
    widths = tuple(json.loads(key))

    block_choices = _csv_list(base_args, "tune_block", ["conv", "res"])
    downsample_choices = _csv_list(base_args, "tune_downsample", ["pool", "stride"])
    norm_choices = _csv_list(base_args, "tune_cnn_norm", ["batch", "instance"])

    block = trial.suggest_categorical("block", block_choices)
    downsample = trial.suggest_categorical("downsample", downsample_choices)
    norm = trial.suggest_categorical("cnn_norm", norm_choices)

    kwargs = {
        "in_channels": getattr(base_args, "in_channels", 1),
        "widths": list(widths),
        "pool_every": 1,
        "norm": norm,
        #"dropout": common["dropout"],
        "block": block,
        "downsample": downsample,
    }

    return "CNN3D", json.dumps(kwargs)


def suggest_UNet3D(trial, base_args, common: Dict) -> Tuple[str, str]:
    use_basic = trial.suggest_categorical("unet_use_basic", [True, False])
    opts = [(16, 16, 32, 64, 128, 32), (16, 16, 32, 64, 128, 16), (32, 32, 64, 96, 128, 32), (32, 32, 64, 128, 256, 32)]
    key = trial.suggest_categorical("unet_channels", [json.dumps(o) for o in opts])
    channels = tuple(json.loads(key))
    num_res = trial.suggest_categorical("unet_num_res_units", [1, 2])

    kwargs = {
        "in_channels": getattr(base_args, "in_channels", 1),
        "out_channels": 1,
        "channels": list(channels),
        "strides": [2, 2, 2, 2],
        "num_res_units": num_res,
        "norm": "instance",
        #"dropout": min(common["dropout"], 0.3),
        "use_basic": use_basic,
    }
    return "UNet3D", json.dumps(kwargs)


SUGGESTORS = {"CNN3D": suggest_CNN3D, "UNet3D": suggest_UNet3D}

def suggest_model(trial, base_args, common: Dict, model_name: Optional[str] = None) -> Tuple[str, str]:
    name = (model_name or getattr(base_args, "model_name", None) or getattr(base_args, "model", None) or getattr(base_args, "arch", None))
    if name is None:
        raise ValueError("Please specify model_name/base_args.model/base_args.arch.")
    if name not in SUGGESTORS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(SUGGESTORS.keys())}")

    trial.set_user_attr("arch", name)
    return SUGGESTORS[name](trial, base_args, common)


def _csv_list(base_args, name, default):
    v = getattr(base_args, name, None)
    if v is None or str(v).strip() == "":
        return default
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _bool_list(base_args, name, default):
    vals = _csv_list(base_args, name, None)
    if vals is None:
        return default
    out = []
    for v in vals:
        if str(v).lower() in {"true", "1", "yes", "y"}:
            out.append(True)
        elif str(v).lower() in {"false", "0", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse bool value '{v}' for {name}.")
    return out


def _float_list(base_args, name, default):
    vals = _csv_list(base_args, name, None)
    return default if vals is None else [float(v) for v in vals]


def _int_list(base_args, name, default):
    vals = _csv_list(base_args, name, None)
    return default if vals is None else [int(v) for v in vals]


def _json_tuple_choices(base_args, name, default):
    """
    Example CLI string:
        --tune_widths "[16,32,64,128];[32,64,128,256]"
    """
    v = getattr(base_args, name, None)
    if v is None or str(v).strip() == "":
        return default
    out = []
    for item in str(v).split(";"):
        item = item.strip()
        if item:
            out.append(tuple(json.loads(item)))
    return out


def _is_vr_task(base_args):
    targets = getattr(base_args, "targets", "")
    return "visual_read" in [t.strip() for t in targets.split(",") if t.strip()]


def _is_cl_task(base_args):
    targets = getattr(base_args, "targets", "")
    return "CL" in [t.strip() for t in targets.split(",") if t.strip()]