# src/train.py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error, root_mean_squared_error, roc_curve, balanced_accuracy_score, f1_score, matthews_corrcoef


def train_one_epoch(model, loader, opt, scaler, device, loss_w_cls, loss_w_reg, reg_loss="mse", smoothl1_beta=1.0, cls_loss="bce", pos_weight=None, loss_w_dataset=0.0):
    model.train()
    losses = []

    for x, y_cls, y_reg, dataset_target, sample_weights, extra, idx in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        extra = extra.to(device)
        idx = idx.to(x.device)
        y_cls = y_cls.to(device) if y_cls is not None else None
        y_reg = y_reg.to(device) if y_reg is not None else None
        sample_weights = sample_weights.to(device).float().view(-1)
        dataset_target = dataset_target.to(device).long().view(-1)
        
        opt.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, _ = compute_total_loss(model, x, y_cls, y_reg, extra=extra, loss_w_cls=loss_w_cls, loss_w_reg=loss_w_reg,
                                             reg_loss=reg_loss, smoothl1_beta=smoothl1_beta, sample_weights=sample_weights,
                                             cls_loss=cls_loss, pos_weight=pos_weight, dataset_target=dataset_target, loss_w_dataset=loss_w_dataset)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss, _ = compute_total_loss(model, x, y_cls, y_reg, extra=extra, loss_w_cls=loss_w_cls, loss_w_reg=loss_w_reg,
                                         reg_loss=reg_loss, smoothl1_beta=smoothl1_beta, sample_weights=sample_weights,
                                         cls_loss=cls_loss, pos_weight=pos_weight, dataset_target=dataset_target, loss_w_dataset=loss_w_dataset)
            loss.backward()
            opt.step()

        losses.append(float(loss.detach().item()))
    losses_dict = {i: v for i, v in enumerate(losses)}

    return (float(np.mean(losses)), losses_dict) if losses else (0.0, {})


@torch.no_grad()
def inference(model, loader, device, cls_threshold):
    probs, ycls, preds, any_cls, cents, yreg, any_reg, ids = evals(model, loader, device, cls_threshold=cls_threshold)

    metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
    # es_metric <- val_metric
    print('ids shape:', ids)
    df_result = pd.DataFrame()
    if any_cls: 
        df_result = pd.DataFrame({'ID_ind': np.concatenate(ids, axis=0), 
                                  'y': np.concatenate(ycls, axis=0), 
                                  'pred':np.concatenate(preds, axis=0),
                                  'prob':np.concatenate(probs, axis=0)})
    if any_reg:
        df_result = pd.DataFrame({'ID_ind': np.concatenate(ids, axis=0), 
                                  'y': np.concatenate(yreg, axis=0),
                                  'pred':np.concatenate(cents, axis=0)})

    return metrics, df_result


@torch.no_grad()
def evals(model, loader, device, cls_threshold):
    model.eval()

    probs, ycls, preds = [], [], []
    cents, yreg = [], []
    ids = []

    for x, y_cls, y_reg, _, _, extra, id in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        extra = extra.to(device)

        out = model(x, extra=extra)
        logit, cent, _, _ = unpack_model_outputs(out, y_reg)

        cls_out = decode_cls_outputs(logit, y_cls, threshold=cls_threshold)
        if cls_out is not None:
            p, pred, y = cls_out
            probs.append(p)
            preds.append(pred)
            ycls.append(y)

        reg_out = decode_reg_outputs(cent, y_reg)
        if reg_out is not None:
            pred_reg, y = reg_out
            cents.append(pred_reg)
            yreg.append(y)

        ids.append(id.detach().cpu().numpy().ravel())

    any_cls = len(ycls) > 0
    any_reg = len(yreg) > 0

    return probs, ycls, preds, any_cls, cents, yreg, any_reg, ids


def compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg):
    metrics = {"auc": np.nan, "acc": np.nan, "acc_opt": np.nan, "bacc": np.nan, "f1": np.nan, "mcc": np.nan, "best_thr": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "eval_metric": 0.0}

    if any_cls:
        metrics.update(compute_cls_metrics(ycls, preds, probs))

    if any_reg:
        metrics.update(compute_reg_metrics(yreg, cents))

    return metrics


def compute_total_loss(model: torch.nn.Module, x: torch.Tensor, y_cls: torch.Tensor, y_reg: torch.Tensor, extra: torch.Tensor,
                       loss_w_cls: float, loss_w_reg: float,
                       reg_loss: str = "mse", smoothl1_beta: Optional[float] = None, sample_weights: Optional[torch.Tensor] = None,
                       cls_loss: str = "bce", pos_weight: Optional[torch.Tensor] = None,
                       dataset_target: Optional[torch.Tensor] = None, loss_w_dataset: float = 0.0,): 
    """
    Forward + weighted loss.
    Returns: (loss, (logit, cent))
    """
    out = model(x, extra=extra)
    logit, cent, feats, dataset_logit = unpack_model_outputs(out, y_reg)

    total = torch.zeros((), device=x.device)
    loss_parts = {}
    used_head = False

    # ---- classification loss ----
    if loss_w_cls > 0:
        loss_cls = compute_cls_loss(logit=logit, y_cls=y_cls, cls_loss=cls_loss, sample_weights=sample_weights, pos_weight=pos_weight)
        if loss_cls is not None:
            total = total + loss_w_cls * loss_cls
            loss_parts["cls"] = float(loss_cls.detach().cpu())
            used_head = True
    # ---- regression loss ----
    if loss_w_reg > 0:
        loss_reg = compute_reg_loss(cent=cent, y_reg=y_reg, reg_loss=reg_loss, smoothl1_beta=smoothl1_beta, sample_weights=sample_weights)
        if loss_reg is not None:
            total = total + loss_w_reg * loss_reg
            loss_parts["reg"] = float(loss_reg.detach().cpu())
            used_head = True
    # ---- dataset loss ----
    if loss_w_dataset > 0:
        loss_dataset = compute_dataset_loss(dataset_logit=dataset_logit, dataset_target=dataset_target, sample_weights=None)
        if loss_dataset is None:
            raise ValueError("loss_w_dataset > 0 but dataset_logit or dataset_target is missing.")
        total = total + loss_w_dataset * loss_dataset
        loss_parts["dataset"] = float(loss_dataset.detach().cpu())
        used_head = True

    if not used_head: raise ValueError("No usable heads/targets: check model outputs and loss weights.")

    return total, loss_parts



def unpack_model_outputs(out: Any, y_reg) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns: logit, cent, feats, dataset_logit
    """
    # dict-like (preferred)
    if hasattr(out, "get"):
        logit = out.get("logit", out.get("cls", None))
        cent  = out.get("cent",  out.get("reg", None))
        feats = out.get("feats", None)
        dataset_logit = out.get("dataset_logit", out.get("domain", None))
        return logit, cent, feats, dataset_logit

    if isinstance(out, (tuple, list)):
        if len(out) == 4:
            return out[0], out[1], out[2], out[3]
        if len(out) == 3:
            return out[0], out[1], out[2], None
        if len(out) == 2:
            return out[0], out[1], None, None
        if len(out) == 1:
            t = out[0]
            return t, None, None, None  # treat as single-head (classification by default)

    # single tensor
    if isinstance(out, torch.Tensor):
        if y_reg is not None and not y_reg.isnan().all():
            return None, out, None, None
        return out, None, None, None

    # unknown shape
    return None, None, None, None


def decode_cls_outputs(logit, y_cls, threshold=0.5):
    if logit is None or y_cls is None or y_cls.isnan().all(): return None

    y = y_cls.detach().cpu().numpy().ravel()
    valid = np.isfinite(y)

    if not valid.any(): return None

    logit = logit.detach()

    if logit.ndim == 2 and logit.shape[1] == 1:
        prob = torch.sigmoid(logit).cpu().numpy().ravel()
        pred = (prob > threshold).astype(int)

    else:
        prob_all = F.softmax(logit, dim=1).cpu().numpy()

        if prob_all.shape[1] == 2:
            prob = prob_all[:, 1]
            pred = (prob > threshold).astype(int)
        else:
            prob = prob_all
            pred = prob_all.argmax(axis=1)

    return prob[valid] if prob.ndim == 1 else prob[valid, :], pred[valid], y[valid].astype(int)


def decode_reg_outputs(cent, y_reg):
    if cent is None or y_reg is None or y_reg.isnan().all():
        return None

    pred = cent.detach().cpu().numpy().ravel()
    y = y_reg.detach().cpu().numpy().ravel()
    valid = np.isfinite(pred) & np.isfinite(y)

    if not valid.any():
        return None

    return pred[valid], y[valid]


def compute_cls_metrics(ycls, preds, probs):
    y = np.concatenate(ycls).astype(int)
    pred = np.concatenate(preds).astype(int)
    prob = np.concatenate(probs, axis=0)

    if prob.ndim == 1:
        valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(prob)
    else:
        valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(prob).all(axis=1)

    y = y[valid]
    pred = pred[valid]
    prob = prob[valid] if prob.ndim == 1 else prob[valid, :]

    out = {"auc": np.nan, "acc": np.nan, "acc_opt": np.nan, "bacc": np.nan, "f1": np.nan, "mcc": np.nan, "best_thr": np.nan, "eval_metric": 0.0}

    if len(np.unique(y)) < 2:
        out["acc"] = float(accuracy_score(y, pred)) if len(y) > 0 else np.nan
        return out

    if prob.ndim == 1:
        out["auc"] = float(roc_auc_score(y, prob))
        out["acc_opt"], out["bacc"], out["f1"], out["mcc"], out["best_thr"] = opt_threshold(y, prob, pos=1)

    elif prob.shape[1] == 2:
        pos = int(np.max(y))
        out["auc"] = float(roc_auc_score((y == pos).astype(int), prob[:, pos]))
        out["acc_opt"], out["bacc"], out["f1"], out["mcc"], out["best_thr"] = opt_threshold(y, prob, pos)

    else:
        labels = np.unique(y).astype(int)
        out["auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=labels))

    out["acc"] = float(accuracy_score(y, pred))
    out["eval_metric"] = np.nansum([out["auc"], out["acc"]])

    return out


def compute_reg_metrics(yreg, cents):
    y = np.concatenate(yreg)
    pred = np.concatenate(cents)

    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]

    out = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "eval_metric": 0.0}

    if len(y) == 0:
        return out

    out["mae"] = float(mean_absolute_error(y, pred))
    out["rmse"] = float(root_mean_squared_error(y, pred))
    out["r2"] = float(r2_score(y, pred))

    mae_ref = np.median(np.abs(y - np.median(y)))
    if not np.isfinite(mae_ref) or mae_ref <= 1e-6:
        mae_ref = max(np.std(y), 1e-6)

    mae_good = 1.0 - np.clip(out["mae"] / mae_ref, 0.0, 1.0)
    out["eval_metric"] = np.nansum([mae_good, out["r2"]])

    return out


def weighted_mean(loss: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = loss.view(-1)
    if sample_weights is None:
        return loss.mean()
    sample_weights = sample_weights.view(-1).to(loss.device).float()
    return (loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def compute_cls_loss(logit: torch.Tensor, y_cls: torch.Tensor, cls_loss: str = "bce",
                     sample_weights: Optional[torch.Tensor] = None, pos_weight: Optional[torch.Tensor] = None,) -> torch.Tensor:
    
    if y_cls is None or y_cls.isnan().all() or logit is None: return None

    y = y_cls.view(-1)
    valid = torch.isfinite(y)
    if not valid.any(): return None

    if cls_loss in {"bce", "weighted_bce"}:
        if not (logit.ndim == 2 and logit.shape[1] == 1): raise ValueError("BCE requires logit shape [B, 1]. Use cls_loss='ce' for [B, C].")

        pred = logit.view(-1)[valid]
        target = y[valid].float()
        if not torch.all((target == 0) | (target == 1)): raise ValueError("BCE requires binary targets {0,1}.")

        pw = pos_weight.to(logit.device) if pos_weight is not None else None
        loss_vec = F.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=pw)
        w = sample_weights[valid] if (cls_loss == "weighted_bce" and sample_weights is not None) else None
        return weighted_mean(loss_vec, w)

    if cls_loss == "ce":
        target = y[valid].long()
        pred = logit[valid]
        if target.min() < 0 or target.max() >= pred.shape[1]: raise ValueError("CE target out of range.")
        loss_vec = F.cross_entropy(pred, target, reduction="none")
        return weighted_mean(loss_vec, None)

    raise ValueError(f"Unknown cls_loss='{cls_loss}'. Use 'bce', 'weighted_bce', or 'ce'.")


def compute_reg_loss(cent: torch.Tensor, y_reg: torch.Tensor, reg_loss: str = "mse", smoothl1_beta: float = 1.0,
                     sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if y_reg is None or y_reg.isnan().all() or cent is None: return None

    pred = cent.view(-1)
    target = y_reg.view(-1)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if not valid.any(): return None

    pred = pred[valid]
    target = target[valid]

    if reg_loss in {"mse", "weighted_mse"}:
        loss_vec = F.mse_loss(pred, target, reduction="none")
    elif reg_loss in {"huber", "smoothl1", "weighted_huber", "weighted_smoothl1"}:
        loss_vec = F.smooth_l1_loss(pred, target, beta=smoothl1_beta, reduction="none")
    else:
        raise ValueError(f"Unknown reg_loss='{reg_loss}'.")

    w = sample_weights[valid] if reg_loss.startswith("weighted") and sample_weights is not None else None
    return weighted_mean(loss_vec, w)


def compute_dataset_loss(dataset_logit: torch.Tensor, dataset_target: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if dataset_logit is None or dataset_target is None: return None

    target = dataset_target.view(-1).long().to(dataset_logit.device)
    valid = target >= 0
    if not valid.any(): return None

    loss_vec = F.cross_entropy(dataset_logit[valid], target[valid], reduction="none")
    w = sample_weights[valid] if sample_weights is not None else None

    return weighted_mean(loss_vec, w)

def opt_threshold(ycls, probs, pos):
    ybin = (ycls == pos).astype(int)             # binary ground-truth 0/1
    prob1 = probs if probs.ndim == 1 else probs[:, pos]
    
    # ROC-derived best threshold (Youden's J)
    fpr, tpr, thr = roc_curve(ybin, prob1)
    # roc_curve returns thresholds aligned with tpr/fpr; choose max(tpr - fpr)
    j = tpr - fpr
    best_ix = int(np.argmax(j))
    best_thr = float(thr[best_ix])

    yhat_opt = (prob1 >= best_thr).astype(int)

    acc_opt = float((yhat_opt == ybin).mean())
    bacc    = float(balanced_accuracy_score(ybin, yhat_opt))
    f1      = float(f1_score(ybin, yhat_opt))
    mcc     = float(matthews_corrcoef(ybin, yhat_opt))
    best_thr= best_thr

    return acc_opt, bacc, f1, mcc, best_thr
