# src/utils.py
import os
import random

import numpy as np
import torch

from types import SimpleNamespace


def set_seed(seed: int = 42, deterministic: bool = False, set_pythonhashseed: bool = True):
    if set_pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Seeds CPU and (per PyTorch docs) CUDA RNG; no separate call for MPS exists.
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    # Global determinism (may error if an op lacks a deterministic variant)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[warn] deterministic_algorithms not fully supported: {e}")

    #  MONAI convenience:
    from monai.utils import set_determinism as monai_set_det
    monai_set_det(seed=seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_device(prefer_cuda=True, force_cpu=False):
    """
    Returns a torch.device among: cuda, mps, cpu.
    prefer_cuda: if True, choose CUDA over MPS when both are present (e.g., eGPU on Mac).
    """
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None \
              and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if prefer_cuda and has_cuda and not force_cpu: return torch.device("cuda")
    if not prefer_cuda and has_mps: return torch.device("mps")
    if not force_cpu and has_cuda: return torch.device("cuda")
    if not force_cpu and has_mps: return torch.device("mps")

    torch.backends.cudnn.benchmark = True  # 3D convs benefit
    return torch.device("cpu")


def compute_smooth_sigma_vox(voxel_sizes_mm: tuple[float, float, float], fwhm_current_mm: float, fwhm_target_mm: float) -> tuple[float, float, float] | None:
    """
    Returns per-axis sigma in *voxels* for MONAI.GaussianSmooth to top-up smoothing
    from fwhm_current_mm to fwhm_target_mm. If no extra smoothing needed, returns (0,0,0).
    """
    if fwhm_target_mm <= fwhm_current_mm:
        return (0.0, 0.0, 0.0)

    fwhm_extra_mm = np.sqrt(fwhm_target_mm**2 - fwhm_current_mm**2)  # quadrature
    sigma_mm = fwhm_extra_mm / 2.354820045  # mm → σ
    vx, vy, vz = voxel_sizes_mm
    return (sigma_mm / vx, sigma_mm / vy, sigma_mm / vz)


def clone_args(args, **overrides):
    """Create a shallow, mutable copy of args with some fields overridden."""
    d = vars(args).copy()
    d.update(overrides)
    return SimpleNamespace(**d)
