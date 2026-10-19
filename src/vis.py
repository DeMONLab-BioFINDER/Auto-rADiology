import os
import math
import time
import torch
import numpy as np
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import gaussian_gradient_magnitude
from monai.visualize.class_activation_maps import GradCAM
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity


def run_visualization(model, loader, device, output_path, vis_name="gradcam",
                      vis_kwargs=None, affine_fn=None, # optional: get affine per subject
                      vis_norm="zscore", skull_suppress_pct=20,      # PET intensity percentile
                      use_pseudo_surface=False, pseudo_surface_sigma=1.0, pseudo_surface_pct=90):
    """
    Run volumetric explainability visualizations for 3D PET models.

    For each subject, this function computes a voxel-wise attribution map
    (e.g., Grad-CAM or occlusion) in model input space, suppresses background
    and skull-driven activations using PET intensity thresholds, and saves
    subject-level and group-average results.

    Outputs include:
      - Volumetric NIfTI attribution maps
      - Maximum-intensity-projection (MIP) overlays on PET
      - Optional PET-derived pseudo-surface visualizations (approximate,
        gradient-based; not anatomical surfaces)

    All visualizations are performed in the model input (preprocessed PET)
    space without registration to standard templates.
    """
    vis_type = ['class','pos_vs_neg']
    VIS_REGISTRY = {
        "gradcam": gradcam_vis_fn,
        "occlusion": occlusion_vis_fn,
        }
    vis_fn = VIS_REGISTRY[vis_name]
    print('vis_fn:',vis_fn)
    vis_kwargs = vis_kwargs or {}

    vis_dir = os.path.join(output_path, 'visualization', vis_name)
    os.makedirs(vis_dir, exist_ok=True)

    vis_sum = {'class': None,'pos_vs_neg':None}
    vis_count = {'class': 0, 'pos_vs_neg': 0}
    img_sum, img_n = None, 0  # running mean of the input images, for the group-average MIP background
    for x, _, _, extra, pid in loader:
        x = x.to(device)
        extra = extra.to(device)
        
        for b in range(x.shape[0]):
            xb = x[b:b+1] # [1, C, D, H, W]
            extrab = extra[b:b+1] if extra is not None else None
            pid_b = pid[b]

            # visualize
            heat_class, heat_pos_vs_neg = vis_fn(model=model, x=xb, extra=extrab, **vis_kwargs)  # [D,H,W]

            img_np = xb[0, 0].detach().cpu().numpy()  # (D,H,W)
            img_sum = img_np.astype(np.float64) if img_sum is None else img_sum + img_np
            img_n += 1

            for heat, vt in zip([heat_class, heat_pos_vs_neg], vis_type):
                if heat is None: continue # skip when heat_pos_vs_neg is None
                assert heat.ndim == 3

                # PET-based brain & skull suppression
                brain_mask = img_np > np.percentile(img_np[img_np > 0], skull_suppress_pct)
                heat = heat * brain_mask

                if vis_norm is not None: heat = normalize_cam(heat, vis_norm)

                # save to NIFTI
                affine = affine_fn(pid_b) if affine_fn else np.eye(4)
                nib.save(nib.Nifti1Image(heat.astype(np.float32), affine),
                        os.path.join(vis_dir, f"{pid_b}_{vis_name}_{vt}.nii.gz"))

                # MIP visualization
                save_mip_png(img_np, heat, os.path.join(vis_dir, f"{pid_b}_{vis_name}_{vt}_mip.png"))
                
                # Pseudo-surface visualization (no T1w available, no affine)
                if use_pseudo_surface:
                    shell = pet_pseudo_surface(img_np, sigma=pseudo_surface_sigma, pct=pseudo_surface_pct)
                    shell_cam = heat * shell
                    save_mip_png(img_np, shell_cam, os.path.join(vis_dir, f"{pid_b}_{vis_name}_{vt}_pseudo_surface.png"))

                vis_sum[vt] = heat if vis_sum[vt] is None else vis_sum[vt] + heat
                vis_count[vt] += 1
    
    print('saving average image to', vis_dir)
    # Brains are not registered to a common space (model input space is native/unregistered PET),
    # so the group average and its background image are only approximate overlays.
    img_avg = (img_sum / img_n) if img_n else None
    for vt in vis_type:
        if vis_sum[vt] is None:
            print(f"Skipping {vt} visualization (no heatmaps generated)")
            continue
        # group-level average (map type in the filename, so 'pos_vs_neg' no longer overwrites 'class')
        vis_avg = vis_sum[vt] / vis_count[vt]
        nib.save(nib.Nifti1Image(vis_avg.astype(np.float32), np.eye(4)),
                os.path.join(vis_dir, f"group_average_{vis_name}_{vt}.nii.gz"))
        save_mip_png(img_avg, vis_avg, os.path.join(vis_dir, f"group_average_{vis_name}_{vt}_mip.png"))

        if use_pseudo_surface:
            shell = pet_pseudo_surface(img_avg, pseudo_surface_sigma, pseudo_surface_pct)
            save_mip_png(img_avg, vis_avg * shell, os.path.join(vis_dir, f"group_average_{vis_name}_{vt}_pseudo_surface.png"))


def gradcam_vis_fn(model, x, extra, target_layer=None, normalize=True, upsample_mode="trilinear"):
    model.eval()
    
    def fwd(x_):
        return model(x_, extra=extra)
    
    # pick last Conv3d
    if target_layer is None:
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv3d):
                target_layer = name
    if target_layer is None: raise ValueError("No Conv3d layer found")

    cam = GradCAM(model, target_layers=[target_layer])

    with torch.amp.autocast("cuda", enabled=False):
        cam_map = cam(x, extra=extra)  # calls fwd(x) # class_idx – Default to None (computing class_idx from argmax); layer_idx – index of the target layer if there are multiple target layers

    # post-proc
    if cam_map.shape[-3:] != x.shape[-3:]: cam_map = F.interpolate(cam_map, x.shape[-3:], mode=upsample_mode) # upsample
    cam_np = cam_map[0, 0].cpu().numpy()
    if normalize: # normalize
        vmin, vmax = cam_np.min(), cam_np.max()
        cam_np = (cam_np - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(cam_np)

    return cam_np, None


def _parse_occlusion_mode(mode):
    """MONAI's OcclusionSensitivity `mode`: 'gaussian' (its default: the patch is dimmed by a
    Gaussian-weighted mask, not blanked outright), 'mean_img', 'mean_patch', or a constant
    float fill (e.g. 0.0 to blank the patch to zero). A numeric string becomes a float."""
    if isinstance(mode, str):
        try:
            return float(mode)
        except ValueError:
            return mode
    return mode


def _brain_bbox(x):
    """Inclusive bounding box [min0, max0, min1, max1, min2, max2] of the non-zero voxels of
    x ([1,C,D,H,W]); the dataset zeroes everything outside the brain, so this is the
    brain's bounding box in model input space. None if x is entirely zero."""
    nz = torch.nonzero(x[0, 0] != 0)
    if nz.numel() == 0:
        return None
    lo, hi = nz.min(0).values.tolist(), nz.max(0).values.tolist()
    return [v for pair in zip(lo, hi) for v in pair]


def _softmax_over_classes(out):
    return out.softmax(dim=1)


@torch.no_grad()
def occlusion_vis_fn(model, x, extra, **kwargs):
    """Occlusion sensitivity via MONAI, returned as the DROP in class probability.

    For every occluded patch, drop = baseline probability (nothing occluded) minus the
    probability with that patch occluded. Positive = the patch supports the class (hiding
    it lowers the model's confidence), ~0 = no effect (background and everywhere that does
    not matter), negative = hiding it raises the class probability (evidence against).
    Returns (map for the predicted class, map of "positive over negative", i.e. the drop in
    P(class 1) minus the drop in P(class 0)), both [D,H,W].

    Probabilities are the model's own softmax over its 2 logits. MONAI's activate=True would
    instead apply a per-logit sigmoid here (it picks sigmoid whenever the input has 1 channel,
    which PET always does), which is not the probability the classifier reports.

    kwargs: mask_size (int or 3-tuple, default 8), overlap (default 0.75), n_batch (default
    16), mode (see _parse_occlusion_mode, default 'gaussian'), activate (default: softmax
    over classes; False = raw logits, or pass your own callable), brain_bbox (default False:
    only occlude inside the bounding box of the brain, the map outside is 0)."""
    model.eval()

    activate = kwargs.get("activate", _softmax_over_classes)
    if activate is True:
        activate = _softmax_over_classes
    elif activate is False:
        activate = None

    mask_size = kwargs.get("mask_size", (8, 8, 8))
    if isinstance(mask_size, list):
        mask_size = tuple(mask_size)
    occ = OcclusionSensitivity(nn_module=model, mask_size=mask_size,
                               n_batch=kwargs.get("n_batch", 16), overlap=kwargs.get("overlap", 0.75),
                               mode=_parse_occlusion_mode(kwargs.get("mode", "gaussian")),
                               activate=activate if activate is not None else False,)

    b_box = _brain_bbox(x) if kwargs.get("brain_bbox", False) else None
    occ_map, _ = occ(x, b_box=b_box, extra=extra)  # [1,N,d,h,w] occluded-patch probabilities (cropped to b_box if given), N = classes
    occ_map = occ_map.float()[0].cpu().numpy()  # [N,d,h,w]

    baseline_out = model(x, extra=extra)
    pred_class = baseline_out.argmax(dim=1).item()
    baseline = (activate(baseline_out) if activate is not None else baseline_out).float()[0].cpu().numpy()  # [N]
    heat = baseline[:, None, None, None] - occ_map  # drop in class probability, [N,d,h,w]

    if b_box is not None:
        # MONAI crops to the bounding box plus half a mask on each side (clamped at 0); put the
        # cropped map back into a full-size volume (0 = no effect) so everything downstream is unchanged.
        m = _as_triple(mask_size)
        start = [max(b - (mi - 1) // 2, 0) for b, mi in zip(b_box[::2], m)]
        full = np.zeros((heat.shape[0],) + tuple(x.shape[-3:]), dtype=heat.dtype)
        d, h, w = heat.shape[1:]
        full[:, start[0]:start[0] + d, start[1]:start[1] + h, start[2]:start[2] + w] = heat
        heat = full

    heat_class = heat[pred_class] # Why did the model predict this class?
    heat_pos_vs_neg = heat[1] - heat[0]

    return heat_class, heat_pos_vs_neg


def _as_triple(v):
    return tuple(v) if isinstance(v, (list, tuple)) else (int(v),) * 3


def count_occlusion_passes(spatial_shape, mask_size, overlap):
    """Approximate number of occluded forward passes MONAI's sliding-window occlusion runs
    for a volume of spatial_shape: per axis ceil((size - mask) / interval) + 1 windows, with
    interval = max(int(mask * (1 - overlap)), 1), multiplied over the 3 axes."""
    n = 1
    for size, m in zip(spatial_shape, _as_triple(mask_size)):
        interval = max(int(m * (1 - overlap)), 1)
        n *= int(math.ceil(float(max(size - m, 0)) / interval)) + 1
    return n


@torch.no_grad()
def benchmark_occlusion(model, x, extra, device, mask_size=8, overlap=0.75, n_batch=16,
                        n_timed_batches=3, extra_configs=((16, 0.5), (16, 0.75), (8, 0.5), (8, 0.75))):
    """Time a few batches of plain forward passes of the model on this subject's volume
    (n_batch copies per batch, the same batch shape occlusion uses) and project the total
    time of each occlusion setting from passes/second. Cheaper than running the full
    occlusion just to learn how long it would take. Projection only: the occlusion's own
    overhead (building the masks, sliding-window bookkeeping) is not included."""
    model.eval()
    xb = x[:1].expand(n_batch, -1, -1, -1, -1).contiguous()
    eb = extra[:1].expand(n_batch, -1).contiguous() if extra is not None else None

    def _sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elif str(device).startswith("mps"):
            torch.mps.synchronize()

    model(xb, extra=eb)  # warm-up (cuDNN autotune, allocator)
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_timed_batches):
        model(xb, extra=eb)
    _sync()
    elapsed = time.perf_counter() - t0
    passes_per_sec = n_batch * n_timed_batches / elapsed

    configs = [(mask_size, overlap)] + [c for c in extra_configs if c != (mask_size, overlap)]
    shape = tuple(x.shape[-3:])
    print(f"[benchmark] {passes_per_sec:.1f} forward passes/s ({n_batch} x {n_timed_batches} passes in {elapsed:.2f}s) on {device}, volume {shape}")
    print("[benchmark] projected whole-volume occlusion time (mask size, overlap -> passes -> est. time):")
    rows = []
    for m, ov in configs:
        n_pass = count_occlusion_passes(shape, m, ov)
        secs = n_pass / passes_per_sec
        label = f"{secs / 60:.1f} min" if secs < 3600 else f"{secs / 3600:.1f} h"
        tag = "  <- current settings" if (m, ov) == (mask_size, overlap) else ""
        print(f"[benchmark]   mask {m}, overlap {ov}: {n_pass:,} passes -> {label}{tag}")
        rows.append((m, ov, n_pass, secs))
    return passes_per_sec, rows


def normalize_cam(cam, mode="zscore", eps=1e-6):
    mask = cam > 0
    if not np.any(mask):
        return cam

    vals = cam[mask]

    if mode == "zscore":
        cam = (cam - vals.mean()) / (vals.std() + eps)
        cam[cam < 0] = 0
    elif mode == "percentile":
        lo, hi = np.percentile(vals, [80, 99])
        cam = np.clip((cam - lo) / (hi - lo + eps), 0, 1)

    return cam


def save_mip_png(img, cam, out_png, cmap="hot"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    views = [(img.max(0), cam.max(0), "Axial"),
             (img.max(1), cam.max(1), "Coronal"),
             (img.max(2), cam.max(2), "Sagittal")]

    for ax, (im, hm, title) in zip(axes, views):
        ax.imshow(im, cmap="gray")
        ax.imshow(hm, cmap=cmap, alpha=0.6)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def pet_pseudo_surface(img, sigma=1.0, pct=90):
    """
    PET-derived cortical shell proxy using gradient magnitude.
    """
    grad = gaussian_gradient_magnitude(img, sigma=sigma)
    thr = np.percentile(grad[grad > 0], pct)
    shell = grad >= thr
    return shell.astype(np.float32)
