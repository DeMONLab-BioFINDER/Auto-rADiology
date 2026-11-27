"""MRI-free alignment of a tau-PET scan to a PET template (no warping), with SimpleITK.

Used by scripts/convert_tau_pkl_to_pt.py --rigid_template. Two uses:
- straighten(): rotation + shift only, the scan keeps its own 2 mm grid and head size
  (tau_raw_2mm_rigid_pts). Then the usual 2 mm recipe runs on it.
- align_to_template() (--template_grid): rotation + shift + ONE overall size factor, fitted inside
  an MNI brain mask and limited to a range (default 0.85-1.05), and the result is resampled onto the
  template's own grid (91x109x91 at 2 mm, like the MNI scans) - an MRI-free stand-in for tau_T1MNI.

Arrays and affines follow nibabel / MONAI: array axes (i, j, k), affine in RAS mm. SimpleITK works
in LPS, so affines are flipped on the way in and out.

SimpleITK is imported lazily, so training does not need it (only the conversion does).
"""
import numpy as np

_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError("SimpleITK is needed for rigid registration: pip install SimpleITK (in the ai-pet env)") from e
    return sitk


def to_sitk(arr, affine_ras):
    """numpy (i, j, k) + RAS affine -> SimpleITK image (float32)."""
    sitk = _sitk()
    aff = _RAS_TO_LPS @ np.asarray(affine_ras, dtype=float)
    m = aff[:3, :3]
    spacing = np.linalg.norm(m, axis=0)
    direction = m / spacing
    # scanner affines can carry a tiny shear; SimpleITK needs an orthonormal direction matrix
    u, _, vt = np.linalg.svd(direction)
    direction = u @ vt
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.asarray(arr, dtype=np.float32).transpose(2, 1, 0)))
    img.SetSpacing([float(s) for s in spacing])
    img.SetOrigin([float(o) for o in aff[:3, 3]])
    img.SetDirection([float(d) for d in direction.flatten()])
    return img


def from_sitk(img):
    """SimpleITK image -> numpy (i, j, k), RAS affine."""
    sitk = _sitk()
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = direction * np.array(img.GetSpacing(), dtype=float)
    aff[:3, 3] = img.GetOrigin()
    return arr, _RAS_TO_LPS @ aff


def _scaled(img):
    """Divide by the 99th percentile of the non-zero voxels, so template and scan are on a similar scale
    (mutual information does not need it, but the initial alignment by centre of mass is less
    sensitive to a bright neck)."""
    sitk = _sitk()
    a = sitk.GetArrayViewFromImage(img)
    v = a[a > 0]
    p = float(np.percentile(v, 99)) if v.size else 1.0
    return sitk.Cast(img, sitk.sitkFloat32) / max(p, 1e-6)


def head_top_mask(img, frac=0.1, top_mm=140.0):
    """Head voxels (> frac x 99th percentile) in the top `top_mm` mm of the head (scanner z), so the
    neck and shoulders, which vary with the field of view, don't steer the registration."""
    sitk = _sitk()
    a = sitk.GetArrayFromImage(img)  # (k, j, i)
    v = a[a > 0]
    if not v.size:
        return None
    m = a > frac * np.percentile(v, 99)
    if not m.any():
        return None
    # physical superior coordinate (LPS z) of every voxel, whichever array axis points up
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    step_z = direction[2] * np.array(img.GetSpacing(), dtype=float)  # z change per step along i, j, k
    k, j, i = np.indices(a.shape, sparse=True)
    z = np.broadcast_to(img.GetOrigin()[2] + step_z[0] * i + step_z[1] * j + step_z[2] * k, a.shape)
    m &= z >= z[m].max() - top_mm
    mask = sitk.GetImageFromArray(m.astype(np.uint8))
    mask.CopyInformation(img)
    return mask


def register_rigid(moving, fixed, fixed_mask=None, seed=42, scale=False, init_tx=None, moving_mask="top"):
    """Rigid registration of `moving` (a native scan) to `fixed` (the template): 6 parameters
    (3 rotations, 3 shifts), or with scale=True 7 (plus ONE overall scale factor, the same in every
    direction, so the head's shape cannot be distorted - only its size matched to the template).
    Mattes mutual information, 3 resolution levels. Starts from the centres of mass of the top
    140 mm of each head, or from `init_tx` (an Euler3DTransform, e.g. from multistart_search).
    moving_mask="top" limits the scan side to the top 140 mm of the head (head_top_mask); None uses
    the whole scan (no assumption about the neck). Returns (transform mapping template points to scan
    points, final metric)."""
    sitk = _sitk()
    fixed_s, moving_s = _scaled(fixed), _scaled(moving)
    moving_mask = head_top_mask(moving_s) if moving_mask == "top" else None

    # initial shift: centre of mass of the top of the head in each image (the neck pulls the plain
    # centre of mass down by several cm in tall fields of view)
    def com(img, mask):
        a = sitk.GetArrayFromImage(img)
        if mask is not None:
            a = a * sitk.GetArrayFromImage(mask)
        idx = np.array(np.nonzero(a > 0))
        w = a[a > 0]
        c = (idx * w).sum(axis=1) / w.sum()  # (k, j, i)
        return np.array(img.TransformContinuousIndexToPhysicalPoint([float(c[2]), float(c[1]), float(c[0])]))

    if init_tx is not None:
        tx = sitk.Euler3DTransform(init_tx)
    else:
        c_fixed = com(fixed_s, fixed_mask)
        c_moving = com(moving_s, moving_mask)
        tx = sitk.Similarity3DTransform() if scale else sitk.Euler3DTransform()
        tx.SetCenter([float(c) for c in c_fixed])
        tx.SetTranslation([float(d) for d in (c_moving - c_fixed)])

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=300,
                                                 relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 1.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fixed_s, moving_s)
    return tx, float(reg.GetMetricValue())


def rotation_summary(tx):
    """Head-pose change from a rigid transform, in degrees and mm. pitch = about the left-right
    axis (nodding), roll = about the back-front axis, yaw = about the up-down axis; total = the
    single rotation angle; shift = length of the translation."""
    flip = np.diag([-1.0, -1.0, 1.0])
    m = np.array(tx.GetMatrix(), dtype=float).reshape(3, 3)
    # template -> scan scale (Similarity3DTransform; 1 for rigid): > 1 = the head is bigger than the template
    scale = float(np.cbrt(np.linalg.det(m)))
    r = flip @ (m / scale) @ flip  # LPS -> RAS, rotation only
    total = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))
    # same convention as src.data.rotation_matrix: r = Rz(yaw) @ Ry(roll) @ Rx(pitch), RAS axes
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    roll = np.degrees(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return {"pitch_deg": float(pitch), "roll_deg": float(roll), "yaw_deg": float(yaw), "total_deg": float(total),
            "shift_mm": float(np.linalg.norm(tx.GetTranslation())), "scale": scale}


def resample_aligned(moving, tx, fixed, spacing_mm=2.0):
    """Resample the whole native scan into a grid lined up with the template's axes (so the head
    pose matches the template), keeping every voxel of the original field of view: the grid is the
    box around the scan's corners mapped into template space. Linear interpolation, 0 outside."""
    sitk = _sitk()
    inv = tx.GetInverse()
    size = moving.GetSize()
    corners = [moving.TransformIndexToPhysicalPoint((i, j, k))
               for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)]
    pts = np.array([inv.TransformPoint(c) for c in corners])
    direction = np.array(fixed.GetDirection(), dtype=float).reshape(3, 3)
    local = pts @ direction  # coordinates along the template's axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    out_size = [int(np.ceil((h - l) / spacing_mm)) + 1 for l, h in zip(lo, hi)]
    origin = direction @ lo
    ref = sitk.Image(out_size, sitk.sitkFloat32)
    ref.SetSpacing([float(spacing_mm)] * 3)
    ref.SetDirection([float(d) for d in direction.flatten()])
    ref.SetOrigin([float(o) for o in origin])
    return sitk.Resample(sitk.Cast(moving, sitk.sitkFloat32), ref, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def straighten(arr, affine_ras, fixed, fixed_mask=None, spacing_mm=2.0, scale=False):
    """One native scan -> (straightened array, its RAS affine, pose-change summary dict)."""
    moving = to_sitk(arr, affine_ras)
    tx, metric = register_rigid(moving, fixed, fixed_mask, scale=scale)
    out = resample_aligned(moving, tx, fixed, spacing_mm)
    out_arr, out_aff = from_sitk(out)
    summary = rotation_summary(tx)
    summary["metric"] = metric
    return out_arr, out_aff, summary


def load_template(path):
    """Template NIfTI -> (SimpleITK image, mask of the template's head)."""
    import nibabel as nib
    sitk = _sitk()
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata(), dtype=np.float32)
    fixed = to_sitk(arr, nii.affine)
    v = arr[arr > 0]
    m = (arr > 0.1 * np.percentile(v, 99)).astype(np.uint8)
    mask = to_sitk(m, nii.affine)
    mask = sitk.Cast(mask > 0.5, sitk.sitkUInt8)
    return fixed, mask


# ---------------------------------------------------------------------------------------------
# Template-grid alignment: rotation + shift + one size factor, fitted on the brain only
# ---------------------------------------------------------------------------------------------

def load_brain_mask(template_path, mask_path=None, erode_vox=1):
    """Brain mask on the template grid, as a SimpleITK image. Default: nilearn's MNI152 brain mask
    resampled onto the template (the template is on the SPM MNI grid, same world coordinates);
    or `mask_path` (any NIfTI in MNI space). Shrunk by `erode_vox` voxels so the bright skull /
    meninges uptake at the brain edge stays out of the size fit (that uptake is what shrank the heads
    in the first rigid+scale try)."""
    import nibabel as nib
    from scipy.ndimage import binary_erosion
    sitk = _sitk()
    tmpl = nib.load(str(template_path))
    target = nib.Nifti1Image(np.zeros(tmpl.shape[:3], np.int8), tmpl.affine)
    from nilearn import image
    if mask_path:
        src = nib.load(str(mask_path))
    else:
        from nilearn import datasets
        src = datasets.load_mni152_brain_mask(resolution=2)
    m = image.resample_to_img(src, target, interpolation="nearest", force_resample=True,
                              copy_header=True).get_fdata() > 0.5
    if erode_vox:
        m = binary_erosion(m, iterations=int(erode_vox))
    if m.sum() < 1000:
        raise ValueError(f"Brain mask on the template grid has only {int(m.sum())} voxels - wrong space?")
    mask = to_sitk(m.astype(np.uint8), tmpl.affine)
    return sitk.Cast(mask > 0.5, sitk.sitkUInt8), m


def _similarity_from(tx, scale):
    """Euler3D or Similarity3D transform -> Similarity3D with the same pose and the given size factor."""
    sitk = _sitk()
    m = np.array(tx.GetMatrix(), dtype=float).reshape(3, 3)
    r = m / np.cbrt(np.linalg.det(m))
    out = sitk.Similarity3DTransform()
    out.SetCenter(tx.GetCenter())
    out.SetMatrix([float(v) for v in r.flatten()])
    out.SetTranslation(tx.GetTranslation())
    out.SetScale(float(scale))
    return out


def _brain_metric(fixed_s, moving_s, brain_mask, seed=42):
    """Mattes mutual information over the template's brain voxels (all of them, so the value is
    deterministic), for evaluating fixed transforms."""
    sitk = _sitk()
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetMetricFixedMask(brain_mask)
    reg.SetInterpolator(sitk.sitkLinear)

    def value(tx):
        reg.SetInitialTransform(tx, inPlace=False)
        return float(reg.MetricEvaluate(fixed_s, moving_s))
    return value


def _fit_scale(tx, value, lo, hi):
    """Best size factor in [lo, hi] for a fixed pose: grid in steps of 0.01, then 0.002 around the
    best one. A plain search instead of the optimizer, so the size can never leave the range."""
    def best(grid):
        vals = [value(_similarity_from(tx, s)) for s in grid]
        i = int(np.argmin(vals))  # mutual information: lower = better match
        return float(grid[i]), float(vals[i])
    s, _ = best(np.round(np.arange(lo, hi + 1e-9, 0.01), 4))
    fine = np.round(np.arange(max(lo, s - 0.01), min(hi, s + 0.01) + 1e-9, 0.002), 4)
    return best(fine)


def _refine_pose(tx, fixed_s, moving_s, brain_mask, moving_mask, seed=42):
    """Re-fit rotation + shift with the size factor frozen (optimizer weight 0), on the brain mask."""
    sitk = _sitk()
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed)
    reg.SetMetricFixedMask(brain_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=200,
                                                 relaxationFactor=0.5, gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetOptimizerWeights([1.0] * 6 + [0.0])  # Similarity3D: versor (3), shift (3), scale (frozen)
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fixed_s, moving_s)
    return tx


def _com_physical(img, thresh_frac=0.1):
    """Intensity-weighted centre of everything brighter than thresh_frac x 99th percentile."""
    sitk = _sitk()
    a = sitk.GetArrayFromImage(img)
    v = a[a > 0]
    if not v.size:
        return np.array(img.TransformContinuousIndexToPhysicalPoint([s / 2 for s in img.GetSize()]))
    w = np.where(a > thresh_frac * np.percentile(v, 99), a, 0)
    idx = np.array(np.nonzero(w))
    c = (idx * w[w > 0]).sum(axis=1) / w[w > 0].sum()  # (k, j, i)
    return np.array(img.TransformContinuousIndexToPhysicalPoint([float(c[2]), float(c[1]), float(c[0])]))


def _resample_iso(img, spacing_mm):
    """Same field of view and axes, cubic voxels of spacing_mm (linear)."""
    sitk = _sitk()
    old_sp, old_size = np.array(img.GetSpacing()), np.array(img.GetSize())
    size = [max(1, int(round(n * o / spacing_mm))) for n, o in zip(old_size, old_sp)]
    return sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), [float(spacing_mm)] * 3,
                         img.GetDirection(), 0.0, sitk.sitkFloat32)


def multistart_search(moving, fixed, fixed_mask, step_mm=15.0, ap_mm=(-30, -15, 0, 15, 30),
                      pitch_deg=(-30, -15, 0, 15, 30), keep=3):
    """Starting positions for the rigid fit that do not depend on the neck, shoulders or field of
    view: the template head's centre is placed at every height of the scan (bottom to top, every
    `step_mm`), at several front/back offsets and nod (pitch) angles; each start is scored by mutual
    information inside the template's head (fixed_mask) on coarse 8 mm copies of both images. Left-
    right comes from the scan's centre of brightness (heads are centred left-right). Returns the
    `keep` best starts as Euler3DTransforms (template -> scan), best first."""
    sitk = _sitk()
    shrink = 4  # template 2 mm -> 8 mm
    fixed_c = sitk.Shrink(sitk.SmoothingRecursiveGaussian(_scaled(fixed), 4.0), [shrink] * 3)
    mask_c = sitk.Resample(fixed_mask, fixed_c, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    moving_c = _resample_iso(sitk.SmoothingRecursiveGaussian(_scaled(moving), 4.0), 8.0)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=24)
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetMetricFixedMask(mask_c)
    reg.SetInterpolator(sitk.sitkLinear)

    a = sitk.GetArrayFromImage(fixed_c) * sitk.GetArrayFromImage(mask_c)
    idx = np.array(np.nonzero(a > 0))
    c = (idx * a[a > 0]).sum(axis=1) / a[a > 0].sum()
    c_fixed = np.array(fixed_c.TransformContinuousIndexToPhysicalPoint([float(c[2]), float(c[1]), float(c[0])]))
    c_scan = _com_physical(moving_c)
    size = moving.GetSize()
    z = [moving.TransformIndexToPhysicalPoint((i, j, k))[2]
         for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)]
    heights = np.arange(min(z), max(z) + 1e-6, step_mm)
    scored = []
    for h in heights:
        for dy in ap_mm:  # LPS y = posterior; sign does not matter, both directions are tried
            target = np.array([c_scan[0], c_scan[1] + dy, h])
            for p in pitch_deg:
                tx = sitk.Euler3DTransform()
                tx.SetCenter([float(v) for v in c_fixed])
                tx.SetRotation(float(np.radians(p)), 0.0, 0.0)  # about the left-right axis
                tx.SetTranslation([float(v) for v in target - c_fixed])
                reg.SetInitialTransform(tx, inPlace=False)
                try:
                    val = float(reg.MetricEvaluate(fixed_c, moving_c))
                except RuntimeError:  # most of the template head falls outside the scan
                    continue
                scored.append((val, tx))
    if not scored:
        raise RuntimeError("Multi-start search: no starting position overlaps the scan")
    scored.sort(key=lambda t: t[0])
    return [t for _, t in scored[:keep]]


def template_correlation(moving, tx, fixed, brain_mask):
    """Quality score of an alignment: Pearson r between the aligned scan and the template inside the
    template's brain mask (needs no MRI, so it also works for new scans). Low = look at this scan."""
    sitk = _sitk()
    out = sitk.Resample(sitk.Cast(moving, sitk.sitkFloat32), fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    m = sitk.GetArrayViewFromImage(brain_mask) > 0
    a = sitk.GetArrayViewFromImage(out)[m].astype(np.float64)
    b = sitk.GetArrayViewFromImage(fixed)[m].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def register_template(moving, fixed, fixed_head_mask, brain_mask, scale_range=(0.85, 1.05), rounds=2, seed=42,
                      multistart=True):
    """Rotation + shift + one size factor (7 parameters, no warping):
    1. rigid fit. multistart=True (default): from the best few starts of multistart_search, on the
       whole scan (no neck / top-of-head assumption), keeping the best by brain-mask match.
       multistart=False: one start from the top of the head (as for tau_raw_2mm_rigid_pts; failed for
       ~1 in 4 hard pilot scans, Geneva / AVID / OASIS, Nov 2026);
    2. size factor by search inside the brain mask, limited to scale_range;
    3. rotation + shift re-fitted on the brain with that size; 2-3 repeated `rounds` times.
    Returns (Similarity3D transform template -> scan, info dict)."""
    sitk = _sitk()
    lo, hi = scale_range
    # light smoothing (2 mm) so the size search sees a smooth metric, not voxel noise
    fixed_s = sitk.SmoothingRecursiveGaussian(_scaled(fixed), 2.0)
    moving_s = sitk.SmoothingRecursiveGaussian(_scaled(moving), 2.0)
    value = _brain_metric(fixed_s, moving_s, brain_mask, seed)
    if multistart:
        best = None
        for start in multistart_search(moving, fixed, fixed_head_mask):
            try:
                cand, _ = register_rigid(moving, fixed, fixed_head_mask, seed=seed, init_tx=start, moving_mask=None)
                v = value(_similarity_from(cand, 1.0))
            except RuntimeError:
                continue
            if best is None or v < best[0]:
                best = (v, cand)
        if best is None:
            raise RuntimeError("Multi-start: every rigid fit failed")
        rigid = best[1]
        moving_mask = None
    else:
        rigid, _ = register_rigid(moving, fixed, fixed_head_mask, seed=seed, scale=False)
        moving_mask = head_top_mask(_scaled(moving))
    tx = _similarity_from(rigid, 1.0)
    metric_rigid = value(tx)
    for _ in range(rounds):
        s, _ = _fit_scale(tx, value, lo, hi)
        tx = _similarity_from(tx, s)
        tx = _refine_pose(tx, _scaled(fixed), _scaled(moving), brain_mask, moving_mask, seed)
    s = float(tx.GetScale())
    return tx, {"metric_rigid": metric_rigid, "metric": value(tx),
                "scale_at_limit": bool(min(s - lo, hi - s) < 0.005),
                "r_template": template_correlation(moving, tx, fixed, brain_mask)}


def align_to_template(arr, affine_ras, fixed, fixed_head_mask, brain_mask, scale_range=(0.85, 1.05), multistart=True):
    """One native scan -> (array on the template grid, template RAS affine, summary dict).
    The output has exactly the template's grid (size, 2 mm voxels, orientation), like the MNI scans."""
    sitk = _sitk()
    moving = to_sitk(arr, affine_ras)
    tx, info = register_template(moving, fixed, fixed_head_mask, brain_mask, scale_range, multistart=multistart)
    out = sitk.Resample(sitk.Cast(moving, sitk.sitkFloat32), fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    out_arr, out_aff = from_sitk(out)
    summary = rotation_summary(tx)
    summary.update(info)
    return out_arr, out_aff, summary
