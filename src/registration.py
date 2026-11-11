"""MRI-free rigid alignment of a tau-PET scan to a PET template (rotation + shift only, no scaling,
no warping), with SimpleITK.

Used by scripts/convert_tau_pkl_to_pt.py --rigid_template to make a "straightened" version of the
2 mm input: each native scan is turned so its head pose matches the template, then the usual 2 mm
recipe runs on it. Only the head pose changes; voxel size, head size and brightness scaling are as
in tau_raw_2mm_pts.

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


def register_rigid(moving, fixed, fixed_mask=None, seed=42, scale=False):
    """Rigid registration of `moving` (a native scan) to `fixed` (the template): 6 parameters
    (3 rotations, 3 shifts), or with scale=True 7 (plus ONE overall scale factor, the same in every
    direction, so the head's shape cannot be distorted - only its size matched to the template).
    Mattes mutual information, 3 resolution levels. Starts from the centres of mass of the top
    140 mm of each head. Returns (transform mapping template points to scan points, final metric)."""
    sitk = _sitk()
    fixed_s, moving_s = _scaled(fixed), _scaled(moving)
    moving_mask = head_top_mask(moving_s)

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
