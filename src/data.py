# src/data.py
import os
import re
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from pathlib import Path
from typing import List, Optional, Union
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from monai.data import MetaTensor
from monai.transforms import (LoadImage, EnsureChannelFirst, Orientation, Resize,
        ScaleIntensityRangePercentiles, Compose, CropForeground, GaussianSmooth,
        Lambda, Spacing, ResizeWithPadOrCrop, SpatialCrop, BorderPad)

from src.utils import seed_worker
# ------------------------------
# Master table
# ------------------------------
def build_master_table(input_path: str, preproce_method: str, targets: List[str], dataset: str, data_type: str, subjects: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build the table required by the training code, given the custom folder layout.
    Normal mode: discover images + join demographics.
    Cached mode: if preproce_method is empty/None OR cache files exist in cache_dir,
                 load demo.csv only (no disk scans), and return that table.
    In cached mode, adds 'ID' to preserve the order matching data.pt.
    """
    # Detect cached mode
    use_cache = (not preproce_method) or (str(preproce_method).strip() == "")
    if use_cache:
        csv = Path(input_path) / "demo.csv"
        if not csv.exists():
            csv = Path(input_path) / f"demo_{dataset}_{data_type}.csv"
        df = pd.read_csv(csv)
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        print(f"[cache] Loaded {csv} with {len(df)} rows (no filesystem scan).")
    else:
        pets = find_pet_files(input_path=input_path, preproc_method=preproce_method, allow=subjects)
        if pets.empty:
            raise FileNotFoundError(f"No NIfTI files found under '{input_path}' with preproc suffix  '{preproce_method}' for dataset '{dataset}'.")
        else:
            print(f'Found {pets.shape[0]} scans')
            print(pets.columns, pets.head(3))

        labels = load_participants_labels(input_path, dataset=dataset)
        labels["ID"] = labels["ID"].astype(str).str.strip()
        #df = pd.merge(pets, labels, on="ID", how="inner")
        has_date = "ScanDate" in labels.columns
        if has_date:
            # ensure YYYY-MM-DD strings
            labels["ScanDate"] = pd.to_datetime(labels["ScanDate"]).dt.strftime("%Y-%m-%d")
            pets["ScanDate"] = pets["ScanDate"].astype(str).str.slice(0,10)
            keys = ["ID", "ScanDate"]
        else:
            keys = ["ID"]

        df = pd.merge(pets, labels, on=keys, how="inner", suffixes=("", "_selected"))
        df = df.sort_values(keys + [c for c in ["pet_path"] if c in df.columns]).reset_index(drop=True)

    # Only scans with targets value
    targets_list = [t.strip() for t in targets.split(",") if t.strip()]
    # Keep only rows that have ALL target values
    df = df.dropna(subset=targets_list).reset_index(drop=True)
    
    print(f'Found {df.shape[0]} scans with demographics for {targets}')

    return df


def find_pet_files(input_path: str, preproc_method: str, allow: Optional[Union[pd.DataFrame, str, Path]] = None) -> pd.DataFrame:
    """
    Discover PET NIfTI files via:
      (A) input_path/PET/**/*_{preproc_method}/*/*/*.nii*
      (B) input_path/<ID>/PET_<ScanDate>_<tracer>/SCANS/*.nii[.gz], tracer in {FBB, FBP}

    If `allow` (DataFrame or CSV) is provided:
      - must contain 'ID'; optional 'ScanDate'
      - if only 'ID': filter by ID (keep all columns combined)
      - if 'ID' + 'ScanDate': inner-join on both (keep all columns combined)
    """
    ipath = Path(input_path)
    rows = []

    # Pattern A
    root_a = ipath / "PET"
    pattern_a = f"**/*{preproc_method}/*/*/*.nii*"
    if root_a.exists():
        for nii in root_a.glob(pattern_a):
            try:
                sid = nii.relative_to(root_a).parts[0]
            except Exception:
                continue
            rows.append({
                "ID": str(sid),
                "pet_path": str(nii).replace('/._','/'),
                "imagefile": nii.name,
                "ScanDate": None,
                "tracer": None,
            })
    else:
        # Pattern B -- ADNI data on Berkeley cluster
        # e.g. /116-S-6550/PET_2018-08-29_FTP/SCANS/116-S-6550_AV1451_2018-08-29_P4-6mm_I1600375.nii
        if '.nii' in preproc_method:
            print(f'finding ADNI data /analysis/{preproc_method}')
            method_pat = re.escape(preproc_method) + r'(?:\.gz)?'
            regex_b = re.compile(
                rf'^(?P<ID>[^/]+)/PET_(?P<ScanDate>\d{{4}}-\d{{2}}-\d{{2}})_(?P<tracer>FBB|FBP)/analysis/{re.escape(preproc_method)}$')
            #regex_b = re.compile(
            #    r'^(?P<ID>[^/]+)/PET_(?P<ScanDate>\d{4}-\d{2}-\d{2})_(?P<tracer>FBB|FBP)/analysis/'r'wsuvr_cere[^/]*\.nii(?:\.gz)?$')
            glob_pattern = "*analysis/*.nii*"
        else: # step 4 ADNI data
            print(f'finding ADNI data /SCANS/')
            regex_b = re.compile(
                r'^(?P<ID>[^/]+)/PET_(?P<ScanDate>\d{4}-\d{2}-\d{2})_(?P<tracer>FBB|FBP)/SCANS/[^/]+\.nii(\.gz)?$')
            glob_pattern = "*SCANS/*.nii*"
        print("PATTERN:", regex_b.pattern)
        
        for nii in ipath.rglob(glob_pattern):
            m = regex_b.match(nii.relative_to(ipath).as_posix())
            if not m: 
                continue
            rows.append({
                "ID": m.group("ID").replace('-','_'),
                "pet_path": str(nii),
                "imagefile": nii.name,
                "ScanDate": m.group("ScanDate"),
                "tracer": m.group("tracer"),
            })

    df = pd.DataFrame(rows, columns=["ID", "pet_path", "imagefile", "ScanDate", "tracer"])
    if df.empty: return df

    # Deterministic dedupe: keep first path per ID
    df = (df.sort_values(["ID", "pet_path"])
            .groupby("ID", as_index=False, group_keys=False)
            .head(1).reset_index(drop=True))

    # --- Unified filtering / intersection with `allow` ---
    if allow is not None:
        if isinstance(allow, (str, Path)):
            allow_df = pd.read_csv(input_path + allow, index_col=0)
        else:
            allow_df = allow.copy()

        if "ID" not in allow_df.columns:
            raise ValueError("`allow` must contain column 'ID'.")

        # Normalize types
        df["ID"] = df["ID"].astype(str)
        allow_df["ID"] = allow_df["ID"].astype(str)

        has_date = "ScanDate" in allow_df.columns
        if has_date:
            # ensure YYYY-MM-DD strings
            allow_df["ScanDate"] = pd.to_datetime(allow_df["ScanDate"]).dt.strftime("%Y-%m-%d")
            df["ScanDate"] = df["ScanDate"].astype(str).str.slice(0,10)
            keys = ["ID", "ScanDate"]
        else:
            keys = ["ID"]

        df = pd.merge(df, allow_df, on=keys, how="inner", suffixes=("", "_allow"))
        df = df.sort_values(keys + [c for c in ["pet_path"] if c in df.columns]).reset_index(drop=True)

    return df


def load_participants_labels(input_path: str, dataset: Optional[str] = None) -> pd.DataFrame: #cache: Optional[bool] = False
    """
    Load demographics.csv from input_path and return:
    ID, site, visual_read, CL, age, gender
    """
    csv = Path(input_path) / "demographics.csv"
    if not csv.exists(): # try alternative path
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        csv = Path(os.path.join(proj_path, "data")) / f"demographics_{dataset}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing {csv}. Provide columns: ID, site, visual_read, CL, age, gender, ...")
    df = pd.read_csv(csv, index_col=0)
    
    print(f'loaded participants from {csv}:', df.shape, df.columns, df.head(3))
    # Ensure required columns are present in the dataframe.
    required = {"ID", "site", "visual_read", "CL", "age", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    return df

# ------------------------------
# Transforms & Dataset
# ------------------------------
def get_train_val_loaders(train_df, val_df, args, repeat_train: bool = True, eval_augment: bool = False):
    # Detect cached mode
    use_cache = (not args.data_suffix) or (str(args.data_suffix).strip() == "")
    if use_cache:
        tfm = None
        p = Path(args.input_path) / "data.pt"
        if p.exists():
            data_file = torch.load(p, map_location="cpu", weights_only=True) # torch tensor with shape [S, D, H, W]
            # Note: don't print here — loader-specific messages are emitted in `get_loader`.
        else:
            data_file = Path(args.input_path) / args.data_type
            # Defer logging to `get_loader` so messages indicate which split (train/val/test) is being created.
            if any(Path(data_file).glob("tau_batch_*.pt")):
                pass
            elif any(Path(data_file).glob("data_*.pt")):
                pass
            else:
                tfm = get_transforms(tuple(args.image_shape))
    else:
        tfm = get_transforms(tuple(args.image_shape))
        data_file = None

    if repeat_train and args.train_repeat > 1:
        train_df = train_df.loc[train_df.index.repeat(args.train_repeat)].reset_index(drop=True)

    dl_tr = get_loader(train_df, tfm, data_file, args, batch_size=args.batch_size, augment=True, shuffle=True, train_test='train')
    dl_va = get_loader(val_df, tfm, data_file, args, batch_size=args.batch_size, augment=eval_augment, shuffle=False, train_test='test')
    
    return dl_tr, dl_va


def get_loader(df, tfm, data_file, args, batch_size, augment=False, shuffle=False, train_test='train'):
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Informative logging about how this loader is constructed (train/val/test)
    try:
        dfpath = Path(data_file) if data_file is not None else None
    except Exception:
        dfpath = None
    if dfpath is not None:
        if any(dfpath.glob("tau_batch_*.pt")):
            print(f"Reconstructed {train_test} loader from preprocessed tau_batch_*.pt batches. Skipping MONAI preprocessing.")
        elif any(dfpath.glob("data_*.pt")):
            print(f"Reconstructed {train_test} loader from data_*.pt per-scan files.")
        elif dfpath.name == 'data.pt':
            print(f"Reconstructed {train_test} loader from data.pt (in-memory tensor).")
        else:
            # No preprocessed pt batches found; images will be loaded with MONAI transforms
            print(f"Creating {train_test} loader using MONAI transforms (from raw images).")

    dataset = PETDataset(df, tfm, args.targets, data_file=data_file, input_cl=args.input_cl, extra_global_feats=args.extra_global_feats, augment=augment)

    balance_col = getattr(args, "balance_sampling_by", "") or None
    if balance_col and balance_col in df.columns and train_test == 'train':
        print(f'------ Balanced sampling by {balance_col} ------')
        # inverse-frequency weights
        counts = df[balance_col].value_counts().to_dict()
        weights = df[balance_col].map(lambda d: 1.0 / counts[d]).values
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    elif balance_col and train_test == 'train':
        print(f"[WARNING] --balance_sampling_by {balance_col} set but '{balance_col}' not in dataframe columns; skipping balanced sampling.")
        sampler = None
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                        worker_init_fn=seed_worker, generator=g, 
                        num_workers=args.num_workers, pin_memory=False)

    return loader


def brain_outer_mask(x):
    """
    Returns mask of same shape: 1 inside brain, 0 outside.
    """
    vol = x.squeeze().cpu().numpy()

    # Simple foreground threshold (very safe)
    thr = vol.mean() * 0.2
    init_mask = vol > thr

    # Largest connected component (outer boundary)
    lbl, n = ndi.label(init_mask) # label connected components, lbl=labels, n=number of components, by default 4-connectivity for 2D, 6-connectivity for 3D
    if n < 1: return torch.ones_like(x)  # fallback: no masking

    largest = (lbl == np.argmax(np.bincount(lbl.flat)[1:]) + 1) # boolean array [D, H, W], largest connected component

    # Fill interior holes (this is the fix!)
    filled = ndi.binary_fill_holes(largest) # fill holes in binary object, 6-connectivity for 3D
    # erode the mask → removes skull!
    # eroded = ndi.binary_erosion(filled, iterations=2)  # 1–2 voxels is ideal, shrinks the forefround region (1s) by one layer of voxels for each iteration

    mask = torch.from_numpy(filled).float().unsqueeze(0)
    return mask.to(x.device)

class ScaleIntensityRangePercentilesInsideHead(ScaleIntensityRangePercentiles):
    """ScaleIntensityRangePercentiles with the percentiles taken over the non-zero voxels only
    (= inside the head, once everything outside it has been zeroed), so they do not depend on how
    much empty box surrounds the head. Zeros stay 0 (clipped)."""

    def bounds(self, img):
        v = np.asarray(torch.as_tensor(img).detach().cpu()).ravel()
        v = v[v > 0] if (v > 0).any() else v
        lo, hi = np.percentile(v, [self.lower, self.upper])
        return float(lo), float(hi)

    def __call__(self, img):
        lo, hi = self.bounds(img)
        if hi <= lo:
            return img * 0
        out = (img - lo) / (hi - lo) * (self.b_max - self.b_min) + self.b_min
        return out.clamp(self.b_min, self.b_max) if self.clip else out


def head_mask(x, frac: float = 0.1):
    """Like brain_outer_mask, but the threshold is a share of the 99th percentile instead of
    the volume mean. The mean depends on how much empty field of view surrounds the head (a big
    FOV -> low mean -> background noise passes the threshold and the "head" becomes the whole
    box); the 99th percentile sits inside the head for any FOV. Used by head_crop."""
    vol = x.squeeze().cpu().numpy()
    init_mask = vol > frac * np.percentile(vol, 99)
    lbl, n = ndi.label(init_mask)
    if n < 1: return torch.ones_like(x)
    largest = (lbl == np.argmax(np.bincount(lbl.flat)[1:]) + 1)
    filled = ndi.binary_fill_holes(largest)
    return torch.from_numpy(filled).float().unsqueeze(0).to(x.device)


def keep_top(x, height):
    """If the box is taller than `height` voxels along the last axis (RAS z = bottom -> top), keep
    the top `height` slices: a tall field of view goes down into the neck, so the cut should
    remove neck, never the top of the brain (a centred crop would cut both ends)."""
    z = x.shape[-1]
    if z <= height:
        return x
    return SpatialCrop(roi_start=(0, 0, z - height), roi_end=(x.shape[1], x.shape[2], z))(x)


BRAIN_MM = 140  # the top 140 mm of the head hold the whole brain; centring uses only this part


def centre_on_brain(x, size_xy, brain_vox):
    """Cut a size_xy window (left-right, back-front) centred on the head at brain level, i.e. the
    non-zero voxels in the top `brain_vox` slices. CropForeground's box can include shoulders that
    keep_top later cut away (wide, or lopsided when the subject lies tilted); centring on that box
    left an empty strip on one side and cut the brain on the other (seen in Geneva, 2026-10-22).
    Pads with zeros where the window goes past the scan. Keeps the affine (MONAI pad + crop)."""
    nz = (x[0] != 0).cpu().numpy()
    filled_z = np.flatnonzero(nz.any(axis=(0, 1)))
    if filled_z.size == 0:
        return x
    top = filled_z[-1]
    slab = nz[:, :, max(top - brain_vox + 1, 0):top + 1]
    sx, sy = size_xy
    x = BorderPad(spatial_border=[sx, sx, sy, sy, 0, 0])(x)  # room for the window on every side
    start = []
    for axis, size in ((0, sx), (1, sy)):
        idx = np.flatnonzero(slab.any(axis=tuple(a for a in (0, 1, 2) if a != axis)))
        centre = (idx[0] + idx[-1] + 1) / 2 + size  # + size: shift from the padding
        start.append(int(round(centre - size / 2)))
    return SpatialCrop(roi_start=(start[0], start[1], 0),
                       roi_end=(start[0] + sx, start[1] + sy, x.shape[3]))(x)


def get_transforms(target_shape=(128, 128, 128), 
                pct_lo: float = 1.0, 
                pct_hi: float = 99.0,
                crop_foreground: bool = True, 
                ras: bool = True, 
                interp: str = "trilinear", 
                out_range: tuple = (0.0, 1.0),
                smooth_sigma_vox: tuple | None = None, 
                apply_brain_mask: bool = True, 
                process: bool = True,
                head_crop: bool = False,
                crop_margin: int = 0,
                spacing_mm: float | None = None,
                pct_inside_head: bool = False) -> Compose:
    """
    PET-optimized preprocessing pipeline using MONAI.

    Parameters
    ----------
    target_shape : tuple of int
        Desired (D, H, W) volume size.
    pct_lo, pct_hi : float
        Percentiles for intensity scaling.
    crop_foreground : bool
        Whether to crop to nonzero foreground.
    ras : bool
        Reorient to RAS anatomical orientation.
    interp : str
        Interpolation mode for resizing (e.g., 'trilinear').
    out_range : tuple
        Output intensity range (min, max).
    smooth_sigma_vox: tuple
        mm-based smoothing → computed for *original* voxels)
    apply_brain_mask : bool
        Whether to apply a brain mask to zero out non-brain regions.
    head_crop : bool
        Zero everything outside the head (head_mask) on the native scan *before* CropForeground, so the
        crop box is the head and not the whole field of view (PET background noise is > 0, so
        the default crop keeps almost everything). Needs crop_foreground=True.
    crop_margin : int
        Voxels of margin CropForeground keeps around the box.
    spacing_mm : float or None
        None (default): stretch the crop box to target_shape (each axis by its own factor).
        A number: resample to cubic voxels of this size first, then pad/crop the box (centred) to
        target_shape instead of stretching, so proportions are true and 1 voxel = spacing_mm in
        every scan. target_shape must then be big enough for a head (112x128x96 at 2 mm = 224x256x192
        mm fits the real data; 96 wide was too narrow for some skulls); a head taller than the box loses its bottom (neck), not its top.
        Left-right and back-front the box is centred on the head's top 140 mm (centre_on_brain), not on shoulders.
    pct_inside_head : bool
        Take the scaling percentiles over the non-zero (inside-the-head) voxels only. Needs the
        head already zeroed outside (head_crop), otherwise every voxel counts.
    Returns
    -------
    monai.transforms.Compose
        A composed MONAI transform pipeline.
    """
    steps = [LoadImage(image_only=True),
             EnsureChannelFirst()] # adds a channel dimension: (D, H, W) → (C=1, D, H, W)
    if ras: steps.append(Orientation(axcodes="RAS", labels=None)) # Reorients the image to a standard anatomical orientation: Right–Anterior–Superior
    
    if smooth_sigma_vox is not None: # Smooth BEFORE crop/resize (improves SNR for bounding box & interpolation)
        steps.append(GaussianSmooth(sigma=smooth_sigma_vox))

    if process:
        if spacing_mm: steps.append(Spacing(pixdim=(spacing_mm,) * 3, mode="bilinear")) # same cubic voxel size for every scan
        if head_crop: steps.append(Lambda(lambda x: x * head_mask(x))) # true zeros outside the head, so the crop below finds the head
        if crop_foreground: steps.append(CropForeground(margin=crop_margin)) # Crop out huge empty regions
        if spacing_mm:
            steps.append(Lambda(lambda x: keep_top(x, target_shape[2]))) # too tall: cut the neck, not the top of the head
            # left-right / back-front: centre on the head at brain level, not on CropForeground's box (shoulders)
            steps.append(Lambda(lambda x: centre_on_brain(x, target_shape[:2], int(round(BRAIN_MM / spacing_mm)))))
            steps.append(ResizeWithPadOrCrop(spatial_size=target_shape)) # centre the box in a fixed grid, no stretching
        else: steps.append(Resize(spatial_size=target_shape, mode=interp)) # Resamples to the target size
        # Intensity normalization: maps voxel values between the 1st–99th percentile to [0,1] (clipping outliers)
        scale_cls = ScaleIntensityRangePercentilesInsideHead if pct_inside_head else ScaleIntensityRangePercentiles
        steps.append(scale_cls(lower=pct_lo, upper=pct_hi, b_min=float(out_range[0]), b_max=float(out_range[1]), clip=True))

        # ---- Force background = 0 (critical for model to focus on brain) ----
        if apply_brain_mask and not head_crop: # with head_crop the outside is already 0 (and stays 0 after clipping)
            steps.append(Lambda(lambda x: x * brain_outer_mask(x))) # apply brain mask by threshold
    
    return Compose(steps)


class PETDataset(Dataset):
    """
    Expects:
      - table columns: ["ID", "pet_path", "visual_read", <regression_target>, ...]
      - transforms: a MONAI Compose returning a (1, D, H, W) Tensor/MetaTensor (float-like)
    """
    def __init__(self, table: pd.DataFrame, transforms, targets, data_file=None,input_cl=None, extra_global_feats: str | None = None, augment: bool = False, dtype=torch.float32):
        self.table = table.reset_index(drop=True)
        self.transforms = transforms
        self.data_file = data_file
        self.targets = [t.strip() for t in targets.split(",") if t.strip()]
        self.regression_targets = [t for t in self.targets if t != "visual_read"]
        self.input_cl = input_cl
        self.extra_global_feats = ([f.strip() for f in extra_global_feats.split(",")]
                                   if extra_global_feats is not None else [])
        self.augment = augment
        self.dtype = dtype
        self._pt_batch_size = None
        self._pt_batch_has_channel = None
        self._pt_manifest = None
        self._cached_pt_batch_id = None
        self._cached_pt_batch = None

    def __len__(self):
        return len(self.table)

    def _load_pt_manifest(self, data_dir: Path):
        if self._pt_manifest is not None:
            return self._pt_manifest

        manifest_path = data_dir / "tau_pt_manifest.json"
        if manifest_path.exists():
            self._pt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            self._pt_manifest = {}
        return self._pt_manifest

    def _load_pt_batch(self, data_dir: Path, batch_id: int):
        if self._cached_pt_batch_id == batch_id and self._cached_pt_batch is not None:
            return self._cached_pt_batch

        batch_path = data_dir / f"tau_batch_{batch_id:03d}.pt"
        batch = torch.load(batch_path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(batch):
            raise ValueError(f"{batch_path} must contain a torch.Tensor.")

        self._cached_pt_batch_id = batch_id
        self._cached_pt_batch = batch
        return batch

    def __getitem__(self, idx):
        row = self.table.iloc[idx]

        if self.data_file is None and isinstance(self.transforms, Compose): # expect MAINAI Compose class
            # MONAI pipeline -> Tensor/MetaTensor with shape [C=1, D, H, W]
            path = row["pet_path"]
            x = self.transforms(path)
        elif torch.is_tensor(self.data_file): # torch tensor with shape  [S, D, H, W]
            fid = str(int(row["ID"]))
            x = self.data_file[fid]
            x = x.unsqueeze(0) # ➜ becomes [1, D, H, W]
        elif isinstance(self.data_file, (str, bytes, os.PathLike)):
            fid_int = int(row["ID"])
            fid = str(fid_int)
            data_dir = Path(self.data_file)
            pt_batch0 = data_dir / "tau_batch_000.pt"
            if pt_batch0.exists():
                manifest = self._load_pt_manifest(data_dir)
                if self._pt_batch_size is None:
                    batch0 = self._load_pt_batch(data_dir, 0)
                    self._pt_batch_size = batch0.shape[0]
                    self._pt_batch_has_channel = (batch0.ndim == 5)
                    manifest_batch_size = manifest.get("batch_size")
                    if manifest_batch_size is not None and int(manifest_batch_size) != int(self._pt_batch_size):
                        raise ValueError(
                            f"Manifest batch_size={manifest_batch_size} does not match tau_batch_000.pt shape[0]={self._pt_batch_size}."
                        )

                batch_id = fid_int // self._pt_batch_size
                idx = fid_int % self._pt_batch_size
                batch = self._load_pt_batch(data_dir, batch_id)
                x = batch[idx]
                if x.ndim == 3:
                    x = x.unsqueeze(0)
            elif isinstance(self.transforms, Compose):
                # Load the preprocessed image from the corresponding pickle file (assuming it's stored in batches of 100)
                with open(Path(self.data_file) / f"tau_batch_000.pkl", "rb") as f: data_0 = pickle.load(f)
                batch_id = int(fid) // len(data_0)
                idx = int(fid) % len(data_0)

                pkl_path = Path(self.data_file) / f"tau_batch_{batch_id:03d}.pkl"
                with open(pkl_path, "rb") as f: data = pickle.load(f)
                sample = data[idx]

                # for strange warped images, put all nan and <0 values to 0
                sample["image"] = np.nan_to_num(sample["image"], nan=0.0, posinf=0.0, neginf=0.0)
                sample["image"] = np.clip(sample["image"], 0, None)

                # Apply the same transforms except loading (e.g., cropping, resizing, smoothing) to the loaded image
                load_img = MetaTensor(torch.as_tensor(sample["image"]), meta=sample["meta"])
                t_no_load = Compose(self.transforms.transforms[1:])  # remove LoadImage
                x = t_no_load(load_img)
                '''
                print(fid, idx)
                print(
                    f"{sample['image'].__class__.__name__}: "
                    f"shape={tuple(sample['image'].shape)}, "
                    f"min={np.nanmin(sample['image']):.4f}, "
                    f"max={np.nanmax(sample['image']):.4f}, "
                    f"mean={np.nanmean(sample['image']):.4f}, "
                    f"nan={np.isnan(sample['image']).sum()/sample['image'].size:.6f}"
                )
                x = load_img
                for t in t_no_load.transforms:
                    x = t(x)

                    xt = x.as_tensor() if isinstance(x, MetaTensor) else x
                    xt = xt.detach().cpu()
                    if torch.isnan(xt).all().item():
                        print(t.__class__.__name__, 'all nan')
                    print(
                        f"{t.__class__.__name__}: "
                        f"shape={tuple(xt.shape)}, "
                        f"min={xt[~torch.isnan(xt)].min().item():.4f}, "
                        f"max={xt[~torch.isnan(xt)].max().item():.4f}, "
                        f"mean={xt[~torch.isnan(xt)].mean().item():.4f}, "
                        f"nan={torch.isnan(xt).sum()/xt.numel()}"
                    )
                '''
            else:
                path = row["pet_path"]
                path = Path(self.data_file) / "data" / "data_{}.pt".format(fid)
                x = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(x, MetaTensor): x = x.as_tensor()
        x = x.to(dtype=self.dtype)
        ndim = x.ndim
        
        # Lightweight augmentation: random left-right flip only. After RAS
        # reorientation, dim D is the Right-Left axis (flipping it mirrors the
        # brain, anatomically valid); dims H (Anterior-Posterior) and W
        # (Inferior-Superior) are NOT symmetric axes, so flipping them used to
        # produce front-to-back or upside-down brains. Only D is flipped now.
        if self.augment:
            r = torch.rand(1, device=x.device)
            if r[0] < 0.5: x = torch.flip(x, dims=[ndim-3])  # D (Right-Left)

        # Targets
        y_cls = torch.tensor([float('nan')], dtype=torch.float32)
        y_reg = torch.tensor([float('nan')], dtype=torch.float32)
        if 'visual_read' in self.targets:
            y_cls = torch.tensor([row["visual_read"]], dtype=torch.float32)
        if self.regression_targets:
            y_reg = torch.tensor([row[t] for t in self.regression_targets], dtype=torch.float32)

        # extra inputs
        extras = []
        # optional input CL
        if self.input_cl is not None and pd.notna(row[self.input_cl]):
            extras.append(torch.tensor([row[self.input_cl]], dtype=torch.float32))
        # global PET features
        if self.extra_global_feats:
            feats = self.global_feats_from_x(x)
            extras.append(feats)
        
        extra_input = torch.cat(extras) if extras else torch.tensor([float("nan")])

        return x, y_cls, y_reg, extra_input, int(row["ID"])
    
    def global_feats_from_x(self, x, hi_thr=0.7, eps=1e-6):
        """
        x: torch tensor [1, D, H, W] or [D,H,W], already normalized to [0,1]
        background is exactly 0.
        returns torch tensor [F]
        """
        if x.ndim == 4:  # [1,D,H,W]
            x = x[0]
        mask = x > 0
        v = x[mask]

        # after masking less than 10 voxels, return zero
        if v.numel() < 10: return torch.zeros(len(self.extra_global_feats), device=x.device)

        feats = []
        for name in self.extra_global_feats:
            if name == "p95":
                feats.append(torch.quantile(v, 0.95))
            elif name == "std":
                feats.append(v.std(unbiased=False))
            elif name == "frac_hi":
                feats.append((v > hi_thr).float().mean())
            else:
                raise ValueError(f"Unknown global feature: {name}")

        return torch.stack(feats)
