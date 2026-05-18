# src/data.py
import os
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
        ScaleIntensityRangePercentiles, Compose, CropForeground, 
        Lambda, Spacing, NormalizeIntensity)

from src.utils import seed_worker
# ------------------------------
# Master table
# ------------------------------
def build_master_table(input_path: str, preproce_method: str, targets: List[str], datasets: str, data_type: str, modality: str = "abeta") -> pd.DataFrame:
    """
    Build the table required by the training code, given the custom folder layout.
    Normal mode: discover images + join demographics.
    Cached mode: if preproce_method is empty/None OR cache files exist in cache_dir,
                 load demo.csv only (no disk scans), and return that table.
    In cached mode, adds 'ID' to preserve the order matching data.pt.
    """
    targets_list = [t.strip() for t in targets.split(",") if t.strip()]
    if len(targets_list) != 1 or targets_list[0] not in {"visual_read", "CL"}: # the code now only works for single head regression/classification
        raise ValueError("Use exactly one target: 'visual_read' for classification OR 'CL' for regression.")

    # Detect cached mode
    use_cache = (not preproce_method) or (str(preproce_method).strip() == "")
    if use_cache:
        csv = Path(input_path) / "demo.csv"
        if not csv.exists():
            dataset_name = datasets.replace(",", "-")
            csv = Path(input_path) / f"demo_{dataset_name}_{data_type}.csv"
        df = pd.read_csv(csv, index_col=0) # Must have 'ID' column from 0 to len(df)
        print(f"[cache] Loaded {csv} with {len(df)} rows (no filesystem scan).")
    else:
        pets = find_pet_files(input_path=input_path, dataset=datasets, modality=modality)
        if pets.empty:
            raise FileNotFoundError(f"No NIfTI files found under '{input_path}' with preproc suffix  '{preproce_method}' for dataset '{datasets}'.")
        else:
            print(f'Found {pets.shape[0]} scans')
            print(pets.columns, pets.head(3))

        labels = load_participants_labels(input_path, datasets=datasets)
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
    #df = df[~df[targets_list].isna().values].reset_index(drop=True)
    df = df.dropna(subset=targets_list).reset_index(drop=True)
    print(f'Found {df.shape[0]} scans with demographics for {targets}')

    return df


def find_pet_files(input_path: str,  datasets: str, modality="abeta") -> pd.DataFrame:
    """
    Find amyloid PET files in BIDS format:

      dataset/raw/sub-xxxx/ses-xxx/pet/*trc-{fbp,fbb,nav,pib,flut}*.nii*

    Returns:
      dataset, ID, ses, tracer, pet_path, imagefile
    """

    TRACERS = {"abeta": ["fbp", "fbb", "nav", "pib", "flut"],
               "tau": ["ftp", "mk6240", "pi"]}
    
    ipath = Path(input_path)
    datasets_list = [d.strip() for d in datasets.split(",") if d.strip()]
    rows = []

    allowed_tracers = TRACERS.get(modality, set())
    if allowed_tracers is None: raise ValueError(f"Unknown modality '{modality}'. Available: {list(TRACERS)}")
    
    pattern = "raw/sub-*/ses-*/pet/*trc-*.nii*" ### find raw images only

    for dataset in datasets_list:
        print(f"search {ipath / dataset / pattern}")

        for nii in (ipath / dataset).glob(pattern):
            try:
                sub_dir = nii.parents[2].name   # sub-xxxx
                ses_dir = nii.parents[1].name   # ses-xxx
                fname = nii.name.lower()

                sid = sub_dir.replace("sub-", "")
                ses = int(ses_dir.replace("ses-", ""))

                tracer = None
                for trc in allowed_tracers:
                    if f"trc-{trc}" in fname:
                        tracer = trc
                        break
                if tracer is None: continue

            except Exception:
                continue

            rows.append({"dataset": dataset, "ID": sid, "ses": ses,
                        "tracer": tracer, "pet_path": str(nii), "imagefile": nii.name})

    df = pd.DataFrame(rows, columns=["dataset", "ID", "ses", "tracer", "pet_path", "imagefile"])

    if df.empty: return df

    df = (df.sort_values(["dataset", "ID", "ses", "tracer", "pet_path"])
          .drop_duplicates(subset=["dataset", "ID", "ses", "tracer"], keep="first")
          .reset_index(drop=True))

    return df


def load_participants_labels(input_path: str, datasets: Optional[str] = None) -> pd.DataFrame: #cache: Optional[bool] = False
    """
    Load demographics.csv from input_path and return:
    ID, dataset, visual_read, CL, age, gender
    """
    csv = Path(input_path) / "demographics.csv"
    if not csv.exists(): # try alternative path
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        dataset_name = datasets.replace(",", "-")
        csv = Path(os.path.join(proj_path, "data")) / f"demographics_{dataset_name}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing {csv}. Provide columns: ID, dataset, visual_read, CL, age, gender, ...")
    df = pd.read_csv(csv, index_col=0)
    
    print(f'loaded participants from {csv}:', df.shape, df.columns, df.head(3))
    # Ensure required columns are present in the dataframe.
    required = {"ID", "dataset", "visual_read", "CL", "age", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    return df

# ------------------------------
# Transforms & Dataset
# ------------------------------
def get_train_val_loaders(train_df, val_df, args):
    # Detect cached mode
    use_cache = (not args.data_suffix) or (str(args.data_suffix).strip() == "")
    
    tfm = get_transforms(target_shape=tuple(args.image_shape), add_spacing=args.spacing, pixdim=args.pixdim,
                                     intensity_norm=args.intensity_norm, pct_lo=args.intensity_pct[0],
                                     pct_hi=args.intensity_pct[1])
    if use_cache:
        p = Path(args.input_path) / "data.pt"
        if p.exists():
            tfm = None
            data_file = torch.load(p, map_location="cpu", weights_only=True) # torch tensor with shape [S, D, H, W]
            print(f"Reconstructed loaders from data.pt with shape [S, D, H, W].")
        else:
            data_file = Path(args.input_path) / 'data' / args.data_type
            if any(Path(data_file).glob("*.pt")):
                tfm = None
                print(f"Reconstructed loaders from data_idx.pt for each scan.")
            else:
                print(f"Reconstructed imgs from tau_batch_x.pkl for batched images.")
    else:
        data_file = None

    dl_tr = get_loader(train_df, tfm, data_file, args, batch_size=args.batch_size, augment=True, shuffle=True, train_test='train')
    dl_va = get_loader(val_df, tfm, data_file, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False, train_test='test')
    
    return dl_tr, dl_va


def get_loader(df, tfm, data_file, args, batch_size, augment=False, shuffle=False, train_test='train'):
    g = torch.Generator()
    g.manual_seed(args.seed)

    SAMPLE_WEIGHT_LABELS = ["expert consensus w CL", "expert consensus", "expert", "non-expert"]
    sample_weight_map = dict(zip(SAMPLE_WEIGHT_LABELS, args.sample_weight_scheme))

    dataset = PETDataset(df, tfm, args.targets, data_file=data_file, input_cl=args.input_cl, sample_weights=args.sample_weights, 
                         sample_weight_map=sample_weight_map, extra_global_feats=args.extra_global_feats, augment=augment)

    if "dataset" in df.columns and train_test=='train': #### domain-balanced sampling
        print('------ Balanced sampling ------')
        # inverse-frequency weights
        counts = df["dataset"].value_counts().to_dict()
        weights = df["dataset"].map(lambda d: 1.0 / counts[d]).values
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
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

def get_transforms(target_shape=(128, 128, 128), pct_lo: float = 1.0, pct_hi: float = 99.0,
    crop_foreground: bool = True, ras: bool = True, interp: str = "trilinear", out_range: tuple = (0.0, 1.0),
    apply_brain_mask: bool = True, process: bool = True, #smooth_sigma_vox: tuple | None = None,
    add_spacing: bool = False, pixdim: tuple[float, float, float] = (2.0, 2.0, 2.0), # pixdim: Target voxel spacing in mm, e.g., (2.0, 2.0, 2.0).
    intensity_norm: str = "percentile_01",) -> Compose: # percentile_01, zscore, none
    """
    PET-optimized preprocessing pipeline using MONAI.
    """
    steps = [LoadImage(image_only=True),
             EnsureChannelFirst()] 
    if ras: steps.append(Orientation(axcodes="RAS", labels=None)) 
    
    if add_spacing: steps.append(Spacing(pixdim=pixdim, mode=interp))

    #if smooth_sigma_vox is not None: # Smooth BEFORE crop/resize (improves SNR for bounding box & interpolation) -- should do with freesurfer
    #    steps.append(GaussianSmooth(sigma=smooth_sigma_vox))

    if process:
        if crop_foreground: steps.append(CropForeground()) # Crop out huge empty regions
        steps.append(Resize(spatial_size=target_shape, mode=interp)) # Resamples to the target size
        if intensity_norm == "percentile_01":
            steps.append(ScaleIntensityRangePercentiles(lower=pct_lo, upper=pct_hi, b_min=float(out_range[0]), b_max=float(out_range[1]), clip=True)) # Intensity normalization: maps voxel values between the 1st–99th percentile to [0,1] (clipping outliers)
        elif intensity_norm == "zscore":
            steps.append(NormalizeIntensity(nonzero=True, channel_wise=False))
        elif intensity_norm == "none":
            pass

        # ---- Force background = 0 (critical for model to focus on brain) ----
        if apply_brain_mask:
            steps.append(Lambda(lambda x: x * brain_outer_mask(x))) # apply brain mask by threshold
    
    return Compose(steps)


class PETDataset(Dataset):
    """
    Expects:
      - table columns: ["ID", "pet_path", "visual_read", "CL", ...]
      - transforms: a MONAI Compose returning a (1, D, H, W) Tensor/MetaTensor (float-like)
    """
    def __init__(self, table: pd.DataFrame, transforms, targets, data_file=None, input_cl=None, sample_weights=None, sample_weight_map=None,
                 extra_global_feats: str | None = None, augment: bool = False, dtype=torch.float32):
        self.table = table.reset_index(drop=True)
        self.transforms = transforms
        self.data_file = data_file
        self.targets = [t.strip() for t in targets.split(",") if t.strip()]
        self.input_cl = input_cl
        self.sample_weights = sample_weights
        self.sample_weight_map = sample_weight_map or {}
        self.extra_global_feats = ([f.strip() for f in extra_global_feats.split(",")]
                                   if extra_global_feats is not None else [])
        self.augment = augment
        self.dtype = dtype
        self.index_col = "index"

        self.dataset_idx = {s: i for i, s in enumerate(sorted(self.table["dataset"].dropna().unique()))} if "dataset" in self.table.columns else {}
        if self.sample_weights is not None and self.sample_weights not in self.table.columns: raise ValueError(f"sample_weights column '{self.sample_weights}' not found.")
        
        if self.data_file is not None and self.index_col not in self.table.columns: raise ValueError(f"data index column '{self.index_col}' not found in dataframe.")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        fid = int(row[self.index_col]) if self.data_file is not None else None

        if self.data_file is None and isinstance(self.transforms, Compose): # expect MAINAI Compose class
            # MONAI pipeline -> Tensor/MetaTensor with shape [C=1, D, H, W]
            path = row["pet_path"]
            x = self.transforms(path)
        else:
            fid = int(row[self.index_col])

            if torch.is_tensor(self.data_file): # torch tensor with shape  [S, D, H, W]
                x = self.data_file[fid]
                x = x.unsqueeze(0) # ➜ becomes [1, D, H, W]
            elif isinstance(self.data_file, (str, bytes, os.PathLike)):
                if isinstance(self.transforms, Compose): 
                    # Load the preprocessed image from the corresponding pickle file (assuming it's stored in batches of 100)
                    with open(Path(self.data_file) / f"batch_000.pkl", "rb") as f: data_0 = pickle.load(f)
                    batch_id = int(fid) // len(data_0)
                    local_idx = int(fid) % len(data_0)

                    pkl_path = Path(self.data_file) / f"batch_{batch_id:03d}.pkl"
                    with open(pkl_path, "rb") as f: data = pickle.load(f)
                    
                    sample = data[local_idx]

                    # for strange warped images, put all nan and <0 values to 0
                    sample["image"] = np.nan_to_num(sample["image"], nan=0.0, posinf=0.0, neginf=0.0)
                    sample["image"] = np.clip(sample["image"], 0, None)

                    # Apply the same transforms except loading (e.g., cropping, resizing, smoothing) to the loaded image
                    load_img = MetaTensor(torch.as_tensor(sample["image"]), meta=sample["meta"])
                    t_no_load = Compose(self.transforms.transforms[1:])  # remove LoadImage
                    x = t_no_load(load_img)
                else:
                    path = Path(self.data_file) / "data_{fid}.pt".format(fid)
                    x = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(x, MetaTensor): x = x.as_tensor()
        x = x.to(dtype=self.dtype)
        ndim = x.ndim
        
        # Lightweight augmentation: random flips along spatial dims (D, H, W)
        if self.augment:
            r = torch.rand(3, device=x.device)  # one draw per spatial dim
            if r[0] < 0.5: x = torch.flip(x, dims=[ndim-3])  # D
            if r[1] < 0.5: x = torch.flip(x, dims=[ndim-2])  # H
            if r[2] < 0.5: x = torch.flip(x, dims=[ndim-1])  # W

        # Targets
        y_cls, y_reg = torch.tensor([float('nan')]), torch.tensor([float('nan')])
        if 'visual_read' in self.targets:
            y_cls = torch.tensor([row["visual_read"]], dtype=torch.float32)
        if 'CL' in self.targets:
            y_reg = torch.tensor([row["CL"]], dtype=torch.float32)

        # dataset
        dataset_target = torch.tensor(self.dataset_idx[row["dataset"]], dtype=torch.long) if "dataset" in self.table.columns and pd.notna(row["dataset"]) else torch.tensor(-1, dtype=torch.long)

        # sample weight
        sample_weights = torch.tensor(self.get_sample_weight(row), dtype=torch.float32)

        # extra inputs
        extras = []
        # optional input CL
        if self.input_cl is not None:
            val = float(row[self.input_cl]) if pd.notna(row[self.input_cl]) else 0.0
            extras.append(torch.tensor([val], dtype=torch.float32))
        # global PET features
        if self.extra_global_feats:
            feats = self.global_feats_from_x(x)
            extras.append(feats)
        
        extra_input = torch.cat(extras) if extras else torch.tensor([float("nan")], dtype=torch.float32)

        return x, y_cls, y_reg, dataset_target, sample_weights, extra_input, int(row["ID"])
    
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
    
    def get_sample_weight(self, row):
        if self.sample_weights is None or pd.isna(row[self.sample_weights]):
            return 1.0
        v = row[self.sample_weights]
        if not isinstance(v, str):
            return float(v)
        key = v.strip()
        if key not in self.sample_weight_map:
            raise ValueError(f"Unknown sample weight category: {key}")
        return float(self.sample_weight_map[key])
