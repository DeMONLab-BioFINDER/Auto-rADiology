"""Region masks on the model's MNI grid: the Desikan-Killiany (DK) cortex outline (which voxels count as cortex)
and the published CenTauR tau masks (the regions every result is reported in).

Atlas: FreeSurfer aparc+aseg run on the MNI152NLin2009cAsym template itself (Lotter LD 2025, G-Node,
doi:10.12751/g-node.2mnxpm), the same file plot_fda_regions.py uses, downloaded on first use. Same template
as the MNI tau images, so no extra registration.

Cortex = the 68 DK cortical labels (ctx-lh-* / ctx-rh-*). The ribbon is thin at 1 mm, so on a coarser grid a
voxel counts as cortex when at least half of it is cortex (linear resampling of the 0/1 mask). The template
brain is larger than an average brain, so the cortex is about 1,060 ml (brain mask about 1,880 ml).
"""
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy.ndimage import distance_transform_edt

from plot_fda_regions import DK_FILE, DK_TSV, fetch

DEFAULT_ATLAS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "atlases" / "lotter2025_mni_freesurfer"


# CenTauR masks (Villemagne 2023), the same masks as the SUVRs in the demographics csv. They overlap (meta-temporal
# includes mesial temporal), so shares over them do not add up to 1.
CENTAUR_FILES = {"Meta-temporal": "Meta_CenTauR.nii.gz", "Temporoparietal": "TP_CenTauR.nii.gz",
                 "Mesial temporal": "Mesial_CenTauR.nii.gz", "Frontal": "Frontal_CenTauR.nii.gz"}


def centaur_masks(shape, affine, centaur_dir) -> dict:
    """{region name: boolean mask} on the given grid (nearest-neighbour; the files may be 4D (x, y, z, 1))."""
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    out = {}
    for name, fname in CENTAUR_FILES.items():
        img = nib.load(Path(centaur_dir) / fname)
        img = nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj)), img.affine)
        out[name] = image.resample_to_img(img, target, interpolation="nearest", force_resample=True,
                                          copy_header=True).get_fdata() > 0
    return out


# Choroid plexus stand-in. The template aseg has no choroid plexus label (FreeSurfer 31/63), so: the lateral ventricle
# (4/43) + its temporal horn (5/44), from the foramen of Monro backwards (MNI y < 0; the front horns have no choroid
# plexus), grown by 2 mm but only into white matter / ventricle, never into hippocampus, amygdala, cortex or deep grey.
# At PET resolution the plexus signal (and its spill-over) fills this space; an approximation, stated as such.
CHP_VENTRICLE = [4, 5, 43, 44]
CHP_GROW_INTO = [0, 2, 41, 4, 5, 43, 44, 31, 63]  # unlabelled, cerebral white matter, ventricles, choroid plexus
CHP_GROW_MM = 2.0
CHP_MAX_Y_MM = 0.0


def choroid_plexus_frac(shape, affine, atlas_dir=None) -> np.ndarray:
    """Fraction (0..1) of each voxel of the given grid inside the choroid plexus stand-in (see above)."""
    from plot_fda_regions import FS_ASEG
    atlas_dir = Path(atlas_dir) if atlas_dir else DEFAULT_ATLAS_DIR
    if not (atlas_dir / FS_ASEG).exists():
        fetch(atlas_dir)
    fs = nib.load(atlas_dir / FS_ASEG)
    K = np.asarray(fs.dataobj).astype(int)
    vox = np.sqrt((fs.affine[:3, :3] ** 2).sum(0))
    vent = np.isin(K, CHP_VENTRICLE)
    near = distance_transform_edt(~vent, sampling=vox) <= CHP_GROW_MM
    m = vent | (near & np.isin(K, CHP_GROW_INTO))
    ijk = np.argwhere(m)
    y = (fs.affine @ np.c_[ijk, np.ones(len(ijk))].T)[1]
    m[tuple(ijk[y >= CHP_MAX_Y_MM].T)] = False
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return image.resample_to_img(nib.Nifti1Image(m.astype(np.float32), fs.affine), target, interpolation="linear",
                                     force_resample=True, copy_header=True).get_fdata()


def pretty(name: str) -> str:
    """ctx-lh-superiortemporal -> Left superiortemporal."""
    _, hemi, region = name.split("-", 2)
    return f"{'Left' if hemi == 'lh' else 'Right'} {region}"


class DKCortex:
    """DK cortex on a given grid: .mask (True = cortex), .labels (DK label number of each cortex voxel, 0
    elsewhere), .names {label number: 'Left superiortemporal'}."""

    def __init__(self, shape, affine, atlas_dir=None, min_frac: float = 0.5):
        atlas_dir = Path(atlas_dir) if atlas_dir else DEFAULT_ATLAS_DIR
        if not (atlas_dir / DK_FILE).exists():
            fetch(atlas_dir)
        dk = nib.load(atlas_dir / DK_FILE)
        K = np.asarray(dk.dataobj).astype(int)
        names = pd.read_csv(atlas_dir / DK_TSV, sep="\t", header=None, index_col=0)[1]
        ctx_ids = [int(i) for i, n in names.items() if str(n).startswith("ctx-")]
        self.names = {i: pretty(names[i]) for i in ctx_ids}

        target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
        resample = lambda data, interp: image.resample_to_img(
            nib.Nifti1Image(data, dk.affine), target, interpolation=interp, force_resample=True,
            copy_header=True).get_fdata()
        with warnings.catch_warnings():  # "resampling a binary image linearly": intended (fraction of cortex)
            warnings.simplefilter("ignore", UserWarning)
            frac = resample(np.isin(K, ctx_ids).astype(np.float32), "linear")
        self.mask = frac >= min_frac
        # label of each cortex voxel: the nearest DK cortical label (nearest-neighbour resampling misses some
        # of the voxels the fraction rule keeps)
        lab = resample(np.where(np.isin(K, ctx_ids), K, 0).astype(np.int16), "nearest").astype(int)
        _, idx = distance_transform_edt(lab == 0, return_indices=True)
        self.labels = np.where(self.mask, lab[tuple(idx)], 0)

    def name_at(self, ijk) -> str:
        return self.names.get(int(self.labels[tuple(ijk)]), "outside cortex")
