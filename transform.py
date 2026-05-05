import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from monai.data import MetaTensor

from src.data import build_master_table, get_transforms
from src.cv import get_stratify_labels
from src.utils import set_seed
from others.utils import compare_preproc

def main(args):
    out_dir = args.input_path + f'/transformed_{args.dataset}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output will be saved to {out_dir}")

    set_seed(args.seed, deterministic=True)
    df = build_master_table(args.input_path, args.data_suffix, args.targets, args.dataset)
    #if not 'all' in args.targets: 
    #    df_clean, _ = get_stratify_labels(df, args.stratifycvby)
    #else:
    df_clean = df

    # 1) Anonymize + shuffle the dataframe ONCE (before DataLoader)
    df_shuf = df_clean.sample(frac=1.0, random_state=args.seed).reset_index(drop=True).drop(columns=["imagefile","raw","preproc","state"], errors="ignore")
    start = args.start_ind or 0
    df_shuf.index = range(start, start + len(df_shuf))

    ids = df_shuf['ID']
    print(df_shuf)
    df_shuf.to_csv(out_dir + '/demo_with_origID-path.csv', index=False)
    np.save(f'{out_dir}/original_ids_shuffled.npy', ids.to_numpy())
    df_shuf['ID'] = df_shuf.index
    print(df_shuf)

    tfm = get_transforms(tuple(args.image_shape))

    xs = []
    os.makedirs(out_dir + '/compare_preproc', exist_ok=True)
    os.makedirs(out_dir + '/data', exist_ok=True)
    for idx in df_shuf['ID']:
        orig_idx = ids.iloc[idx-start]
        print(idx, orig_idx, end='\t')
        
        row = df_shuf[df_shuf.ID==idx]
        path = row["pet_path"]
        x = tfm(path)
        if isinstance(x, MetaTensor): x = x.as_tensor()
        xs.append(x.squeeze(0))
        torch.save(x, out_dir + f'/data/data_{idx}.pt')

        # compare preprocessing steps
        compare_preproc(tfm, path, f'{out_dir}/compare_preproc/tensorID{idx}_origID{orig_idx}',verbose=False)

    df_shuf['pet_path'] = 0
    print(df_shuf)
    df_shuf.to_csv(out_dir + '/demo.csv', index=False)

    print("\nAll done!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform data to input (torch + .csv)")

    parser.add_argument("--input_path", type=str, default='/Volumes/DeMON-SSD/PET/BIDS', help='images save in BIDS format. If not input, will set as <proj_path>/data. /Volumes/DeMON-SSD/PET/IDEAS/ /Volumes/DeMON-SSD/PET/BIDS')
    parser.add_argument("--dataset", type=str, default='ADNI', help='dataset')
    parser.add_argument("--data_suffix", type=str, default='_Inten_Norm', help='images finding pattern **/*<suffix>/*/*/*.nii* for find_pet_images function, specifically to IDEAS data')
    parser.add_argument("--targets", type=str, default="all", help="Predict variables name, corresponds to column names in demographics.csv, seperate by ,")
    #parser.add_argument("--stratifycvby", default="visual_read", help=",site,visual_read, List of column names to stratify by (e.g., visual_read CL age gender).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_shape", nargs=3, type=int, default=[128,128,128])
    parser.add_argument("--start_ind", type=int, default=20000)

    args, unknown = parser.parse_known_args()

    print(args)

    main(args)
