from src.warnings import ignore_warnings
ignore_warnings()

import os
import torch
import pandas as pd

from src.params import parse_arguments
from src.utils import get_device, set_seed
from src.model_factory import build_model_from_args
from src.data import get_train_val_loaders
from src.vis import run_visualization

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")

def main(args):
    df_train = pd.read_csv(args.best_model_folder + '/splits/train_subjects.csv')
    df_test = pd.read_csv(args.best_model_folder + '/splits/test_subjects.csv')
    df = pd.concat([df_train, df_test], ignore_index=True)
    if args.vis_img_list:
        subject_id_list = [int(t.strip()) for t in args.vis_img_list.split(',') if t.strip()]
        df_select = df[df['ID'].isin(subject_id_list)].reset_index(drop=True)
    else:
        df_select = df
    print('df_select:',df_select)
    _, dl_va = get_train_val_loaders(df_select, df_select, args, repeat_train=False)

    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None
    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    ckpt_dir = os.path.join(args.best_model_folder, 'final_model', 'checkpoints')
    ckpt_last = os.path.join(ckpt_dir, 'final_model_last.pt')
    ckpt_best = os.path.join(ckpt_dir, 'final_model_best.pt')
    ckpt = ckpt_last if os.path.exists(ckpt_last) else ckpt_best
    sd = torch.load(ckpt, map_location=args.device, weights_only=True)
    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)

    run_visualization(model, dl_va, args.device, args.output_path, vis_name=args.visualization_name, vis_norm=None)

    print('DONE!')

    return 


if __name__ == "__main__":
    args = parse_arguments()
    args.device = get_device()
    #args.device = get_device(force_cpu=True)
    print("Using device:", args.device)
    print(args)

    set_seed(args.seed)

    main(args)
