from src.warnings import ignore_warnings
ignore_warnings()

import os
import torch
import pandas as pd

from src.params import parse_arguments
from src.utils import get_device, set_seed
from src.model_factory import build_model_from_args, classifier_out_dim
from src.data import get_train_val_loaders
from src.vis import run_visualization, benchmark_occlusion
from src.validation import load_validation_data

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")

def main(args):
    if 'AVID' in args.dataset:
        # External set (AVID_unseen): the same scan table run_val.py evaluates on
        _, _, df, _ = load_validation_data(args)
    else:
        df_train = pd.read_csv(args.best_model_folder + '/splits/train_subjects.csv')
        df_test = pd.read_csv(args.best_model_folder + '/splits/test_subjects.csv')
        df = pd.concat([df_train, df_test], ignore_index=True)
    if args.vis_img_list:
        subject_id_list = [int(t.strip()) for t in args.vis_img_list.split(',') if t.strip()]
        df_select = df[df['ID'].isin(subject_id_list)].reset_index(drop=True)
        missing = sorted(set(subject_id_list) - set(df_select['ID']))
        if missing:
            raise SystemExit(f"IDs not found in the {args.dataset} scan table: {missing} "
                             f"(AVID IDs need --dataset AVID_unseen; Gothenburg IDs must be in the run's train/test splits)")
    else:
        df_select = df
    print('df_select:',df_select)
    _, dl_va = get_train_val_loaders(df_select, df_select, args, repeat_train=False)

    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    if 'visual_read' in targets_list:
        n_unique = int(df["visual_read"].dropna().nunique())
        n_classes = classifier_out_dim(n_unique, args.cls_loss)
    else:
        n_classes = None
    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    ckpt_dir = os.path.join(args.best_model_folder, 'final_model', 'checkpoints')
    ckpt_last = os.path.join(ckpt_dir, 'final_model_last.pt')
    ckpt_best = os.path.join(ckpt_dir, 'final_model_best.pt')
    ckpt = ckpt_last if os.path.exists(ckpt_last) else ckpt_best
    sd = torch.load(ckpt, map_location=args.device, weights_only=True)
    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # before anything runs the model: in train mode, dropout and batch-of-1 BatchNorm give wrong probabilities

    if args.vis_benchmark:
        x, _, _, extra, pid = next(iter(dl_va))
        print(f"[benchmark] subject {pid[0]}")
        benchmark_occlusion(model, x[:1].to(args.device), extra[:1].to(args.device) if extra is not None else None,
                            args.device, mask_size=args.vis_mask_size, overlap=args.vis_overlap, n_batch=args.vis_n_batch)
        return

    vis_kwargs = {}
    if args.visualization_name == "occlusion":
        vis_kwargs = dict(mask_size=args.vis_mask_size, overlap=args.vis_overlap, n_batch=args.vis_n_batch,
                          mode=args.vis_occlusion_mode, brain_bbox=args.vis_brain_bbox,
                          target=args.vis_occlusion_target)
    run_visualization(model, dl_va, args.device, args.output_path, vis_name=args.visualization_name,
                      vis_kwargs=vis_kwargs, vis_norm=None, color_max=args.vis_color_max,
                      show_prob='visual_read' in targets_list)

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
