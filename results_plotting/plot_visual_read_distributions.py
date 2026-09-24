#!/usr/bin/env python3
"""Ground-truth visual_read class balance for the discovery dataset and the AVID
external test set - dataset-level, not tied to any one training/validation run.
Mirrors plot_suvr_distributions.py for the regional-SUVR model."""

import argparse
from pathlib import Path

import plot_utils as mod


def find_avid_demo_csv() -> Path | None:
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / "demo_AVID_unseen.csv",
        script_path.parents[2] / "data" / "demo_AVID_unseen.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create visual_read class-balance plots from demo.csv and AVID demo files."
    )
    parser.add_argument(
        "--demo_csv",
        default=None,
        help="Path to demo.csv (defaults to data/demo.csv if found).",
    )
    parser.add_argument(
        "--demo_unseen_csv",
        default=None,
        help="Path to demo_AVID_unseen.csv (defaults to data/demo_AVID_unseen.csv if found).",
    )
    parser.add_argument(
        "--group_col",
        default="site",
        help="Demographic column to break class balance down by.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for the distribution panels (defaults to <proj_path>/results/visual_read_distributions).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    script_path = Path(__file__).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        # <proj_path>/results, matching where run.py/run_val.py write everything else —
        # never inside the git repo itself.
        out_dir = script_path.parents[2] / "results" / "visual_read_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_path = mod.resolve_path(args.demo_csv, mod.find_demo_csv)
    if demo_path is not None:
        df_demo = mod.pd.read_csv(demo_path)
        mod.make_class_balance_panel(
            df_demo,
            out_dir,
            filename="visual_read_class_balance_demo.png",
            group_col=args.group_col,
            title="Visual read distribution: discovery dataset",
        )
    else:
        print("[WARNING] Could not find demo.csv for visual_read class balance panel.")

    demo_unseen_path = mod.resolve_path(args.demo_unseen_csv, find_avid_demo_csv)
    if demo_unseen_path is not None:
        df_demo_unseen = mod.pd.read_csv(demo_unseen_path)
        mod.make_class_balance_panel(
            df_demo_unseen,
            out_dir,
            filename="visual_read_class_balance_avid.png",
            group_col=args.group_col,
            title="Visual read distribution: AVID external test set",
        )
    else:
        print("[WARNING] Could not find demo_AVID_unseen.csv for visual_read class balance panel.")

    print(f"[done] visual_read distribution panels saved to {out_dir}")


if __name__ == "__main__":
    main()
