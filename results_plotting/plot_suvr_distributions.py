#!/usr/bin/env python3

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
        description="Create Universal SUVR distribution plots from demo.csv and AVID demo files."
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
        "--out_dir",
        default=None,
        help="Output directory for the distribution panels (defaults to <proj_path>/results/suvr_distributions).",
    )
    return parser.parse_args()


def resolve_path(path_value: str | None, fallback_fn) -> Path | None:
    if path_value:
        return Path(path_value).expanduser().resolve()
    return fallback_fn()


def main():
    args = parse_args()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    script_path = Path(__file__).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        # <proj_path>/results, matching where run.py/run_val.py write everything else —
        # never inside the git repo itself.
        out_dir = script_path.parents[2] / "results" / "suvr_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    region_targets = ["Universal", "MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]

    demo_path = resolve_path(args.demo_csv, mod.find_demo_csv)
    if demo_path is not None:
        mod.make_suvr_distribution_panel(
            demo_path,
            out_dir,
            ["Universal"],
            title="Ground truth SUVR distribution for discovery dataset",
            filename="suvr_distribution_universal_demo.png",
        )
        mod.make_suvr_distribution_panel(
            demo_path,
            out_dir,
            region_targets,
            title="Ground truth regional SUVR distributions: discovery dataset",
            filename="suvr_distribution_regions_demo.png",
        )
    else:
        print("[WARNING] Could not find demo.csv for SUVR distribution panel.")

    demo_unseen_path = resolve_path(args.demo_unseen_csv, find_avid_demo_csv)
    if demo_unseen_path is not None:
        mod.make_suvr_distribution_panel(
            demo_unseen_path,
            out_dir,
            ["Universal"],
            title="Ground truth SUVR distribution for AVID external test set",
            filename="suvr_distribution_universal_avid.png",
        )
        mod.make_suvr_distribution_panel(
            demo_unseen_path,
            out_dir,
            region_targets,
            title="Ground truth regional SUVR distributions: AVID external test set",
            filename="suvr_distribution_regions_avid.png",
        )
    else:
        print("[WARNING] Could not find demo_AVID_unseen.csv for SUVR distribution panel.")

    print(f"[done] SUVR distribution panels saved to {out_dir}")


if __name__ == "__main__":
    main()
