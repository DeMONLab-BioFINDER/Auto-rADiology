# Voxelwise permutation statistics on SUVR (per set), and every other analysis + figure
# (sbatch_scripts/run_visual-read-analysis.sh, test set and each external set separately).


def set_demographics(s):
    if s == "test":
        return DEMO_POOL
    return rules.split_demographics.output.external if s == f"{EXCLUDE}_unseen" else f"{DATA}/demo_{s}.csv"


def set_of(lab):
    """test / oasis / avid -> the set's full name (test, OASIS_unseen, AVID_unseen)."""
    return next(s for s in SETS if label(s) == lab)


def voxelwise_set_args(s):
    if s == "test":
        return "--set test"
    return f"--set avid --ext_dataset {s} --ext_demo_csv {set_demographics(s)}"  # "avid" = the external slot


rule voxelwise_stats:
    """GLMs at every cortical voxel (model output beyond the read, missed vs detected, ...), TFCE, FWE-corrected;
    peaks named with the Desikan-Killiany atlas."""
    input:
        lambda wc: predictions(set_of(wc.label)),
    output:
        f"{RUN}/figures/voxelwise_stats_{{label}}/summary.csv",
    wildcard_constraints:
        label="|".join(label(s) for s in SETS),
    params:
        set_args=lambda wc: voxelwise_set_args(set_of(wc.label)),
    log:
        f"{RESULTS}/workflow_logs/{NAME}-voxelwise-{{label}}.log",
    resources:
        runtime=720,
        cpus_per_task=16,   # Snakemake asks SLURM for 1 CPU unless told; the permutations need many
    shell:
        f"python results_plotting/voxelwise_stats_visual_read.py {RUN} {{params.set_args}} --demo_csv {DEMO_POOL} "
        f"--pkl_dir {config['pkl_dir']} --ref_mask {DATA}/masks/voi_CerebGry_tau_2mm.nii "
        f"--atlas_dir {ATLAS} --n_perm {config['voxelwise_permutations']} "
        "--n_jobs {resources.cpus_per_task} --out_dir $(dirname {output}) 2>&1 | tee {log}"


rule analysis:
    """Performance, confidence, discordant cases, subgroups, regressions, region rule, reader votes,
    asymmetry, Figure 1: on the test set and on each external set (never pooled)."""
    input:
        [predictions(s) for s in SETS],
        DEMO_POOL,
    output:
        touch(f"{FLAGS}/analysis.done"),
    log:
        f"{RESULTS}/workflow_logs/{NAME}-analysis.log",
    resources:
        runtime=120,
    shell:
        f"DEMO={DEMO_POOL} DATA={DATA} EXTS='{' '.join(EXTERNALS)}' "
        f"bash sbatch_scripts/run_visual-read-analysis.sh {RUN} 2>&1 | tee {{log}}"
