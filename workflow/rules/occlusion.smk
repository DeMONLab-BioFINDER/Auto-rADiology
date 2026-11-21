# Occlusion maps for every scan of each set, split into parallel chunks (submit_vis-occlusion-all.sh does one
# chunk), then merged and summarised as group maps by outcome (Model +/- x Read +/-).

CHUNKS = range(config["occlusion_chunks"])


rule occlusion_chunk:
    """One chunk: every N-th scan of the set (scans already done are skipped, so a rerun only does the rest)."""
    input:
        lambda wc: predictions(wc.set),
    output:
        touch(f"{FLAGS}/occlusion_{{set}}_chunk{{chunk}}.done"),
    wildcard_constraints:
        set="|".join(SETS),
        chunk=r"\d+",
    log:
        f"{RESULTS}/workflow_logs/{NAME}-occlusion-{{set}}-{{chunk}}.log",
    resources:
        runtime=480,
    params:
        dataset=lambda wc: DATASET[wc.set],
    shell:
        f"SLURM_ARRAY_TASK_ID={{wildcards.chunk}} SLURM_ARRAY_TASK_COUNT={config['occlusion_chunks']} "
        f"POOL={config['pool']} DATA_TYPE={config['data_type']} DATASET={{params.dataset}} "
        f"bash sbatch_scripts/submit_vis-occlusion-all.sh {RUN} 2>&1 | tee {{log}}"


rule occlusion_group_maps:
    """Merge the chunks (the chunk folders are emptied, hence the flag files); group mean maps, FDA-region
    shares and their tests, in <occlusion folder>/group_maps/."""
    input:
        lambda wc: [f"{FLAGS}/occlusion_{wc.set}_chunk{k}.done" for k in CHUNKS],
    output:
        touch(f"{FLAGS}/group_maps_{{set}}.done"),
    wildcard_constraints:
        set="|".join(SETS),
    params:
        base=lambda wc: occlusion_dir(wc.set),
    resources:
        runtime=60,
    shell:
        "python results_plotting/merge_occlusion_chunks.py {params.base}; "
        f"python results_plotting/occlusion_group_maps.py {{params.base}} --regions {config['fda_regions']} "
        "--label {wildcards.set}"
