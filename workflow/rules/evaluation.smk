# Zero-shot evaluation of the final model on each external set (no weight updates), as in
# submit_val_AVID_unseen_visual-read.sh.


def external_demographics(wildcards):
    """demo_<set>.csv; the left-out cohort's file is made by split_demographics."""
    if EXCLUDE and wildcards.ext == f"{EXCLUDE}_unseen":
        return rules.split_demographics.output.external
    return f"{DATA}/demo_{wildcards.ext}.csv"


rule evaluate_external:
    input:
        model=predictions("test"),
        demo=external_demographics,
    output:
        f"{RUN}/validation/External_validation_{{ext}}__visual_read_zeroshot_results.csv",
    wildcard_constraints:
        ext="|".join(EXTERNALS),
    log:
        f"{RESULTS}/workflow_logs/{NAME}-val-{{ext}}.log",
    resources:
        runtime=60,
    params:
        model_kwargs=MODEL_KWARGS,
    shell:
        f"python run_val.py --dataset {{wildcards.ext}} --best_model_folder {RUN} --model CNN3D "
        f"--data_type {config['data_type']} --input_path {DATA} --targets visual_read --cls_loss softmax --few_shot 0 "
        "--model_kwargs '{params.model_kwargs}' 2>&1 | tee {log}; "
        f"python results_plotting/plot_visual_read_results.py {RUN} --dataset {{wildcards.ext}} "
        "|| echo 'figures failed (results are fine)'"
