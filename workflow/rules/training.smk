# Training: leave the cohort out, 5-fold CV for the epoch count, then the final model.
# Same recipe as submit_visual-read-cv5.sh / submit_visual-read-final.sh (fixed learning rate; the final model
# trains for the CV's median best epoch on the whole 80% training pool, then is tested once on the 20%).

TRAIN_ARGS = (
    f"--no-tune --dataset Gothenburg --data_type {config['data_type']} --input_path {DATA} "
    f"--targets visual_read --stratifycvby site,visual_read --train_size 0.80 --val_size 0.00 --test_size 0.20 "
    f"--cls_loss softmax" + (f" --exclude_sites {EXCLUDE}" if EXCLUDE else "")
)


# main split (exclude ""): nothing to split off, demo.csv is used as it is
if EXCLUDE:
  rule split_demographics:
      """demo.csv -> the training pool without the left-out cohort, and the cohort on its own (an external set)."""
      input:
          f"{DATA}/demo.csv",
      output:
          pool=DEMO_POOL,
          external=f"{DATA}/demo_{EXCLUDE}_unseen.csv",
      run:
          import pandas as pd
          demo = pd.read_csv(input[0])
          demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
          out = demo["site"].astype(str) == EXCLUDE
          if not out.any():
              raise ValueError(f"No scans from site {EXCLUDE!r}; sites: {sorted(demo['site'].astype(str).unique())}")
          demo[~out].to_csv(output.pool, index=False)
          demo[out].to_csv(output.external, index=False)
          print(f"{EXCLUDE}: {out.sum()} scans left out; {(~out).sum()} in the training pool + test set")


rule cv5:
    """5-fold CV on the training pool; writes the median best epoch for the final model."""
    input:
        DEMO_POOL,
    output:
        epochs=f"{CV_RUN}/median_best_epoch.txt",
    log:
        f"{RESULTS}/workflow_logs/{NAME}-cv5.log",
    resources:
        runtime=240,
    params:
        model_kwargs=MODEL_KWARGS,
    shell:
        f"python run.py {TRAIN_ARGS} --model_kwargs '{{params.model_kwargs}}' --run_kfold_cv --n_splits 5 --es_patience 15 --es_min_delta 0.001 "
        f"--no-lr_scheduler --run_name {NAME}-cv5 2>&1 | tee {{log}}; "
        "grep -o 'Median best_epoch across folds: [0-9]*' {log} | grep -o '[0-9]*$' > {output.epochs}; "
        "test -s {output.epochs}; "
        f"python results_plotting/plot_cv_summary.py {CV_RUN} || echo 'CV figures failed (results are fine)'"


rule final_model:
    """Final model for the CV's epoch count; evaluated once on the held-out test set."""
    input:
        epochs=rules.cv5.output.epochs,
    output:
        predictions("test"),
    log:
        f"{RESULTS}/workflow_logs/{NAME}-final.log",
    resources:
        runtime=240,
    params:
        model_kwargs=MODEL_KWARGS,
    shell:
        f"python run.py {TRAIN_ARGS} --model_kwargs '{{params.model_kwargs}}' --epochs $(cat {{input.epochs}}) --run_name {NAME}-final "
        "2>&1 | tee {log}; "
        f"python results_plotting/plot_visual_read_results.py {RUN} --dataset Gothenburg "
        "|| echo 'figures failed (results are fine)'"
