# Real-data analysis: bleeding on probing during supportive periodontal therapy

This folder applies `torchpalmm` to the [Supportive Periodontal Therapy Clinical Examination Data](https://physionet.org/content/supportive-periodontal-therapy/1.0.0/) from PhysioNet (883 patients, about 11,800 visits). The outcome is the percentage of bleeding sites at each supportive periodontal therapy visit.

Run everything from `analysis/`.

## Workflow

1. **Data.** Download the PhysioNet data into `spt_data/`. It is third-party data and is ignored by git.
2. **Curation.** `curate_spt_bleeding.py` turns the downloaded files into one table with a row per visit: the outcome, the history from each patient's previous visit, follow-up time, and baseline patient covariates.
3. **Sample description.** `table1_spt.R` writes the patient and visit characteristics table (`table1_spt.tex`).
4. **Model.** `fit.ipynb` fits an additive mixed model with `torchpalmm`: a smooth follow-up curve for each baseline disease class and sex group, linear fixed effects, and patient-level random intercepts and slopes. It reports estimates, curves, tests and diagnostics.
5. **Benchmark.** `benchmark.ipynb` measures the peak memory and run time of the fit.

`profile_mgcv.R` and the `mgcv` and sleep-duration scripts in `benchmark/` belong to a separate analysis whose data is not in this repo.
