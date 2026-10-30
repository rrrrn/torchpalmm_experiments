# torchpalmm-experiments

Numerical experiments for [`torchpalmm`](https://github.com/rrrrn/torchpalmm), a PyTorch package for REML estimation of linear mixed models with additive smoothing-spline components.

## Structure

```text
SimScripts/                          data generators shared by the experiments
simulation_experiments/
  randeff_comparison/                random-effects scaling vs gamm and bam
  smoothspline_comparison/           smooth-term scaling vs gamm and bam, learning-rate diagnostic, curve library
  scaling_py/                        scaling in smooth terms (ss_scaling) and random effects (pz_scaling_ss)
  gpu_cpu_comparison/                GPU vs CPU run time and agreement
slurm/                               cluster submission scripts
```

Each experiment folder holds its `results/`, the figures (`plots/`) and tables (`tables/`) reported in the paper, and the scripts that generate data, fit models and regenerate the outputs (`python analyze_results.py`).

For the data-generating process, factor grids and cluster workflow, see [`SIMULATION.md`](SIMULATION.md).

## Data

The real-data analysis in `analysis/` uses the [Supportive Periodontal Therapy Clinical Examination Data](https://physionet.org/content/supportive-periodontal-therapy/1.0.0/) from PhysioNet. It is third-party data and is not included in this repo. Download it from PhysioNet into `analysis/spt_data/`, which git ignores.

`analysis/curate_spt_bleeding.py` turns the PhysioNet files into the long-format table used by the analysis. It reads `01_initial_periodontal_therapy_data.csv`, `02_supportive_periodontal_therapy_data.csv` and `03_supplementary_data.csv` from `analysis/spt_data/` and writes `curated_spt_bleeding.csv` (and a 445-patient `subsample_5years` subset) back to that folder:

```bash
python analysis/curate_spt_bleeding.py
```

- One row per supportive periodontal therapy visit. The outcome is bleeding on probing, as a percentage of four sites per tooth.
- The percentage of sites with probing depth of at least 5 mm and at least 6 mm is recomputed from the counts and checked against the published supplementary file.
- Visit-history predictors (previous probing-depth percentages, previous bleeding, cumulative follow-up years, days since the previous visit, number of previous visits) come from the patient's earlier visit. They are missing at visit 1.
- A patient whose visit 9 is missing has the gap flagged in `after_missing_visit`.
- Patient-level baseline covariates (sex, age, smoking, baseline diagnosis, baseline and post-therapy probing-depth percentages) are joined from the initial-therapy file.

## Setup

```bash
pip install -r requirements.txt
```

This installs the Python dependencies and `torchpalmm` from GitHub. With `uv`, run `uv sync` and then `uv pip install git+https://github.com/rrrrn/torchpalmm.git` instead.

The R baselines need R with `mgcv`.

The SLURM scripts read your account from the environment:

```bash
export SLURM_ACCOUNT=<your SLURM account>
```
