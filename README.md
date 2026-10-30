# torchpalmm-experiments

Numerical experiments for [`torchpalmm`](https://github.com/rrrrn/torchpalmm), a PyTorch package for REML estimation of linear mixed models with additive smoothing-spline components.

## Structure

```text
SimScripts/                          data generators shared by the experiments
experiments/
  randeff_comparison/                random-effects scaling vs gamm and bam
  smoothspline_comparison/           smooth-term scaling vs gamm and bam, learning-rate diagnostic, curve library
  scaling_py/                        scaling in smooth terms (ss_scaling) and random effects (pz_scaling_ss)
  gpu_cpu_comparison/                GPU vs CPU run time and agreement
slurm/                               cluster submission scripts
```

Each experiment folder holds its `results/`, the figures (`plots/`) and tables (`tables/`) reported in the paper, and the scripts that generate data, fit models and regenerate the outputs (`python analyze_results.py`).

For the data-generating process, factor grids and cluster workflow, see [`SIMULATION.md`](SIMULATION.md).

## Setup

```bash
uv sync
uv pip install git+https://github.com/rrrrn/torchpalmm.git
```

The R baselines need R with `mgcv`.

The SLURM scripts read your account from the environment:

```bash
export SLURM_ACCOUNT=<your SLURM account>
```
