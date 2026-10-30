# torchpalmm-experiments

Numerical experiments for [`torchpalmm`](https://github.com/rrrrn/torchpalmm), a PyTorch package for REML estimation of linear mixed models with additive smoothing-spline components.

## Structure

```text
simulation_experiments/   simulation studies (accuracy, scaling, GPU vs CPU) and their data generators
analysis/                 real-data analysis of periodontal therapy data
slurm/                    cluster submission scripts for the simulations
```

## Simulation experiments

`simulation_experiments/` compares `torchpalmm` with `mgcv` (`gamm` and `bam`) as the number of random effects and of smooth terms grows, and measures how it scales with and without a GPU. Each experiment folder holds its generating, fitting and analysis scripts together with the results, figures and tables reported in the paper. See [`SIMULATION.md`](SIMULATION.md) for the design.

## Real-data analysis

`analysis/` applies `torchpalmm` to a longitudinal periodontal therapy study from [PhysioNet](https://physionet.org/content/supportive-periodontal-therapy/1.0.0/). The data is third-party and not included here; see [`analysis/README.md`](analysis/README.md) for where to put it and for the workflow.

## Setup

```bash
pip install -r requirements.txt
```

This installs the Python dependencies and `torchpalmm` from GitHub. The R baselines need R with `mgcv`. The SLURM scripts read your account from `SLURM_ACCOUNT`.
