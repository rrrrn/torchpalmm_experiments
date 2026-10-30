# Simulation Setup

## Overview

This repo holds the numerical experiments for [`torchpalmm`](https://github.com/rrrrn/torchpalmm): REML estimation for linear mixed models with additive smoothing-spline components. The data generators live in `SimScripts/` and the experiments in `simulation_experiments/`.

---

## 1. Smoothing Spline LMM Generator (`SimScripts/LMMSS.py`)

### Model

$$y_i = X_i \beta + Z_i u_i + \sum_{d=1}^{S} f_d(t_{id}) + \varepsilon_i, \quad u_i \sim \mathcal{N}(0, D), \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2 I)$$

- $S$ = `spline_dimension` (sn): number of additive smooth components
- Subject $i$ has $n_i \in [\ell_{\min}, \ell_{\max}]$ observations; the experiments use $n_i = 15$
- `beta = [1, ..., px]`; $X$ is i.i.d. $\mathcal{N}(0,1)$; $Z$ is a column of ones plus `pz-1` i.i.d. $\mathcal{N}(0,1)$ columns

### True curves

$t_{id} \in [0, 1]$. Each component uses one of the ten curves in `MIXED_SS_CURVES` (for example $2\sin(\pi t)$, a Hermite step, Gaussian bumps, $\tanh$, $|t - 0.5|$), standardized to zero mean and unit variance so that changing sn changes the model dimension without changing the signal-to-noise ratio. `ss_curve_ids(repeat_id, sn)` picks which curves a replicate uses; the choice depends only on `repeat_id`, so it is constant across N, D type and sigma.

### Random-effects covariance `D`

| `rand_D` | Structure |
|----------|-----------|
| `"diagonal"` | diagonal with entries drawn from $\text{Uniform}(0.5, 2)$ |
| `"unstructured"` | $D = LL^\top$, $L \sim \mathcal{N}(0,1)^{pz \times pz}$ |

### Knots

`ss_knot_count(N, lmin, lmax)` returns $3.5\,n^{1/3}$ interior knots ($n$ = total observations), clamped to $[50, 300]$.

### Output (tuple)

```python
x, z, y, spline_vars, subject_id, beta, D, sigma2
```

`y` is mean-centered before it is saved by the experiments.

`SimScripts/LMMSS.py` also contains `generate_pure_ss_data`, a single-smooth generator with no fixed or random effects (scenarios `global` and `local`).

---

## 2. Random-Effects Generator (`SimScripts/RandeffLMM.py`)

`generate_randeff_data` simulates a single smooth ($f(t) = 2\sin(\pi t)$, sn = 1) with `pz` random effects and iid or compound-symmetry residuals. It reuses `MIXED_SS_CURVES` from `LMMSS.py`. The seed is derived from `(N, pz, repeat_id, lmin)`.

---

## 3. Experiments

Each experiment has its own `generate_data.py` (writes `data/`, which is git-ignored), `run_python.py` (our method), and an `analyze_results.py` that writes the figures and tables kept in the repo.

| Folder | Varies | Fixed | Baselines | Outputs |
|--------|--------|-------|-----------|---------|
| `simulation_experiments/randeff_comparison/` | $q$ = pz in {5, 8, 20}; N in {500, 1000}; D diagonal or unstructured | px = 2, sn = 1, 15 observations per subject, iid residuals | mgcv `gamm` and `bam` (`run_r_baselines.R`) | `plots/randeff_r_failure_diagnostic.png`, `tables/randeff_param_table.tex` |
| `simulation_experiments/smoothspline_comparison/` | $m$ = sn in {1, 5, 10, 15}; N in {500, 1000}; D diagonal or unstructured | px = 2, pz = 2, 15 observations per subject | mgcv `gamm` and `bam` (`run_r_baselines.R`) | `plots/mixed_curve_library.png`, `plots/mixed_r_failure_diagnostic.png`, `plots/mixed_lr_summary.png`, `tables/mixed_param_table.tex` |
| `simulation_experiments/scaling_py/` | `ss_scaling`: sn in {5, 10, 20, 30}; `pz_scaling_ss`: pz in {5, 10, 20, 30} | N = 1000, px = 2 | none (our method only) | `plots/scaling_task_boxplot_relerr.png` |
| `simulation_experiments/gpu_cpu_comparison/` | device (GPU vs CPU) at sn = 10 and 30 | the `ss_scaling` datasets | none | `plots/gpu_cpu_combined.png` |

All experiments use 100 replicates per cell and report the L-BFGS fit at lr = 1.0. `smoothspline_comparison/run_python.py` also fits lr in {1e-3, 1e-1, 1}; `analyze_lr.py` compares them.

The CPU rows in `gpu_cpu_comparison` are borrowed from `scaling_py/ss_scaling`, not re-run. The GPU fit reuses that pilot's `run_python.py` and datasets so only the device differs.

---

## 4. Cluster Workflow (SLURM)

Run from the project root. Each script first checks that `torchpalmm` is installed from GitHub (`slurm/check_torchpalmm.sh`).

```bash
bash slurm/run_array_randeff.sh       # randeff_comparison (--lang py | r | both)
bash slurm/run_array_ss.sh            # smoothspline_comparison (--lang py | r | both)
bash slurm/py_scaling_all.sh          # scaling_py: ss_scaling and pz_scaling_ss
bash slurm/run_gpu_cpu_m.sh           # gpu_cpu_comparison at sn = 10 and 30 (plus CPU control)
bash slurm/run_gpu_cpu_comparison.sh  # gpu_cpu_comparison at sn = 30 only
```

**Dependency chain:**

```
generate_data.py -> run_python.py (array: one task per cell)
                 -> run_r_baselines.R (array: same grid, parallel with the Python fits)
analysis (analyze_results.py) runs locally after the results are synced
```

**Cluster:** `standard` partition (CPU), `gpu` partition for the GPU comparison, with the SLURM account read from `SLURM_ACCOUNT`, which must be exported before submitting.
