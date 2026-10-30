"""torchpalmm cell-means model for the memory / runtime benchmark: reads the analysis sample, builds the model matrices and fits (REML, L-BFGS).
usage: python benchmark_torchpalmm.py MODE CSV OUT.csv     MODE = baseline (imports and data only) | fit (matrices + REML fit) | summary (fit, then the library's inference summary: SEs, p-values, curve tests)"""
import sys, time
import numpy as np, pandas as pd, torch
import torchpalmm as tp

mode, csv, out_csv = sys.argv[1:4]
df = pd.read_csv(csv).sort_values(["Subject", "wear_day"]).reset_index(drop=True)
if not pd.api.types.is_numeric_dtype(df["Gender"]):
    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})                                       # female = 1
if mode == "baseline":
    print(f"RESULT mode=baseline rows={len(df)}"); sys.exit()

covs = ["ALR1_00_19", "ALR1_19_24", "ALR2_00_19", "ALR2_19_24"]                               # exposure x interval
x_cols = ["Age_scaled", "Gender", "WAZ", "HAZ", "bps_scaled", "weekend_night"]
gen, wk = df["Gender"].to_numpy(float), df["weekend_night"].to_numpy(float)
X = np.column_stack([df[x_cols].to_numpy(float), gen * wk])                                  # + female x weekend night
x_names = x_cols + ["Female x weekend night"]
Z = np.ones((len(df), 1))                                                                    # random intercept by subject
y = torch.as_tensor(df["SleepDur_FINAL"].to_numpy(float)).view(-1, 1)
obs_id = pd.factorize(df["Subject"])[0]
strata = [("female weekday", 1, 0), ("female weekend", 1, 1), ("male weekday", 0, 0), ("male weekend", 0, 1)]
spline_inputs = np.column_stack([df[covs].to_numpy(float)] * len(strata))                    # one curve per group and exposure x interval (cell-means)
smooth_mask = np.column_stack([np.repeat(((gen == g) & (wk == w))[:, None].astype(float), len(covs), axis=1) for _, g, w in strata])
names = [f"{c}, {lab}" for lab, _, _ in strata for c in covs]

t0 = time.perf_counter()
ss = tp.SmoothingSplineMixed(y=y, X=X, Z=Z, spline_inputs=spline_inputs, obs_id=obs_id, smooth_mask=smooth_mask, corr_structure="ar1", rand_D_type="unstructured")
total_basis = sum(b.shape[2] for b in ss.NB_list)
torch.manual_seed(0)
ss.fit(optimizer=torch.optim.LBFGS, lr=1, max_iter=10000, fit_type="reml", convg_check="all", grad_control_tol=1e-4, tol=1e-9, n_restarts=1)
secs = time.perf_counter() - t0
beta = ss.beta.detach().cpu().numpy().ravel()[:X.shape[1]]                                  # fixed effects, in the order of the columns of X
sum_secs = 0.0
if mode == "summary":
    t1 = time.perf_counter(); res = ss.summary(names=x_names, spline_names=names); sum_secs = time.perf_counter() - t1
pd.DataFrame({"term": x_names, "estimate": beta}).to_csv(out_csv, index=False)
print(f"RESULT mode={mode} rows={len(df)} subjects={df['Subject'].nunique()} total_basis={total_basis} fit_seconds={secs:.1f} summary_seconds={sum_secs:.1f}")
