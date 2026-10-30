"""torchpalmm SPT bleeding-on-probing model for the memory / runtime benchmark: reads the analysis sample, builds the model matrices and fits (REML, L-BFGS).
Same model as fit.ipynb: one follow-up curve per baseline class x sex group, fixed effects, random intercept + slopes (age, log interval) with unstructured D, iid residuals.
usage: python benchmark_spt_torchpalmm.py MODE CSV OUT.csv     MODE = baseline (imports and data only) | fit (matrices + REML fit) | summary (fit, then the library's inference summary: SEs, p-values, curve tests)"""
# Data: Supportive Periodontal Therapy Clinical Examination Data (PhysioNet), https://physionet.org/content/supportive-periodontal-therapy/1.0.0/
# Download it into analysis/spt_data/ (not tracked by git) and curate it with analysis/curate_spt_bleeding.py.
import sys, time
import numpy as np, pandas as pd, torch
import torchpalmm as tp

mode, csv, out_csv = sys.argv[1:4]
df = pd.read_csv(csv).sort_values(["pat_id", "spt_id"]).reset_index(drop=True)
if mode == "baseline":
    print(f"RESULT mode=baseline rows={len(df)}"); sys.exit()

x_cols = ["age_z", "year_z", "teeth_z", "logdays_z", "cls2", "cls3", "female", "ever", "cls2_female", "cls3_female"]   # column order of the notebook's X
X = df[x_cols].to_numpy(float)
Z = np.column_stack([np.ones(len(df)), df[["age_z", "logdays_z"]].to_numpy(float)])      # random intercept + slopes on age at visit and log interval
y = torch.as_tensor(df["y"].to_numpy(float)).view(-1, 1)                                 # log(bleeding % + 1)
obs_id = pd.factorize(df["pat_id"])[0]
groups = sorted(df["grp"].unique())                                                        # class x sex groups: one follow-up curve each (group-specific)
spline_inputs = np.column_stack([df["fu"].to_numpy(float)] * len(groups))
smooth_mask = np.column_stack([(df["grp"] == g).to_numpy(float) for g in groups])
names = [f"Follow-up time, {g}" for g in groups]

t0 = time.perf_counter()
ss = tp.SmoothingSplineMixed(y=y, X=X, Z=Z, spline_inputs=spline_inputs, obs_id=obs_id, smooth_mask=smooth_mask, corr_structure="iid", rand_D_type="unstructured")
total_basis = sum(b.shape[2] for b in ss.NB_list)
torch.manual_seed(0)
ss.fit(optimizer=torch.optim.LBFGS, lr=1, max_iter=10000, fit_type="reml", convg_check="all")
secs = time.perf_counter() - t0
beta = ss.beta.detach().cpu().numpy().ravel()[:X.shape[1]]                                  # fixed effects, in the order of the columns of X
sum_secs = 0.0
if mode == "summary":
    t1 = time.perf_counter(); res = ss.summary(names=x_cols, spline_names=names, re_names=["intercept", "age_z", "logdays_z"]); sum_secs = time.perf_counter() - t1
pd.DataFrame({"term": x_cols, "estimate": beta}).to_csv(out_csv, index=False)
print(f"RESULT mode={mode} rows={len(df)} subjects={df['pat_id'].nunique()} total_basis={total_basis} woodbury={ss.lmm._woodbury} epochs={ss.lmm.n_epochs} lbfgs_steps={ss.lmm.n_lbfgs_steps} func_evals={ss.lmm.n_func_evals} fit_seconds={secs:.1f} summary_seconds={sum_secs:.1f}")
