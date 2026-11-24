"""torchpalmm SPT bleeding-on-probing model for the memory / runtime benchmark: reads the analysis sample, builds the model matrices and fits (REML, L-BFGS).
Same model as fit_logit_noround_BOP_SP.ipynb: per baseline class x sex group one follow-up curve and one curve of BOP at the first visit, fixed effects, random intercept + slope on age with unstructured D, iid residuals.
usage: python benchmark_spt_torchpalmm.py MODE CSV OUT.csv [D_TYPE]     MODE = baseline (imports and data only) | fit (matrices + REML fit) | summary (fit, then the library's inference summary: SEs, p-values, curve tests)"""
# Data: Supportive Periodontal Therapy Clinical Examination Data (PhysioNet), https://physionet.org/content/supportive-periodontal-therapy/1.0.0/
# Download it into analysis/spt_data/ (not tracked by git) and curate it with analysis/curate_spt_bleeding.py.
import sys, time
import numpy as np, pandas as pd, torch
import torchpalmm as tp
torch.set_num_threads(1)                                                                   # one thread, as for the mgcv fits (runtime and memory comparison)

mode, csv, out_csv = sys.argv[1:4]
D_type = sys.argv[4] if len(sys.argv) > 4 else "unstructured"                          # random-effect covariance: "unstructured" (default) or "diagonal" (the structure bam fits)
df = pd.read_csv(csv).sort_values(["pat_id", "spt_id"]).reset_index(drop=True)
if mode == "baseline":
    print(f"RESULT mode=baseline rows={len(df)}"); sys.exit()

x_cols = ["age_z", "year_z", "teeth_z", "logdays_z", "cls2", "cls3", "female", "ever", "cls2_female", "cls3_female"]   # column order of the notebook's X
X = df[x_cols].to_numpy(float)
Z = np.column_stack([np.ones(len(df)), df["age_z"].to_numpy(float)])                      # random intercept + slope on age at visit
y = torch.as_tensor(df["y"].to_numpy(float)).view(-1, 1)                                 # logit((bleeding % + 1) / 102)
obs_id = pd.factorize(df["pat_id"])[0]
groups = sorted(df["grp"].unique())                                                        # class x sex groups: one curve per smooth and group (group-specific)
smooth_vars = [("fu", "Follow-up time"), ("bop_first", "BOP at first visit")]
spline_inputs = np.column_stack([df[v].to_numpy(float) for v, _ in smooth_vars for _ in groups])
smooth_mask = np.column_stack([(df["grp"] == g).to_numpy(float) for _ in smooth_vars for g in groups])
names = [f"{lab}, {g}" for _, lab in smooth_vars for g in groups]

t0 = time.perf_counter()
ss = tp.SmoothingSplineMixed(y=y, X=X, Z=Z, spline_inputs=spline_inputs, obs_id=obs_id, smooth_mask=smooth_mask, corr_structure="iid", rand_D_type=D_type)
total_basis = sum(b.shape[-1] for b in ss.NB_list)
torch.manual_seed(0)
ss.fit(optimizer=torch.optim.LBFGS, lr=5, max_iter=10000, fit_type="reml", convg_check="all")
secs = time.perf_counter() - t0
beta = ss.beta.detach().cpu().numpy().ravel()[:X.shape[1]]                                  # fixed effects, in the order of the columns of X
sum_secs = 0.0
if mode == "summary":
    t1 = time.perf_counter(); res = ss.summary(names=x_cols, spline_names=names, re_names=["intercept", "age_z"]); sum_secs = time.perf_counter() - t1
D = ss.D.detach().cpu().numpy(); s2 = float(ss.sigma2)
se = {t: np.nan for t in x_cols}; edf = np.full(len(names), np.nan)
if mode == "summary":
    co = res["coefficients"].set_index("term"); se = {t: float(co.loc[t, "se"]) if t in co.index else np.nan for t in x_cols}
    edf = res["splines"].set_index("spline_id")["edf"].reindex(range(len(names))).to_numpy(float)
pd.DataFrame({"term": x_cols + ["sd_intercept", "sd_age", "corr_int_age", "sigma"],
              "estimate": list(beta) + [np.sqrt(D[0, 0]), np.sqrt(D[1, 1]), D[0, 1] / np.sqrt(D[0, 0] * D[1, 1]) if D_type == "unstructured" else np.nan, np.sqrt(s2)],
              "se": [se[t] for t in x_cols] + [np.nan] * 4}).to_csv(out_csv, index=False)
pd.DataFrame({"term": names, "edf": edf}).to_csv(out_csv.replace(".csv", "_edf.csv"), index=False)
print(f"RESULT mode={mode} D={D_type} rows={len(df)} subjects={df['pat_id'].nunique()} total_basis={total_basis} smooth_coefs={total_basis + len(names)} woodbury={ss.lmm._woodbury} epochs={ss.lmm.n_epochs} lbfgs_steps={ss.lmm.n_lbfgs_steps} func_evals={ss.lmm.n_func_evals} converged={int(bool(ss.lmm.convg))} fit_seconds={secs:.1f} summary_seconds={sum_secs:.1f}")
