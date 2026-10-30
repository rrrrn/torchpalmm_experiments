import numpy as np
import torch
import torch.nn.functional as F
import hashlib
from itertools import product
from tqdm import tqdm
import os


def y_checksum(y):
    """Stable hash of the (centred) response, so chains reading from separate
    data directories can confirm they fit identical data."""
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    return hashlib.sha256(y_np.tobytes()).hexdigest()[:16]
def mu(t, a, b, c):
    return a * torch.sin(t+10) + c * torch.cos(t) + b * (t-10)*(t - 2) * (t-18)

def mu_mixed(t, kappa=10.0, t0=0.5):
    w = torch.sigmoid(kappa * (t - t0))
    return (1 - w) * torch.sin(2 * torch.pi * t) \
           + w * torch.sin(10 * torch.pi * t)


# ─────────────────────────────────────────────────────────────────────────────
# True-curve library for new-style experiments  (t ∈ [0, 1])
# ─────────────────────────────────────────────────────────────────────────────

def simulation_function_ratio(t, mu2, sigma2, A2):
    """Legacy alias: mu(t) with renamed parameters."""
    return mu(t, a=mu2, b=sigma2, c=A2)


def global_curve(t):
    return 2.0 * torch.sin(torch.pi * t.float())


def local_curve(t):
    t = t.float()
    return 0.2 * t**11 * (10*(1 - t))**6 + 10*(10*t)**3 * (1 - t)**10


MIXED_SS_CURVES = [
    lambda t: 2 * torch.sin(torch.pi * t.float()),
    lambda t: 2 * (3 * t.float()**2 - 2 * t.float()**3),           # smooth Hermite step 0→2
    lambda t: 3 * torch.exp(-15 * (t.float() - 0.2)**2),            # Gaussian bump at 0.2
    lambda t: 4 * t.float() * (1 - t.float()),
    lambda t: torch.sin(4 * torch.pi * t.float()),
    lambda t: torch.sqrt(torch.clamp(t.float()*(1-t.float()), min=1e-6)) * torch.sin(2*torch.pi*1.05/(t.float()+0.05)),
    lambda t: torch.tanh(8 * (t.float() - 0.5)),
    lambda t: torch.exp(-50 * (t.float() - 0.5)**2),
    lambda t: torch.exp(-100*(t.float()-0.25)**2) + 0.8*torch.exp(-100*(t.float()-0.75)**2),
    lambda t: torch.abs(t.float() - 0.5),
]

MIXED_SS_CURVE_LABELS = [
    "2·sin(πt)", "Hermite step", "Gaussian(0.2)", "4t(1−t)",
    "sin(4πt)", "√·sin", "tanh", "Gaussian(0.5)",
    "Bimodal", "|t−0.5|",
]


# ─────────────────────────────────────────────────────────────────────────────
# Curve standardization and experiment-design helpers
# ─────────────────────────────────────────────────────────────────────────────

# Standardize each curve to unit signal variance so sn varies the model
# dimension without also changing SNR. Constants persisted to JSON and reused by
# both runners so the ISE "truth" matches what was simulated.
_NORM_GRID = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)


def _curve_norm_stats(fn):
    vals = fn(_NORM_GRID).double()
    std = float(vals.std(unbiased=False))
    return float(vals.mean()), (std if std > 1e-8 else 1.0)


# (mean, std) per raw curve, computed once at import.
MIXED_SS_CURVE_STATS = [_curve_norm_stats(fn) for fn in MIXED_SS_CURVES]


def mixed_curve_normalized(curve_id):
    """Raw curve `curve_id` standardized to zero mean / unit variance on [0,1]."""
    fn = MIXED_SS_CURVES[curve_id]
    mean_d, std_d = MIXED_SS_CURVE_STATS[curve_id]
    return lambda t: (fn(t) - mean_d) / std_d


def ss_curve_ids(repeat_id, spline_dimension):
    """0-based curve ids (length sn) a replicate uses.

    Seeded by repeat_id only, so the subset is constant across N/sigma/D_type and
    sn varies only the count — removing the old fixed-order (curves 0..sn-1)
    confound where more smooths meant the smoother early curves.
    """
    n_curves = len(MIXED_SS_CURVES)
    rng = np.random.default_rng(7919 + int(repeat_id))
    if spline_dimension <= n_curves:
        return sorted(int(c) for c in
                      rng.choice(n_curves, size=spline_dimension, replace=False))
    base = sorted(int(c) for c in rng.permutation(n_curves))  # sn > library: cycle
    return [base[d % n_curves] for d in range(spline_dimension)]


_KNOT_FLOOR = 50
_KNOT_CEIL = 300


def ss_knot_count(N, lmin, lmax):
    """Interior knot count, 3.5·n^{1/3} (Ruppert, Wand & Carroll 2003), n=total obs.

    Clamped to [50, 300]; a fixed ceiling (not the old sqrt(1e9/N·8) memory cap)
    keeps K_c non-decreasing in N instead of shrinking it as N grows.
    """
    n_per = N * (lmin + lmax) / 2.0
    K_c = int(round(3.5 * n_per ** (1.0 / 3.0)))
    return int(np.clip(K_c, _KNOT_FLOOR, _KNOT_CEIL))


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_pure_ss_data(N, scenario="global", sigma_scale=1.0,
                           lmin=30, lmax=60, repeat_id=1):
    """Pure smoothing spline data — no fixed or random effects, t ∈ [0,1]."""
    seed_val = int(N * 997 + repeat_id * 31 + lmin)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    if lmax <= lmin:
        lmax = max(lmin + 1, int(np.ceil(lmin * 1.1)))

    lengths = np.random.randint(lmin, lmax + 1, size=N)
    total_obs = int(lengths.sum())
    K_c       = ss_knot_count(N, lmin, lmax)
    grid_size = K_c + 2
    t_list = [np.sort(np.random.randint(0, grid_size, size=l).astype(float) / grid_size)
              for l in lengths]

    curve_fn = global_curve if scenario == "global" else local_curve
    sigma2   = float(sigma_scale)

    y_list = []
    for t_i in t_list:
        t_torch = torch.tensor(t_i, dtype=torch.float32)
        y_i = curve_fn(t_torch).double() + torch.tensor(
            np.random.randn(len(t_i)) * np.sqrt(sigma2), dtype=torch.float64)
        y_list.append(y_i)

    y           = torch.cat(y_list)
    x           = torch.zeros(total_obs, 0, dtype=torch.float64)
    z           = torch.zeros(total_obs, 0, dtype=torch.float64)
    beta        = torch.zeros(0, dtype=torch.float64)
    D           = torch.zeros(0, 0, dtype=torch.float64)
    subject_id  = torch.cat([
        torch.full((int(l),), i, dtype=torch.long) for i, l in enumerate(lengths)
    ])
    splines_vars = torch.tensor(
        np.concatenate(t_list).reshape(-1, 1), dtype=torch.float64)

    return x, z, y, splines_vars, subject_id, beta, D, sigma2


def generate_smoothing_spline_lmm_data(N, spline_dimension=1, sigma_scale=1.0,
                                        a=None, b=None, c=None,
                                        seed=123, px=3, pz=2, lmin=10, lmax=-1,
                                        repeat_id=1,
                                        correlation_structure="unstructured",
                                        D_correlation_structure="full",
                                        rand_D=None, curve_ids=None,
                                        standardize_curves=True):
    """
    Generate LMM data with smoothing spline effects.

    Legacy mode  (a, b, c provided): integer t in [0, 20+1.5*lmax], true fn = mu(t, a, b, c).
    New mode     (rand_D provided or a is None): t in [0, 1], each component d uses
    the standardized curve ss_curve_ids(repeat_id, sn)[d] from MIXED_SS_CURVES, unless
    `curve_ids` pins a fixed set of curve ids instead (e.g. a single known truth
    that shouldn't vary by repeat_id). `standardize_curves=False` uses each curve's
    raw (un-normalized) form instead -- for a single fixed curve there is no
    cross-curve SNR to equalize, and a caller may want the literal formula (e.g.
    f(t)=2*sin(pi*t), curve id 0) rather than its zero-mean/unit-variance rescaling.
    """
    if rand_D is not None:
        D_correlation_structure = rand_D

    if a is not None:
        return _generate_ss_lmm_legacy(
            N, spline_dimension, sigma_scale, a, b, c,
            seed, px, pz, lmin, lmax, repeat_id,
            correlation_structure, D_correlation_structure)

    # ── New-style: t ∈ [0,1], MIXED_SS_CURVES ──────────────────────────────
    seed_val = int(N * 997 + spline_dimension * 53 + repeat_id * 31 + lmin * 7)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # lmax<=0 means "unspecified, derive from lmin"; lmin>lmax is a genuinely
    # reversed range. lmin==lmax is left alone so callers can request a fixed
    # per-subject length (lmin==lmax==15, say) without being overridden.
    if lmax <= 0 or lmin > lmax:
        lmax = max(lmin + 1, int(np.ceil(lmin * 1.1)))

    lengths  = np.random.randint(lmin, lmax + 1, size=N)
    total_obs = int(lengths.sum())

    K_c       = ss_knot_count(N, lmin, lmax)
    grid_size = K_c + 2
    # Per-replicate random curve subset (seeded by repeat_id), standardized --
    # unless the caller pinned a fixed set via `curve_ids`.
    if curve_ids is None:
        curve_ids = ss_curve_ids(repeat_id, spline_dimension)
    # Each dimension d is sampled independently: g(x1,...,xsn) = f_0(x_1)+...+f_{sn-1}(x_sn).
    # Do NOT sort per-column: sorting each component independently within a
    # subject makes the covariates co-monotonic (pairwise r~0.93), which is
    # spurious concurvity that ill-conditions the stacked linear design and
    # inflates per-component ISE as sn grows.
    t_list = [
        np.column_stack([
            np.random.randint(0, grid_size, size=l).astype(float) / grid_size
            for d in range(spline_dimension)
        ])
        for l in lengths
    ]
    x_list = [np.random.randn(l, px) for l in lengths] if px > 0 else [np.zeros((l, 0)) for l in lengths]
    # Random effects matrix Z: first column is a random intercept, remaining
    # pz-1 columns i.i.d. N(0,1) (mirrors SimScripts/RandeffLMM.py).
    if pz > 0:
        z_list = [np.hstack((np.ones((l, 1)), np.random.randn(l, max(pz - 1, 0))))
                  for l in lengths]
    else:
        z_list = [np.zeros((l, 0)) for l in lengths]

    beta = (torch.arange(1, px + 1, dtype=torch.float64) if px > 0
            else torch.zeros(0, dtype=torch.float64))

    if pz > 0:
        if D_correlation_structure in ("diagonal",):
            D_vals = torch.tensor(np.random.uniform(0.5, 2.0, size=pz), dtype=torch.float64)
            D = torch.diag(D_vals)
        else:
            _L = torch.tensor(np.random.randn(pz, pz), dtype=torch.float64)
            D = _L @ _L.T
        u_list = [
            torch.tensor(
                np.random.multivariate_normal(np.zeros(pz), D.numpy()),
                dtype=torch.float64)
            for _ in range(N)
        ]
    else:
        D      = torch.zeros(0, 0, dtype=torch.float64)
        u_list = [torch.zeros(0, dtype=torch.float64) for _ in range(N)]

    sigma2 = float(sigma_scale)

    y_list = []
    for x_i, z_i, u_i, t_i in zip(x_list, z_list, u_list, t_list):
        l_i = len(x_i)
        y_i = torch.zeros(l_i, dtype=torch.float64)
        if px > 0:
            y_i = y_i + torch.tensor(x_i, dtype=torch.float64) @ beta
        if pz > 0:
            y_i = y_i + torch.tensor(z_i, dtype=torch.float64) @ u_i
        for d in range(spline_dimension):
            fn  = (mixed_curve_normalized(curve_ids[d]) if standardize_curves
                   else MIXED_SS_CURVES[curve_ids[d]])
            t_d = torch.tensor(t_i[:, d], dtype=torch.float32)
            y_i = y_i + fn(t_d).double()
        y_i = y_i + torch.tensor(np.random.randn(l_i) * np.sqrt(sigma2), dtype=torch.float64)
        y_list.append(y_i)

    x = torch.tensor(
        np.vstack(x_list) if px > 0 else np.zeros((total_obs, 0)), dtype=torch.float64)
    z = torch.tensor(
        np.vstack(z_list) if pz > 0 else np.zeros((total_obs, 0)), dtype=torch.float64)
    y          = torch.cat(y_list)
    subject_id = torch.cat([
        torch.full((int(l),), i, dtype=torch.long) for i, l in enumerate(lengths)
    ])
    splines_vars = torch.tensor(np.vstack(t_list), dtype=torch.float64)

    return x, z, y, splines_vars, subject_id, beta, D, sigma2


def _generate_ss_lmm_legacy(N, spline_dimension, sigma_scale, a, b, c,
                              seed, px, pz, lmin, lmax, repeat_id,
                              correlation_structure, D_correlation_structure):
    """Original generator: integer t in [0, 20+1.5*lmax], true fn = mu(t, a, b, c)."""
    seed_val = int(N + 1000*sigma_scale + repeat_id + a*b*c*10000 + seed)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if lmin >= lmax:
        lmax = np.ceil(lmin * 1.1)
    lengths = np.random.randint(lmin, int(lmax), size=N)
    spline_vars_obs = [
        np.random.randint(0, 20 + int(1.5 * lmax), size=(l, spline_dimension))
        for l in lengths
    ]
    x_list = [np.random.randn(l, px) for l in lengths]
    z_list = [np.random.randn(l, pz) for l in lengths]
    beta = torch.arange(1, px + 1, dtype=torch.float32)
    if D_correlation_structure == "diagonal":
        D_vals = torch.tensor(np.random.uniform(0.5, 2.0, size=pz), dtype=torch.float32)
        D = torch.diag(D_vals)
    else:
        L = torch.tensor(np.random.normal(size=(pz, pz)) * 5, dtype=torch.float32)
        D = L @ L.T
    sigma2 = sigma_scale
    if pz != 0:
        u_list = [
            torch.from_numpy(
                np.random.multivariate_normal(np.zeros(D.shape[0]), D.numpy(), size=1).T
            ).float()
            for _ in range(N)
        ]
    else:
        u_list = [torch.zeros((0, 1), dtype=torch.float32) for _ in range(N)]
    e_list = [torch.randn(l, 1) * np.sqrt(sigma2) for l in lengths]
    y_list = []
    for x_i, z_i, u_i, e_i, sv in zip(x_list, z_list, u_list, e_list, spline_vars_obs):
        x_i_t  = torch.from_numpy(x_i).float()
        z_i_t  = torch.from_numpy(z_i).float()
        mu_val = sum(mu(torch.from_numpy(sv[:, d]).float(), a=a, b=b, c=c)
                     for d in range(spline_dimension))
        mu_i   = mu_val.unsqueeze(1)                           # (l, 1)
        y_i    = x_i_t @ beta.unsqueeze(1) + mu_i + e_i
        if pz != 0:
            y_i = y_i + z_i_t @ u_i
        y_list.append(y_i)
    x = torch.cat([torch.tensor(xi, dtype=torch.float32) for xi in x_list], dim=0)
    z = torch.cat([torch.tensor(zi, dtype=torch.float32) for zi in z_list], dim=0)
    y = torch.cat([torch.tensor(yi, dtype=torch.float32) for yi in y_list], dim=0)
    subject_id = torch.cat([
        torch.full((l,), i, dtype=torch.long) for i, l in enumerate(lengths)
    ])
    splines_vars = torch.cat([
        torch.tensor(sv_i, dtype=torch.float32) for sv_i in spline_vars_obs
    ], dim=0)
    return x, z, y, splines_vars, subject_id, beta, D, sigma2

if __name__ == "__main__":
    N_seq = [1000]
    px_seq = [5]
    pz_seq = [1, 5, 10]
    length_min = [5, 10, 50]
    sigma2_seq = [10]
    spline_dim_vec = [1, 3, 5]
    repeat_num = 50
    repeat_grid = np.arange(1, repeat_num + 1).tolist()
    correlation_structure_seq = ["iid"]
    D_cor_structure = ["diagonal", "unstructured"]
    param_grid = list(product(N_seq, px_seq, pz_seq, sigma2_seq, repeat_grid, 
                                correlation_structure_seq, D_cor_structure,
                                length_min, spline_dim_vec))
    ## filter grid to ensure N > px + pz
    grid_filtered = []
    for params in param_grid:
        N, px, pz, sigma2, repeat_id, corr_structure, D_corr_structure, lm, sd = params
        grid_filtered.append(params)
    os.makedirs('Data/ss_scal_h5', exist_ok=True)

    for (N, px, pz, sigma2, repeat_id, correlation_structure, D_corr_structure, lm, sd) in tqdm(grid_filtered):
        # if os.path.exists(f'Data/ss_scal_h5/N{N}_sd{sd}_pz{pz}_lm{lm}_sigma2{sigma2}_repeat{repeat_id}_{correlation_structure}_D{D_corr_structure}.h5'):
        #     continue
        # if number of data points is less than number of parameters, skip
        lmm_data = generate_smoothing_spline_lmm_data(N = N, px = px, pz = pz, sigma_scale = sigma2, repeat_id=repeat_id, correlation_structure=correlation_structure, 
                                        lmin=lm, D_correlation_structure=D_corr_structure, spline_dimension=sd)
        