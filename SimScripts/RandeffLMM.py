"""
Generate random-effects scaling experiment data (sn=1, pz swept).

This module is focused on the randeff experiment: fixed single smooth,
varying random effects dimension pz, with support for both IID and
compound-symmetry residual correlation structures.

Reuses MIXED_SS_CURVES (true function) and adaptive grid logic from LMMSS.
"""

import numpy as np
import torch
from SimScripts.LMMSS import MIXED_SS_CURVES


def generate_randeff_data(
    N,
    px=2,
    pz=2,
    sigma_scale=1.0,
    lmin=20,
    lmax=30,
    repeat_id=1,
    rand_D="diagonal",
    corr_structure="iid",
):
    """Generate random-effects scaling experiment data.

    Args:
        N: number of subjects
        px: number of fixed effects (not including intercept)
        pz: number of random effects
        sigma_scale: residual SD
        lmin, lmax: min/max observations per subject
        repeat_id: random seed (via seed = N*997 + pz*53 + repeat_id*31 + lmin*7)
        rand_D: "diagonal" or "unstructured" D matrix
        corr_structure: "iid" or "compound_symmetry" for residuals

    Returns:
        (x, z, y, splines_vars, subject_id, beta, D, sigma2, rho)
        All as numpy arrays except y is torch.Tensor
    """
    # Set seed deterministically
    seed_val = int(N * 997 + pz * 53 + repeat_id * 31 + lmin * 7)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # Single smooth (sn=1), adaptive grid sizing
    spline_dimension = 1
    n_per = N * (lmin + lmax) / 2.0
    mem_cap = int(np.sqrt(1e9 / (N * 8)))
    K_c = max(50, min(int(round(3.5 * n_per ** (1 / 3))), mem_cap))
    grid_size = K_c + 2

    # Fixed effects
    beta = np.arange(1, px + 1, dtype=float)

    # Random effects covariance D
    if rand_D == "diagonal":
        D_diag = np.random.uniform(0.5, 2.0, size=pz)
        D = np.diag(D_diag)
    else:  # unstructured
        L = np.random.normal(0, 1, size=(pz, pz))
        D = L @ L.T

    sigma2 = float(sigma_scale)

    # Compound symmetry correlation for residuals
    rho_cs = 0.4 if corr_structure == "compound_symmetry" else 0.0

    # Generate data for each subject
    x_list = []
    z_list = []
    y_list = []
    splines_vars_list = []
    lengths = np.random.randint(lmin, lmax + 1, size=N)
    subject_id_list = []

    for i in range(N):
        l_i = int(lengths[i])

        # Fixed effects matrix X (l_i x px)
        x_i = np.random.normal(0, 1, size=(l_i, px))
        x_list.append(x_i)

        # Random effects matrix Z (l_i x pz): first column is a random intercept,
        # remaining pz-1 columns i.i.d. N(0,1) (mirrors SimScripts/LMMSS.py).
        if pz > 0:
            z_i = np.hstack((np.ones((l_i, 1)),
                             np.random.normal(0, 1, size=(l_i, max(pz - 1, 0)))))
        else:
            z_i = np.zeros((l_i, 0))
        z_list.append(z_i)

        # Spline variable t in [0, 1]
        t_i = np.sort(np.random.randint(0, grid_size, size=l_i).astype(float) / grid_size)
        splines_vars_list.append(t_i)

        # Random effects u_i ~ N(0, D)
        if pz > 0:
            u_i = np.random.multivariate_normal(np.zeros(pz), D, size=1).T  # (pz, 1)
        else:
            u_i = np.zeros((pz, 1))

        # Mean: X*beta + smooth(t)
        smooth_val = MIXED_SS_CURVES[0](torch.tensor(t_i, dtype=torch.float32)).numpy()
        mu_i = x_i @ beta + smooth_val

        # Random effects contribution: Z*u
        if pz > 0:
            re_contrib = z_i @ u_i
        else:
            re_contrib = np.zeros((l_i, 1))

        # Residuals with correlation structure
        if rho_cs > 0:
            # Compound symmetry: Cov(e, e') = rho*sigma2 if different, sigma2 if same
            Sigma_e = sigma2 * (
                rho_cs * np.ones((l_i, l_i)) + (1 - rho_cs) * np.eye(l_i)
            )
            e_i = np.random.multivariate_normal(np.zeros(l_i), Sigma_e)
        else:
            # IID residuals
            e_i = np.random.randn(l_i) * np.sqrt(sigma2)

        # Response
        y_i = mu_i + re_contrib.flatten() + e_i
        y_list.append(y_i)

        # Subject ID
        subject_id_list.extend([i] * l_i)

    # Stack into matrices
    x = np.vstack(x_list) if px > 0 else np.zeros((sum(lengths), 0))
    z = np.vstack(z_list) if pz > 0 else np.zeros((sum(lengths), 0))
    y = torch.tensor(np.concatenate(y_list), dtype=torch.float64)
    y = y.unsqueeze(1)
    y = y - y.mean()  # Center
    splines_vars = np.concatenate(splines_vars_list)
    subject_id = np.array(subject_id_list, dtype=np.int32)

    return x, z, y, splines_vars, subject_id, beta, D, sigma2, rho_cs
