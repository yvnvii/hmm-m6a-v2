"""Numerical MLE for the beta-binomial in the Baum-Welch M-step.

Replaces the moment-based estimator. Given posterior responsibilities
gamma_t(z) at adenosine positions, we maximise the weighted log-likelihood
in (logit mu, log kappa) using L-BFGS-B.

Why this matters: the moment estimator equates Var(p_t) where p_t = A_t/n_t
with the latent beta variance mu(1-mu)/(kappa+1). That identity is wrong:
the observed proportion has additional 1/n_t binomial noise. Ignoring this
systematically underestimates kappa, sometimes severely.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from hmm_m6a.hmm import log_betabinom_pmf


def _bb_neg_weighted_ll(theta, k, n, w):
    logit_mu, log_kappa = theta
    mu = 1.0 / (1.0 + np.exp(-logit_mu))
    kappa = float(np.exp(log_kappa))
    a = mu * kappa
    b = (1.0 - mu) * kappa
    if a < 1e-8 or b < 1e-8:
        return 1e12
    ll = log_betabinom_pmf(k, n, a, b)
    return -float(np.sum(w * ll))


def fit_betabinom_weighted(
    k: np.ndarray,
    n: np.ndarray,
    w: np.ndarray,
    mu0: float = 0.1,
    kappa0: float = 20.0,
) -> Tuple[float, float]:
    """Weighted MLE of (mu, kappa) for a beta-binomial.

    Args:
        k: (N,) integer A-counts.
        n: (N,) integer total coverages.
        w: (N,) non-negative weights (posterior responsibilities).
        mu0, kappa0: initial guess used as a fallback if optimisation fails
            or if the input is empty.

    Returns:
        (mu_hat, kappa_hat).
    """
    keep = w > 1e-12
    if not np.any(keep):
        return float(mu0), float(kappa0)
    k_, n_, w_ = k[keep], n[keep], w[keep]

    p_hat = float(np.sum(w_ * (k_ / np.maximum(n_, 1))) / np.sum(w_))
    p_hat = float(np.clip(p_hat, 1e-3, 1 - 1e-3))
    theta0 = np.array(
        [np.log(p_hat / (1 - p_hat)), np.log(max(kappa0, 1.0))]
    )
    bounds = [(-15.0, 15.0), (np.log(1.05), np.log(1e5))]

    res = minimize(
        _bb_neg_weighted_ll,
        theta0,
        args=(k_, n_, w_),
        method="L-BFGS-B",
        bounds=bounds,
    )
    logit_mu, log_kappa = res.x
    mu = 1.0 / (1.0 + np.exp(-logit_mu))
    kappa = float(np.exp(log_kappa))
    return float(np.clip(mu, 1e-4, 1 - 1e-4)), float(np.clip(kappa, 1.0, 1e5))
