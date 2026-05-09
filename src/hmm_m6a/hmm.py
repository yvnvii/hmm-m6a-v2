"""Core HMM machinery: parameters, emissions, forward-backward, Viterbi.

State indexing convention (preserved by the state-ordering constraint after
each Baum-Welch M-step):

    0 = U  (unmethylated / clean background, mu_U lowest)
    1 = M  (true m6A,                         mu_M highest)
    2 = B  (artifact / poor conversion,       mu_B intermediate)

The chain runs over every nucleotide of a transcript. Emission at non-A
positions is neutral (log P = 0), exactly as specified in Methods Section 2.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import gammaln

K = 3  # number of hidden states
STATE_NAMES = ("U", "M", "B")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class HMMParams:
    """Container for HMM parameters.

    Attributes:
        pi:       (3,) start probabilities.
        T:        (3, 3) row-stochastic transition matrix.
        mu:       (3,) state-specific A-retention means.
        kappa:    (3,) state-specific beta-binomial concentrations.
        p_drach:  (3,) state-specific P(DRACH motif | state).
    """

    pi: np.ndarray
    T: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
    p_drach: np.ndarray

    @staticmethod
    def default() -> "HMMParams":
        """Sensible data-informed initialisation. Used as the starting point
        for Baum-Welch in the published validation experiments."""
        return HMMParams(
            pi=np.array([0.95, 0.02, 0.03]),
            T=np.array([
                [0.985, 0.005, 0.010],   # from U
                [0.300, 0.650, 0.050],   # from M
                [0.100, 0.020, 0.880],   # from B
            ]),
            mu=np.array([0.03, 0.65, 0.25]),
            kappa=np.array([60.0, 30.0, 25.0]),
            p_drach=np.array([0.30, 0.85, 0.30]),
        )

    def alpha_beta(self) -> Tuple[np.ndarray, np.ndarray]:
        a = self.mu * self.kappa
        b = (1.0 - self.mu) * self.kappa
        return a, b

    def copy(self) -> "HMMParams":
        return HMMParams(
            pi=self.pi.copy(),
            T=self.T.copy(),
            mu=self.mu.copy(),
            kappa=self.kappa.copy(),
            p_drach=self.p_drach.copy(),
        )

    # JSON serialisation, useful for distributing trained parameters.
    def to_dict(self) -> dict:
        return {
            "pi": self.pi.tolist(),
            "T": self.T.tolist(),
            "mu": self.mu.tolist(),
            "kappa": self.kappa.tolist(),
            "p_drach": self.p_drach.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "HMMParams":
        return HMMParams(
            pi=np.asarray(d["pi"], dtype=float),
            T=np.asarray(d["T"], dtype=float),
            mu=np.asarray(d["mu"], dtype=float),
            kappa=np.asarray(d["kappa"], dtype=float),
            p_drach=np.asarray(d["p_drach"], dtype=float),
        )


# ---------------------------------------------------------------------------
# Beta-binomial PMF
# ---------------------------------------------------------------------------
def log_betabinom_pmf(
    k: np.ndarray,
    n: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Vectorised log P(k | n, alpha, beta) for the beta-binomial distribution."""
    return (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + gammaln(k + alpha) + gammaln(n - k + beta) - gammaln(n + alpha + beta)
        + gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    )


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def emission_logprob(
    A_idx: np.ndarray,
    A_count: np.ndarray,
    n: np.ndarray,
    drach: np.ndarray,
    L: int,
    params: HMMParams,
) -> np.ndarray:
    """Compute the (L, 3) log-emission matrix for a single transcript.

    Args:
        A_idx:    (n_A,) int, indices of adenosine positions in the transcript.
        A_count:  (L,) int, A-like read counts. Only entries at A_idx matter.
        n:        (L,) int, total informative coverage. Only A_idx matters.
        drach:    (L,) bool, DRACH-motif flags. Only A_idx matters.
        L:        transcript length.
        params:   HMMParams.

    Returns:
        (L, 3) array of log-emission probabilities. Non-A rows are zero.
    """
    log_e = np.zeros((L, K))
    if A_idx.size == 0:
        return log_e

    a, b = params.alpha_beta()
    k_obs = A_count[A_idx]
    n_obs = n[A_idx]
    drach_obs = drach[A_idx]

    for z in range(K):
        bb = log_betabinom_pmf(k_obs, n_obs, a[z], b[z])
        p_d = float(np.clip(params.p_drach[z], 1e-12, 1 - 1e-12))
        ctx = np.where(drach_obs, np.log(p_d), np.log(1 - p_d))
        log_e[A_idx, z] = bb + ctx

    return log_e


# ---------------------------------------------------------------------------
# Manual log-sum-exp helpers (faster than scipy.logsumexp on small K)
# ---------------------------------------------------------------------------
def _lse_axis0(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=0)
    return m + np.log(np.exp(x - m).sum(axis=0))


def _lse_axis1(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1)
    return m + np.log(np.exp(x - m[:, None]).sum(axis=1))


def _lse_1d(v: np.ndarray) -> float:
    m = float(v.max())
    return m + float(np.log(np.exp(v - m).sum()))


# ---------------------------------------------------------------------------
# Forward-backward
# ---------------------------------------------------------------------------
def forward_backward(
    log_e: np.ndarray,
    log_pi: np.ndarray,
    log_T: np.ndarray,
):
    """Per-nucleotide forward-backward in log space.

    Returns log_alpha, log_beta, log_gamma, log_Z.
    """
    L, K_ = log_e.shape

    log_alpha = np.empty((L, K_))
    log_alpha[0] = log_pi + log_e[0]
    for t in range(1, L):
        log_alpha[t] = _lse_axis0(log_alpha[t - 1][:, None] + log_T) + log_e[t]

    log_beta = np.empty((L, K_))
    log_beta[-1] = 0.0
    for t in range(L - 2, -1, -1):
        log_beta[t] = _lse_axis1(log_T + (log_e[t + 1] + log_beta[t + 1])[None, :])

    log_Z = _lse_1d(log_alpha[-1])
    log_gamma = log_alpha + log_beta - log_Z
    return log_alpha, log_beta, log_gamma, log_Z


def expected_xi_sum(log_alpha, log_beta, log_e, log_T, log_Z):
    """Sum over t of P(Z_t=a, Z_{t+1}=b | O). Returns (K, K)."""
    la = log_alpha[:-1]
    le_next = log_e[1:]
    lb_next = log_beta[1:]
    log_xi = (
        la[:, :, None]
        + log_T[None, :, :]
        + (le_next + lb_next)[:, None, :]
        - log_Z
    )
    return np.exp(log_xi).sum(axis=0)


def viterbi(
    log_e: np.ndarray,
    log_pi: np.ndarray,
    log_T: np.ndarray,
) -> np.ndarray:
    """Most-likely state path."""
    L, K_ = log_e.shape
    delta = np.full((L, K_), -np.inf)
    psi = np.zeros((L, K_), dtype=np.int32)
    delta[0] = log_pi + log_e[0]
    for t in range(1, L):
        scores = delta[t - 1][:, None] + log_T
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K_)] + log_e[t]
    path = np.zeros(L, dtype=np.int32)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(L - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def posterior_states(
    A_idx: np.ndarray,
    A_count: np.ndarray,
    n: np.ndarray,
    drach: np.ndarray,
    L: int,
    params: HMMParams,
):
    """Convenience wrapper: returns (gamma, log_Z) for a single transcript."""
    log_e = emission_logprob(A_idx, A_count, n, drach, L, params)
    log_pi = np.log(np.clip(params.pi, 1e-300, None))
    log_T = np.log(np.clip(params.T, 1e-300, None))
    _, _, log_gamma, log_Z = forward_backward(log_e, log_pi, log_T)
    return np.exp(log_gamma), log_Z
