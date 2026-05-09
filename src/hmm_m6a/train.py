"""Baum-Welch (EM) training for the per-nucleotide HMM.

Implements Methods Section 2.13: per-nucleotide forward-backward in the
E-step, weighted MLE for the beta-binomial parameters in the M-step,
Dirichlet-pseudocount updates for pi, T, and the DRACH emission, and the
state-ordering constraint mu_U < mu_B < mu_M after each M-step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from hmm_m6a.hmm import (
    HMMParams,
    K,
    emission_logprob,
    expected_xi_sum,
    forward_backward,
)
from hmm_m6a.mle import fit_betabinom_weighted


@dataclass
class TranscriptObs:
    """A single transcript's observation tuple, ready for HMM scoring/training."""

    L: int
    A_idx: np.ndarray   # (n_A,) int, indices of adenosine positions
    A_count: np.ndarray # (L,) int
    n: np.ndarray       # (L,) int
    drach: np.ndarray   # (L,) bool


@dataclass
class TrainHistory:
    logZ: List[float] = field(default_factory=list)
    mu: List[np.ndarray] = field(default_factory=list)
    kappa: List[np.ndarray] = field(default_factory=list)
    p_drach: List[np.ndarray] = field(default_factory=list)


def _enforce_state_order(params: HMMParams) -> HMMParams:
    """Permute states so that mu_U < mu_B < mu_M (i.e. index 0 = U, 1 = M, 2 = B).

    The permutation is applied jointly to pi, T, mu, kappa, and p_drach so
    that all parameters remain mutually consistent.
    """
    order = np.argsort(params.mu)               # ascending mu: lowest, mid, highest
    perm = np.array([order[0], order[2], order[1]])  # -> (U, M, B)
    return HMMParams(
        pi=params.pi[perm],
        T=params.T[perm][:, perm],
        mu=params.mu[perm],
        kappa=params.kappa[perm],
        p_drach=params.p_drach[perm],
    )


def baum_welch(
    transcripts: Sequence[TranscriptObs],
    params: HMMParams | None = None,
    n_iter: int = 30,
    lam: float = 1.0,
    tol: float = 1e-3,
    verbose: bool = True,
) -> Tuple[HMMParams, TrainHistory]:
    """Run Baum-Welch.

    Args:
        transcripts: list of TranscriptObs.
        params: starting parameters; defaults to HMMParams.default().
        n_iter: maximum number of EM iterations.
        lam: Dirichlet pseudocount for pi, T, p_drach.
        tol: relative log-likelihood convergence tolerance.
        verbose: print per-iteration diagnostics.

    Returns:
        (final params, training history).
    """
    if params is None:
        params = HMMParams.default()
    params = params.copy()
    history = TrainHistory()
    prev_ll = -np.inf

    for it in range(n_iter):
        sum_pi = np.zeros(K)
        sum_xi = np.zeros((K, K))
        sum_gamma_A = np.zeros(K)
        sum_gamma_drach = np.zeros(K)
        all_k_chunks: List[np.ndarray] = []
        all_n_chunks: List[np.ndarray] = []
        all_w_chunks: List[List[np.ndarray]] = [[] for _ in range(K)]
        total_logZ = 0.0

        log_pi = np.log(np.clip(params.pi, 1e-300, None))
        log_T = np.log(np.clip(params.T, 1e-300, None))

        for tr in transcripts:
            log_e = emission_logprob(
                tr.A_idx, tr.A_count, tr.n, tr.drach, tr.L, params
            )
            la, lb, lg, lZ = forward_backward(log_e, log_pi, log_T)
            total_logZ += lZ
            gamma = np.exp(lg)
            sum_pi += gamma[0]
            sum_xi += expected_xi_sum(la, lb, log_e, log_T, lZ)

            if tr.A_idx.size:
                gA = gamma[tr.A_idx]
                kA = tr.A_count[tr.A_idx]
                nA = tr.n[tr.A_idx]
                drachA = tr.drach[tr.A_idx]
                sum_gamma_A += gA.sum(axis=0)
                sum_gamma_drach += gA[drachA].sum(axis=0)
                all_k_chunks.append(kA)
                all_n_chunks.append(nA)
                for z in range(K):
                    all_w_chunks[z].append(gA[:, z])

        # M-step
        pi_new = (sum_pi + lam) / (sum_pi.sum() + K * lam)
        T_new = (sum_xi + lam) / (sum_xi.sum(axis=1, keepdims=True) + K * lam)
        p_drach_new = (sum_gamma_drach + lam) / (sum_gamma_A + 2 * lam)

        k_all = np.concatenate(all_k_chunks) if all_k_chunks else np.array([], dtype=int)
        n_all = np.concatenate(all_n_chunks) if all_n_chunks else np.array([], dtype=int)
        mu_new = np.empty(K)
        kappa_new = np.empty(K)
        for z in range(K):
            w_z = (
                np.concatenate(all_w_chunks[z])
                if all_w_chunks[z]
                else np.array([], dtype=float)
            )
            mu_new[z], kappa_new[z] = fit_betabinom_weighted(
                k_all, n_all, w_z,
                mu0=float(params.mu[z]),
                kappa0=float(params.kappa[z]),
            )

        params = HMMParams(
            pi=pi_new, T=T_new, mu=mu_new, kappa=kappa_new, p_drach=p_drach_new
        )
        params = _enforce_state_order(params)

        history.logZ.append(total_logZ)
        history.mu.append(params.mu.copy())
        history.kappa.append(params.kappa.copy())
        history.p_drach.append(params.p_drach.copy())

        if verbose:
            print(
                f"iter {it+1:3d}  logZ={total_logZ:12.2f}  "
                f"mu={params.mu.round(4)}  "
                f"kappa={params.kappa.round(1)}  "
                f"p_drach={params.p_drach.round(3)}"
            )

        if it > 0 and abs(total_logZ - prev_ll) < tol * max(1.0, abs(prev_ll)):
            if verbose:
                print(f"converged at iter {it+1}")
            break
        prev_ll = total_logZ

    return params, history
