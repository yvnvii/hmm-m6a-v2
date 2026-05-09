"""Smoke tests for the package.

Run with `pytest`.

These tests focus on the math (forward-backward correctness, parameter
recovery on synthetic data, calling rule) without requiring a BAM file.
The pileup module is tested separately on a tiny in-memory SAM fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from hmm_m6a.hmm import HMMParams, emission_logprob, forward_backward, posterior_states
from hmm_m6a.train import TranscriptObs, baum_welch
from hmm_m6a.calling import call_sites
from hmm_m6a.pileup import Region


# ---------------------------------------------------------------------------
# Forward-backward sanity
# ---------------------------------------------------------------------------
def test_forward_backward_matches_brute_force():
    """On a tiny problem, FB must match the brute-force enumeration."""
    rng = np.random.default_rng(0)
    L = 6
    log_pi = np.log(np.array([0.7, 0.2, 0.1]))
    T = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    log_T = np.log(T)
    log_e = rng.normal(size=(L, 3))

    _, _, log_gamma, log_Z = forward_backward(log_e, log_pi, log_T)

    # Brute force
    from itertools import product
    log_p_paths = []
    for path in product(range(3), repeat=L):
        lp = log_pi[path[0]] + log_e[0, path[0]]
        for t in range(1, L):
            lp += log_T[path[t-1], path[t]] + log_e[t, path[t]]
        log_p_paths.append(lp)
    from scipy.special import logsumexp
    log_Z_brute = logsumexp(log_p_paths)
    assert abs(log_Z - log_Z_brute) < 1e-10

    # Posteriors normalise to 1 at every position
    gamma = np.exp(log_gamma)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Synthetic GLORI simulator (slim, just for testing parameter recovery)
# ---------------------------------------------------------------------------
def _simulate_transcript(L=800, p_m6a=0.05, artifact_rate=0.04, seed=0):
    r = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    seq = r.choice(bases, size=L, p=[0.27, 0.23, 0.23, 0.27])
    A_idx = np.where(seq == "A")[0]

    artifact = np.zeros(L, dtype=bool)
    n_starts = r.poisson(L * artifact_rate / 30.0)
    for _ in range(int(n_starts)):
        s = int(r.integers(0, L)); ln = int(r.integers(15, 60))
        artifact[s:min(s + ln, L)] = True

    truth = np.zeros(L, dtype=bool)
    for i in A_idx:
        if not artifact[i] and r.random() < p_m6a:
            truth[i] = True

    n = np.zeros(L, dtype=int)
    A_count = np.zeros(L, dtype=int)
    drach = np.zeros(L, dtype=bool)  # not used in this toy test
    for i in A_idx:
        n[i] = max(1, int(r.negative_binomial(8, 0.2)))
        if truth[i]:
            mu, kappa = 0.70, 30.0
        elif artifact[i]:
            mu, kappa = 0.25, 25.0
        else:
            mu, kappa = 0.02, 60.0
        a = mu * kappa; b = (1 - mu) * kappa
        p = r.beta(a, b)
        A_count[i] = r.binomial(n[i], p)

    return TranscriptObs(L=L, A_idx=A_idx.astype(np.int32),
                          A_count=A_count.astype(np.int32),
                          n=n.astype(np.int32), drach=drach), truth, artifact


def test_baum_welch_recovers_means():
    """BW from data-naive defaults should recover mu within a few percent."""
    transcripts = []
    for s in range(15):
        tr, _, _ = _simulate_transcript(L=800, seed=s + 1)
        transcripts.append(tr)
    params, _ = baum_welch(transcripts, HMMParams.default(), n_iter=15, verbose=False)
    # state ordering: 0=U, 1=M, 2=B
    assert abs(params.mu[0] - 0.02) < 0.01, f"mu_U off: {params.mu[0]}"
    assert abs(params.mu[1] - 0.70) < 0.05, f"mu_M off: {params.mu[1]}"
    assert abs(params.mu[2] - 0.25) < 0.05, f"mu_B off: {params.mu[2]}"
    # state ordering invariant
    assert params.mu[0] < params.mu[2] < params.mu[1]


def test_calling_rule_applies_constraints():
    """Calling rule must enforce tau, n>=n_min, and P(M) > P(B)."""
    L = 400
    tr, truth, _ = _simulate_transcript(L=L, seed=42)
    params = HMMParams.default()
    region = Region(name="t1", contig="chrTest", start=0, end=L)
    df = call_sites([(region, tr)], params, tau=0.8, n_min=10)
    assert "call" in df.columns
    called = df[df["call"]]
    # All calls must satisfy all three constraints
    assert (called["pM"] >= 0.8).all()
    assert (called["pM"] > called["pB"]).all()
    assert (called["n"] >= 10).all()


def test_hmm_params_serialise_roundtrip():
    p = HMMParams.default()
    d = p.to_dict()
    q = HMMParams.from_dict(d)
    assert np.allclose(p.pi, q.pi)
    assert np.allclose(p.T, q.T)
    assert np.allclose(p.mu, q.mu)
    assert np.allclose(p.kappa, q.kappa)
    assert np.allclose(p.p_drach, q.p_drach)
