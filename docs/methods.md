# Methods summary

This document is a short reference for what the package implements. For the
full derivation see Sections 2 and 3 of the accompanying manuscript.

## Model

A per-nucleotide hidden Markov model with three states:

- `U` (unmethylated / clean background): low A-retention.
- `M` (true m⁶A): high A-retention.
- `B` (artifact / poor conversion): intermediate A-retention.

State indices in the code are `0 = U`, `1 = M`, `2 = B`. The state-ordering
constraint after each Baum-Welch M-step,

```
mu_U < mu_B < mu_M,
```

keeps the index → semantics mapping stable across runs.

## Emissions

At every adenosine `t`, the emission combines a beta-binomial conversion-count
term and a binary DRACH motif term:

```
log P(O_t | Z_t = z)
  = log BetaBin(A_t | n_t, mu_z * kappa_z, (1 - mu_z) * kappa_z)
    + log P(DRACH | Z_t = z) ** drach[t]
    + log (1 - P(DRACH | Z_t = z)) ** (1 - drach[t])
```

At every non-adenosine position the emission is neutral (`log P = 0`). The
chain still advances, so the hidden state must persist across the physical
distance between informative adenosines. This is what lets the artifact
state model regional poor conversion rather than a run of consecutive
adenosines.

## Inference

The `hmm` module implements:

- `forward_backward` — log-space, manual log-sum-exp inner loop (faster
  than `scipy.special.logsumexp` on K=3 matrices).
- `viterbi` — most-likely state path.
- `posterior_states` — convenience wrapper returning per-position
  `gamma[t, z]` and `log_Z`.

## Training

The `train` module implements `baum_welch`. Key choices:

- E-step posteriors `gamma_t(z)` and pairwise `xi_t(a, b)` summed over `t`.
- M-step:
  - π and T: Dirichlet pseudocount λ (default 1).
  - DRACH emission: Laplace-smoothed binary update.
  - Beta-binomial emission `(mu_z, kappa_z)`: **weighted numerical MLE**
    via L-BFGS-B in `(logit μ, log κ)`. The moment-based update used in
    earlier drafts is biased because the variance of the observed
    proportion `p_t = A_t/n_t` is *not* equal to the latent beta
    variance — it has additional 1/n_t binomial noise.
- State-ordering constraint applied after each M-step.

## Calling

A site at adenosine `t` is called as m⁶A if all three of:

```
P(M | O) >= tau          (default tau = 0.8)
P(M | O) > P(B | O)
n_t >= n_min             (default n_min = 10)
```

The `P(M) > P(B)` constraint prevents the model from calling sites where
the posterior is split between methylation and artifact: such sites
typically arise inside artifact regions and are exactly what we want to
suppress.

## BAM pileup

`pileup.py` walks pysam's pileup over each requested region and tallies:

- `A_t` = number of reads with aligned base `A` at the reference adenosine
- `G_t` = number of reads with aligned base `G`
- `n_t = A_t + G_t`

Bases C, T, indels, and skips are not counted. Reads below `--min-mapq` or
columns below `--min-baseq` are dropped. For `-` strand regions, the
reference is reverse-complemented and read bases are complemented before
the count.

DRACH motif flags are computed from the *reference* 5-mer centred on each
adenosine, on the transcript strand:
`D ∈ {A, G, T}, R ∈ {A, G}, A, C, H ∈ {A, C, T}`.

## Reproducibility

The validation notebook `notebooks/m6a_hmm.ipynb` reproduces every number
in the paper. Random seeds are fixed; the simulator and trainer share a
single master seed.
