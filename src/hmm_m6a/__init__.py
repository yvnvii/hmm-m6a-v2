"""hmm_m6a: per-nucleotide hidden Markov model for m6A detection from GLORI-seq."""
from hmm_m6a.hmm import (
    HMMParams,
    forward_backward,
    posterior_states,
    viterbi,
    emission_logprob,
)
from hmm_m6a.train import baum_welch
from hmm_m6a.pileup import pileup_bam, Region, AdenosineRecord
from hmm_m6a.calling import call_sites

__version__ = "0.1.0"
__all__ = [
    "HMMParams",
    "forward_backward",
    "posterior_states",
    "viterbi",
    "emission_logprob",
    "baum_welch",
    "pileup_bam",
    "Region",
    "AdenosineRecord",
    "call_sites",
    "__version__",
]
