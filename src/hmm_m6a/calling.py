"""Apply the m6A calling rule from Methods Section 2.12.

A site at adenosine position t is called if all three of:
    P(M | O) >= tau          (default tau = 0.8)
    P(M | O) > P(B | O)
    n_t >= n_min             (default n_min = 10)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from hmm_m6a.hmm import HMMParams, posterior_states
from hmm_m6a.pileup import AdenosineRecord, Region
from hmm_m6a.train import TranscriptObs


def call_sites(
    region_obs: Sequence[Tuple[Region, TranscriptObs]],
    params: HMMParams,
    tau: float = 0.8,
    n_min: int = 10,
    require_M_gt_B: bool = True,
) -> pd.DataFrame:
    """Score every adenosine in every region, then apply the calling rule.

    Returns a DataFrame with one row per adenosine, including posterior
    probabilities for every state and a `call` column.
    """
    rows: List[dict] = []
    for region, tr in region_obs:
        if tr.A_idx.size == 0:
            continue
        gamma, _ = posterior_states(
            tr.A_idx, tr.A_count, tr.n, tr.drach, tr.L, params
        )
        for i in tr.A_idx:
            i = int(i)
            n = int(tr.n[i])
            kA = int(tr.A_count[i])
            # Map back to genomic coordinate. For minus-strand regions we
            # stored sequence on the transcript strand, so position 0 of the
            # buffer is region.end - 1 in genomic coordinates.
            if region.strand == "-":
                pos0 = region.end - 1 - i
            else:
                pos0 = region.start + i
            rows.append({
                "region": region.name,
                "contig": region.contig,
                "pos0": pos0,
                "strand": region.strand,
                "n": n,
                "A": kA,
                "rate": kA / max(n, 1),
                "drach": bool(tr.drach[i]),
                "pU": float(gamma[i, 0]),
                "pM": float(gamma[i, 1]),
                "pB": float(gamma[i, 2]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df["call"] = []
        return df

    call = (df["pM"] >= tau) & (df["n"] >= n_min)
    if require_M_gt_B:
        call &= df["pM"] > df["pB"]
    df["call"] = call.values
    return df


def df_to_bed(df: pd.DataFrame) -> str:
    """Format a call table as BED6 lines (called sites only).

    Score column is round(1000 * pM) so it lands in BED's 0-1000 range.
    """
    called = df[df["call"]] if "call" in df.columns else df
    lines = []
    for _, r in called.iterrows():
        score = int(round(1000.0 * float(r["pM"])))
        score = max(0, min(1000, score))
        name = f"m6A;n={int(r['n'])};A={int(r['A'])};drach={int(bool(r['drach']))}"
        lines.append(
            f"{r['contig']}\t{int(r['pos0'])}\t{int(r['pos0']) + 1}\t"
            f"{name}\t{score}\t{r['strand']}"
        )
    return "\n".join(lines) + ("\n" if lines else "")
