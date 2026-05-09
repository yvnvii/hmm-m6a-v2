"""Generate a small, reproducible example dataset for hmm-m6a.

Produces three files in this directory:

    synthetic.fa       reference FASTA (5 short transcripts, ~5 kb total)
    synthetic.fa.fai   FASTA index (created by pysam.faidx)
    synthetic.bam      coordinate-sorted, indexed BAM with simulated GLORI reads
    synthetic.bam.bai
    synthetic.bed      BED6 of the 5 transcript intervals
    truth.bed          BED6 of the true m6A sites (for validation)

Design
------
We simulate a tiny GLORI-seq experiment with three kinds of adenosines so
the HMM has something to discriminate:

    * "Unmethylated" adenosines: ~98 % of reads convert (A -> G).
      Expected A-rate ~0.02. Most adenosines are this kind.
    * "True m6A" sites:  ~30 % of reads convert.
      Expected A-rate ~0.70. Placed preferentially at DRACH motifs.
    * "Artifact" regions: ~75 % of reads convert in a localized stretch.
      Expected A-rate ~0.25. Simulates a poorly converted patch.

A working caller should call most true m6A sites, no unmethylated sites,
and no artifact-region adenosines (despite their elevated A-rate).

Run with:
    python build_example.py

The output files are committed to the repo so users don't need to run this.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pysam

OUT_DIR = Path(__file__).resolve().parent
SEED = 20251108

# Five short, named transcripts.  Lengths a few kb each, total ~5 kb.
TRANSCRIPTS = [
    ("synth_tx1", "chr1_synth", 0, 1200, "+"),
    ("synth_tx2", "chr2_synth", 0, 1100, "+"),
    ("synth_tx3", "chr3_synth", 0,  900, "-"),  # minus-strand
    ("synth_tx4", "chr4_synth", 0,  950, "+"),
    ("synth_tx5", "chr5_synth", 0,  850, "+"),
]


# ---------------------------------------------------------------------------
# DRACH helper (matches the package's DRACH definition)
# ---------------------------------------------------------------------------
DRACH_D = set("AGT"); DRACH_R = set("AG"); DRACH_H = set("ACT")


def is_drach(seq: str, i: int) -> bool:
    if i < 2 or i > len(seq) - 3:
        return False
    a, b, c, d, e = seq[i - 2 : i + 3]
    return a in DRACH_D and b in DRACH_R and c == "A" and d == "C" and e in DRACH_H


_RC = str.maketrans("ACGT", "TGCA")
def revcomp(s: str) -> str:
    return s.translate(_RC)[::-1]


# ---------------------------------------------------------------------------
# Step 1: build a reference FASTA with controlled DRACH placement
# ---------------------------------------------------------------------------
def build_reference(rng: random.Random) -> dict[str, str]:
    """Return {contig_name: plus_strand_sequence}.

    We bias the sequence to have a reasonable number of DRACH motifs by
    occasionally inserting "GGACT" motifs (a canonical DRACH).
    """
    refs: dict[str, str] = {}
    bases = "ACGT"
    for tx_name, contig, start, end, strand in TRANSCRIPTS:
        L = end - start
        # Random sequence
        seq = [rng.choices(bases, weights=[27, 23, 23, 27])[0] for _ in range(L)]
        # Sprinkle in some DRACH-friendly 5-mers ("GGACT" - canonical DRACH)
        n_drach = max(3, L // 150)
        positions = rng.sample(range(2, L - 3), n_drach)
        for p in positions:
            for j, b in enumerate("GGACT"):
                seq[p - 2 + j] = b
        refs[contig] = "".join(seq)
    return refs


# ---------------------------------------------------------------------------
# Step 2: choose where the true m6A sites and artifact regions live
# ---------------------------------------------------------------------------
def annotate(refs: dict[str, str], rng: random.Random):
    """For each transcript, pick:
        - true m6A adenosines (preferentially at DRACH motifs)
        - one artifact region  (geometric length, intermediate A-rate)

    Returns:
        truth_m6a: dict[contig -> set of plus-strand positions]
        artifact:  dict[contig -> set of plus-strand positions in artifact regions]
    """
    truth_m6a: dict[str, set[int]] = {}
    artifact: dict[str, set[int]] = {}

    for tx_name, contig, start, end, strand in TRANSCRIPTS:
        # Sequence on the *transcript strand* (what the model "sees")
        plus_seq = refs[contig]
        tx_seq = revcomp(plus_seq) if strand == "-" else plus_seq
        L = len(tx_seq)

        # Find DRACH adenosines in transcript-strand coordinates
        drach_pos = [i for i in range(L) if tx_seq[i] == "A" and is_drach(tx_seq, i)]
        nondrach_pos = [
            i for i in range(L) if tx_seq[i] == "A" and not is_drach(tx_seq, i)
        ]

        # Pick true m6A sites: most at DRACH, a few at non-DRACH
        n_drach_m6a = min(len(drach_pos), max(2, len(drach_pos) // 3))
        chosen = set(rng.sample(drach_pos, n_drach_m6a))
        if nondrach_pos:
            extras = rng.sample(nondrach_pos, min(1, len(nondrach_pos)))
            chosen.update(extras)

        # One artifact region per transcript, ~60-100 nt long
        art_start_tx = rng.randint(L // 4, 3 * L // 4)
        art_len = rng.randint(60, 100)
        art_end_tx = min(L, art_start_tx + art_len)
        artifact_tx = set(range(art_start_tx, art_end_tx))

        # Don't let true m6A sites land inside artifact regions (clean truth)
        chosen -= artifact_tx

        # Convert transcript-strand positions back to plus-strand positions
        if strand == "-":
            chosen = {L - 1 - i for i in chosen}
            artifact_tx = {L - 1 - i for i in artifact_tx}

        truth_m6a[contig] = chosen
        artifact[contig] = artifact_tx

    return truth_m6a, artifact


# ---------------------------------------------------------------------------
# Step 3: simulate reads
# ---------------------------------------------------------------------------
def simulate_reads(
    refs: dict[str, str],
    truth_m6a: dict[str, set[int]],
    artifact: dict[str, set[int]],
    rng: np.random.Generator,
    out_unsorted_bam: Path,
    coverage: int = 30,
    read_len: int = 100,
) -> None:
    """Walk every position; for each adenosine, draw a state-dependent A-rate
    and emit `coverage` reads spanning a window centred on it.

    GLORI semantics: with probability `1 - p_A` the A is converted to G;
    otherwise it stays an A.  For other reference bases we just copy the
    reference (no errors).
    """
    header = {
        "HD": {"VN": "1.0", "SO": "coordinate"},
        "SQ": [
            {"SN": contig, "LN": len(seq)} for contig, seq in refs.items()
        ],
    }
    rid = 0

    # State-conditional A-retention parameters (mu, kappa for beta-binomial)
    PARAMS = {
        "U": (0.02, 60.0),  # well-converted
        "M": (0.70, 30.0),  # true m6A
        "B": (0.25, 25.0),  # artifact
    }

    with pysam.AlignmentFile(out_unsorted_bam, "wb", header=header) as out:
        for tx_name, contig, start, end, strand in TRANSCRIPTS:
            seq = refs[contig]
            L = len(seq)
            true_set = truth_m6a[contig]
            art_set = artifact[contig]

            # For every adenosine on the *plus strand*, decide state and per-read
            # base. Reads themselves are independent uniform-tile reads.
            # We pre-compute per-position p_A so reads can sample from it.
            p_A = np.zeros(L)
            for i, base in enumerate(seq):
                if base != "A":
                    continue
                if i in true_set:
                    mu, kappa = PARAMS["M"]
                elif i in art_set:
                    mu, kappa = PARAMS["B"]
                else:
                    mu, kappa = PARAMS["U"]
                a, b = mu * kappa, (1 - mu) * kappa
                # Position-specific A-retention probability (latent beta draw)
                p_A[i] = float(rng.beta(a, b))

            # Tile reads across the transcript so coverage ~= `coverage`
            # Each starting position gets `coverage * read_len / L` * step reads.
            n_starts = max(1, (L - read_len) // 5)
            reads_per_start = max(1, int(round(coverage * 5 / read_len)))
            for step in range(n_starts):
                rstart = step * 5
                rend = rstart + read_len
                if rend > L:
                    break
                ref_chunk = seq[rstart:rend]
                for _ in range(reads_per_start):
                    bases = list(ref_chunk)
                    for k, b in enumerate(bases):
                        if b == "A":
                            i = rstart + k
                            # A retained with probability p_A[i], else converted to G
                            if rng.random() > p_A[i]:
                                bases[k] = "G"
                    qseq = "".join(bases)

                    a = pysam.AlignedSegment(out.header)
                    a.query_name = f"r{rid}"; rid += 1
                    a.flag = 0
                    a.reference_id = list(refs).index(contig)
                    a.reference_start = rstart
                    a.mapping_quality = 60
                    a.cigar = ((0, read_len),)  # all M
                    a.query_sequence = qseq
                    a.query_qualities = [40] * read_len
                    out.write(a)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    refs = build_reference(py_rng)
    truth_m6a, artifact = annotate(refs, py_rng)

    # Write the FASTA
    fasta_path = OUT_DIR / "synthetic.fa"
    with open(fasta_path, "w") as f:
        for contig, seq in refs.items():
            f.write(f">{contig}\n")
            # 70-char wrap is conventional
            for i in range(0, len(seq), 70):
                f.write(seq[i : i + 70] + "\n")
    pysam.faidx(str(fasta_path))
    print(f"wrote {fasta_path} ({sum(len(s) for s in refs.values())} bp across {len(refs)} contigs)")

    # Write the regions BED (the input to hmm-m6a)
    bed_path = OUT_DIR / "synthetic.bed"
    with open(bed_path, "w") as f:
        for tx_name, contig, start, end, strand in TRANSCRIPTS:
            f.write(f"{contig}\t{start}\t{end}\t{tx_name}\t.\t{strand}\n")
    print(f"wrote {bed_path} ({len(TRANSCRIPTS)} regions)")

    # Write a truth BED so users can validate calls afterwards
    truth_path = OUT_DIR / "truth.bed"
    n_truth = 0
    with open(truth_path, "w") as f:
        for tx_name, contig, start, end, strand in TRANSCRIPTS:
            for plus_pos in sorted(truth_m6a[contig]):
                f.write(f"{contig}\t{plus_pos}\t{plus_pos+1}\ttrue_m6a\t1000\t{strand}\n")
                n_truth += 1
    print(f"wrote {truth_path} ({n_truth} true m6A sites)")

    # Simulate reads, sort, index
    unsorted = OUT_DIR / "synthetic.unsorted.bam"
    simulate_reads(refs, truth_m6a, artifact, np_rng, unsorted, coverage=30, read_len=100)
    bam_path = OUT_DIR / "synthetic.bam"
    pysam.sort("-o", str(bam_path), str(unsorted))
    pysam.index(str(bam_path))
    os.unlink(unsorted)

    n_reads = pysam.AlignmentFile(str(bam_path)).count()
    print(f"wrote {bam_path} ({n_reads} reads)")
    print(f"wrote {bam_path}.bai")
    print()
    print("Done. Try:")
    print()
    print("  hmm-m6a run \\")
    print("    --bam     example/synthetic.bam \\")
    print("    --fasta   example/synthetic.fa \\")
    print("    --bed     example/synthetic.bed \\")
    print("    -o        calls.tsv \\")
    print("    --out-bed calls.bed \\")
    print("    --save-params fitted_params.json")


if __name__ == "__main__":
    main()
