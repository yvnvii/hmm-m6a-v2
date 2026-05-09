# hmm-m6a

A per-nucleotide hidden Markov model for m⁶A detection from GLORI-seq data.

The model defines a hidden state at every nucleotide position with three
states (`U` unmethylated, `M` methylated, `B` artifact / poor conversion),
beta-binomial emissions to absorb overdispersion in conversion read counts,
a DRACH sequence-context emission channel, and a state-ordering constraint
that preserves biological interpretability after Baum-Welch training.
See the accompanying paper for details.

## Installation

```bash
git clone https://github.com/yvnvii/hmm-m6a.git
cd hmm-m6a
pip install -e .
```

Requires Python ≥ 3.10. Dependencies (`numpy`, `pandas`, `scipy`, `pysam`,
`pyfaidx`, `tqdm`) install automatically.

## Quick start

End-to-end pipeline on a coordinate-sorted, indexed BAM:

```bash
hmm-m6a run \
  --bam      cd4.rnu.RnuGenes.noAG.bam \
  --fasta    mm10.fa \
  --bed      rnu_genes.bed \
  -o         calls.tsv \
  --out-bed  calls.bed \
  --save-params fitted_params.json
```

This will:

1. **Pile up** A-like and G-like read counts at every reference adenosine
   inside each interval in the BED file.
2. **Train** the HMM via Baum-Welch on those pileups, applying the
   state-ordering constraint after each M-step.
3. **Call** m⁶A sites where `P(M | O) ≥ 0.8`, `P(M | O) > P(B | O)`, and
   `n_t ≥ 10`.
4. Write a per-site TSV (`calls.tsv`), a BED6 of called sites
   (`calls.bed`), and the fitted parameters (`fitted_params.json`).

## Worked example: the `cd4.rnu.RnuGenes.noAG.bam` test file

The example BAM is a coordinate-sorted, mouse mm10-aligned GLORI-seq file
restricted to RNU gene bodies after AG-conversion filtering. To run the
caller on it:

```bash
# 1. Make sure the BAM is indexed and the reference FASTA is too.
samtools index cd4.rnu.RnuGenes.noAG.bam
samtools faidx mm10.fa

# 2. Provide a BED file of the RNU gene intervals (one HMM chain per row).
#    If you already have rnu_genes.bed from your upstream pipeline, use it.
#    Otherwise extract from a GTF, for example:
#    awk '$3=="gene" && /gene_biotype "snRNA"/' mm10.gtf \
#      | awk -v OFS='\t' '{print $1,$4-1,$5,$10,".",$7}' \
#      | tr -d '";'  > rnu_genes.bed

# 3. End-to-end caller
hmm-m6a run \
  --bam      cd4.rnu.RnuGenes.noAG.bam \
  --fasta    mm10.fa \
  --bed      rnu_genes.bed \
  -o         cd4_rnu_calls.tsv \
  --out-bed  cd4_rnu_calls.bed \
  --save-params cd4_rnu_params.json
```

If you don't have a BED yet, you can omit `--bed` and the tool will treat
each BAM contig (under 2 Mb) as one HMM chain. That's only sensible if your
BAM is already restricted to short regions, as the RNU-only file is.

```bash
hmm-m6a run --bam cd4.rnu.RnuGenes.noAG.bam --fasta mm10.fa -o cd4_rnu_calls.tsv
```

### Two-stage workflow (recommended for many samples)

Train once on a high-quality reference sample, then apply to multiple
samples without retraining:

```bash
# 1. Pile up and train on sample A
hmm-m6a pileup --bam sampleA.bam --fasta mm10.fa --bed rnu.bed -o sampleA.pileup.tsv
hmm-m6a train  --pileup sampleA.pileup.tsv -o params.json

# 2. Pile up and call on sample B with the trained parameters
hmm-m6a pileup --bam sampleB.bam --fasta mm10.fa --bed rnu.bed -o sampleB.pileup.tsv
hmm-m6a call   --pileup sampleB.pileup.tsv --params params.json -o sampleB.calls.tsv \
               --bed sampleB.calls.bed
```

## Output

`calls.tsv` has one row per reference adenosine with columns:

| column   | meaning |
|----------|---------|
| region   | name of the interval from the BED |
| contig   | reference contig |
| pos0     | 0-based reference position |
| strand   | strand of the region |
| n        | informative coverage `A_t + G_t` |
| A        | A-like read count `A_t` |
| rate     | empirical A-rate `A / max(n, 1)` |
| drach    | DRACH motif at this adenosine (bool) |
| pU       | posterior `P(Z_t = U | O)` |
| pM       | posterior `P(Z_t = M | O)` |
| pB       | posterior `P(Z_t = B | O)` |
| call     | true if the site passes all three calling criteria |

The BED6 contains only called sites. The BED `score` column is
`round(1000 × pM)` so it lands in BED's 0-1000 range and can be loaded
directly into IGV.

## Python API

```python
from hmm_m6a import (
    HMMParams, baum_welch, pileup_bam, call_sites, Region,
)

regions = [Region("Rnu1b1", "chr3", 84_280_000, 84_280_500, strand="+")]
region_obs = pileup_bam("sample.bam", "mm10.fa", regions)
transcripts = [tr for _, tr in region_obs]

params, history = baum_welch(transcripts, HMMParams.default(), n_iter=20)
df = call_sites(region_obs, params, tau=0.8, n_min=10)
print(df[df["call"]])
```

## Reproducing the manuscript benchmark validation

The synthetic-data validation notebook lives in `notebooks/m6a_hmm.ipynb`.
Open in Jupyter and run all cells; with the published seeds, all numbers
in the manuscript reproduce exactly.


