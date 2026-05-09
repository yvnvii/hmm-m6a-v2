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

The repo includes a small synthetic dataset under `example/` so you can
verify your installation in a few seconds without downloading any
external reference. From the repo root:

```bash
hmm-m6a run \
  --bam      example/synthetic.bam \
  --fasta    example/synthetic.fa \
  --bed      example/synthetic.bed \
  -o         calls.tsv \
  --out-bed  calls.bed \
  --save-params fitted_params.json
```

This finishes in under 10 seconds and should call ~32 of the 39 known true
m⁶A sites with zero false positives. See `example/README.md` for how to
validate the calls against the bundled truth BED.

### Running on your own data

Replace the example paths with your own coordinate-sorted, indexed BAM,
matching reference FASTA, and a BED of regions to score (one HMM chain
per BED row):

```bash
hmm-m6a run \
  --bam      your_sample.bam \
  --fasta    your_reference.fa \
  --bed      your_regions.bed \
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

## Reference data

The package is reference-agnostic and works with any coordinate-matched
BAM/FASTA pair. For the mouse GLORI-seq experiments described in the
accompanying paper, the public reference files used are:

| file | source | size | what it is |
|---|---|---:|---|
| `mm10.fa` (+`.fai`) | UCSC: <https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz> | ~2.6 GB unzipped | mouse genome reference |
| `gencode.vM25.annotation.gtf` | GENCODE: <https://www.gencodegenes.org/mouse/release_M25.html> | ~1.3 GB unzipped | gene annotation (used to derive a regions BED) |

Neither file is bundled with this repository (both are too large and are
already publicly maintained at the URLs above). The convention we use is
to keep them under `data/` at the repo root; that path is gitignored, so
they'll never be accidentally committed.

```bash
mkdir -p data && cd data
wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
gunzip mm10.fa.gz
samtools faidx mm10.fa

wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz
gunzip gencode.vM25.annotation.gtf.gz
cd ..
```

For human GLORI-seq data, swap in GRCh38 and GENCODE human; the package
makes no mouse-specific assumptions.

### Building a regions BED from a GTF

`hmm-m6a` runs one HMM chain per row of the regions BED, so the BED
defines the spatial scope of the model. For the snRNA gene-body analysis
in the paper, we extracted RNU gene intervals from the GENCODE GTF:

```bash
awk -F'\t' '$3 == "gene" && $9 ~ /gene_type "snRNA"/ && $9 ~ /gene_name "Rnu/' \
    data/gencode.vM25.annotation.gtf \
| awk -F'\t' 'BEGIN{OFS="\t"} {
    match($9, /gene_name "[^"]+"/);
    name = substr($9, RSTART+11, RLENGTH-12);
    print $1, $4-1, $5, name, ".", $7
  }' > rnu_genes.bed

wc -l rnu_genes.bed     # sanity check
head rnu_genes.bed
```

For other transcript-type analyses, change the `$9 ~ /gene_type "..."/`
filter (`protein_coding`, `lncRNA`, `miRNA`, etc.).

### Two-stage workflow (recommended for many samples)

When processing several GLORI-seq samples, train once on a high-quality
reference sample and reuse the fitted parameters for the rest. This both
speeds things up and ensures all samples are scored on a common scale:

```bash
# 1. Pile up and train on a reference sample
hmm-m6a pileup --bam reference_sample.bam --fasta data/mm10.fa \
               --bed rnu_genes.bed -o reference.pileup.tsv
hmm-m6a train  --pileup reference.pileup.tsv -o fitted_params.json

# 2. Pile up and call on subsequent samples with the trained parameters
hmm-m6a pileup --bam sampleB.bam --fasta data/mm10.fa \
               --bed rnu_genes.bed -o sampleB.pileup.tsv
hmm-m6a call   --pileup sampleB.pileup.tsv --params fitted_params.json \
               -o sampleB.calls.tsv --bed sampleB.calls.bed
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

## Reproducing the paper benchmark

The synthetic-data validation notebook lives in `notebooks/m6a_hmm.ipynb`.
Open in Jupyter and run all cells; with the published seeds, all numbers
in the manuscript (Tables 1-5 and Figures 1-2) reproduce exactly.

## Layout

```
hmm-m6a/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/hmm_m6a/
│   ├── __init__.py        public API
│   ├── hmm.py             beta-binom PMF, FB, Viterbi, posterior
│   ├── mle.py             weighted MLE for (mu, kappa)
│   ├── train.py           Baum-Welch with state-ordering constraint
│   ├── pileup.py          BAM + FASTA -> per-A counts and DRACH
│   ├── calling.py         posterior thresholding
│   └── cli.py             `hmm-m6a` command-line interface
├── tests/
│   └── test_hmm.py
├── example/                       small synthetic dataset (committed)
│   ├── README.md
│   ├── build_example.py           regenerator script (deterministic)
│   ├── synthetic.fa  (+.fai)      tiny reference (5 transcripts, ~5 kb)
│   ├── synthetic.bam (+.bai)      simulated GLORI-seq reads
│   ├── synthetic.bed              regions to score
│   └── truth.bed                  known m6A sites for validation
├── data/                          gitignored; place mm10.fa, GTF here
│   └── .gitkeep
├── notebooks/
│   └── m6a_hmm.ipynb              validation notebook from the paper
└── docs/
    └── methods.md                 brief methods summary
```

## Citation

If you use this code, please cite:

> Ogawa, Y. *A Hidden Markov Model framework for m⁶A detection from GLORI-seq.* (2025).

## License

MIT, see `LICENSE`.
