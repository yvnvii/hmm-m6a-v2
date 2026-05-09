# Example: synthetic GLORI-seq data

A small, self-contained example so you can verify your installation
without downloading any external data.

## Files

| file | what it is |
|---|---|
| `synthetic.fa` (+`.fai`) | 5 short transcripts, ~5 kb total, with DRACH motifs sprinkled in |
| `synthetic.bam` (+`.bai`) | 1{,}800 simulated GLORI-seq reads (~30× coverage) |
| `synthetic.bed` | 5 BED6 intervals, one per transcript (input to the caller) |
| `truth.bed` | 39 true m⁶A sites (used for validation only — NOT input to the caller) |
| `build_example.py` | the script that generates everything above (already committed; rerun only if you want to regenerate) |

The BAM contains three kinds of adenosines:

- **Unmethylated**: most positions, low A-retention (~2 %).
- **True m⁶A**: 39 sites, mostly at DRACH motifs, high A-retention (~70 %).
- **Artifact regions**: one ~70-nt patch per transcript with intermediate
  A-retention (~25 %) — the kind of signal the model's `B` state is
  designed to absorb.

A correctly working caller should call most of the 39 true m⁶A sites,
none of the unmethylated sites, and none of the artifact-region adenosines
(despite their elevated A-rate).

## Run

From the repo root:

```bash
hmm-m6a run \
  --bam     example/synthetic.bam \
  --fasta   example/synthetic.fa \
  --bed     example/synthetic.bed \
  -o        calls.tsv \
  --out-bed calls.bed \
  --save-params fitted_params.json
```

The whole thing finishes in under 10 seconds. Expected result:

```
pileup: 100% [5/5 regions]
iter 1   logZ=-2310.12   mu=[0.018, 0.707, 0.265]   ...
...
converged at iter 6
wrote calls.tsv  (32 sites called)
wrote calls.bed
```

## Validate

Compare the called sites to the truth:

```python
import pandas as pd
truth = pd.read_csv("example/truth.bed", sep="\t", header=None,
                   names=["contig","start","end","name","score","strand"])
calls = pd.read_csv("calls.tsv", sep="\t")

true_sites   = set(zip(truth["contig"], truth["start"]))
called_sites = set(zip(calls.loc[calls.call, "contig"],
                       calls.loc[calls.call, "pos0"]))
tp = len(true_sites & called_sites)
fp = len(called_sites - true_sites)
fn = len(true_sites - called_sites)
print(f"TP={tp}  FP={fp}  FN={fn}  "
      f"precision={tp/max(tp+fp,1):.3f}  recall={tp/max(tp+fn,1):.3f}")
```

With the committed seed (`SEED=20251108` in `build_example.py`) you should
see something close to:

```
TP=32  FP=0  FN=7  precision=1.000  recall=0.821
```

## Regenerating

If you want to regenerate the dataset (e.g. after editing `build_example.py`):

```bash
cd example
python build_example.py
```

The script is deterministic — same seed produces the same files.
