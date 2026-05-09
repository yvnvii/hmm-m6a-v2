"""Command-line interface.

Subcommands
-----------
    hmm-m6a pileup   --bam ... --fasta ... [--bed ...] -o pileup.tsv
    hmm-m6a train    --pileup pileup.tsv -o params.json
    hmm-m6a call     --pileup pileup.tsv --params params.json -o calls.tsv [--bed calls.bed]
    hmm-m6a run      --bam ... --fasta ... [--bed ...] -o calls.tsv

`run` does pileup + train + call in one invocation. Use the explicit
subcommands when you want to inspect intermediates or reuse pretrained
parameters across samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from hmm_m6a import __version__
from hmm_m6a.calling import call_sites, df_to_bed
from hmm_m6a.hmm import HMMParams
from hmm_m6a.pileup import (
    Region,
    pileup_bam,
    regions_from_bam_contigs,
    regions_from_bed,
)
from hmm_m6a.train import TranscriptObs, baum_welch


# ---------------------------------------------------------------------------
# Pileup TSV (de)serialisation
# ---------------------------------------------------------------------------
def _pileup_to_tsv(
    region_obs: Sequence[Tuple[Region, TranscriptObs]],
    out_path: str,
) -> None:
    """One row per adenosine with all the data the HMM needs.

    Columns: region, contig, strand, pos0_in_region, n, A, drach, region_L,
             region_start, region_end
    """
    rows: List[dict] = []
    for region, tr in region_obs:
        for i in tr.A_idx:
            i = int(i)
            rows.append({
                "region": region.name,
                "contig": region.contig,
                "strand": region.strand,
                "region_start": region.start,
                "region_end": region.end,
                "region_L": tr.L,
                "pos0_in_region": i,
                "A": int(tr.A_count[i]),
                "n": int(tr.n[i]),
                "drach": int(bool(tr.drach[i])),
            })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def _tsv_to_pileup(path: str) -> List[Tuple[Region, TranscriptObs]]:
    """Reverse of `_pileup_to_tsv`."""
    df = pd.read_csv(path, sep="\t")
    out: List[Tuple[Region, TranscriptObs]] = []
    for region_name, sub in df.groupby("region", sort=False):
        sub = sub.sort_values("pos0_in_region")
        contig = sub["contig"].iloc[0]
        strand = sub["strand"].iloc[0]
        L = int(sub["region_L"].iloc[0])
        start = int(sub["region_start"].iloc[0])
        end = int(sub["region_end"].iloc[0])
        A_idx = sub["pos0_in_region"].astype(np.int32).to_numpy()
        A_count = np.zeros(L, dtype=np.int32)
        n = np.zeros(L, dtype=np.int32)
        drach = np.zeros(L, dtype=bool)
        A_count[A_idx] = sub["A"].astype(np.int32).to_numpy()
        n[A_idx] = sub["n"].astype(np.int32).to_numpy()
        drach[A_idx] = sub["drach"].astype(bool).to_numpy()
        region = Region(name=str(region_name), contig=contig, start=start, end=end, strand=strand)
        out.append((region, TranscriptObs(L=L, A_idx=A_idx, A_count=A_count, n=n, drach=drach)))
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def _resolve_regions(bam: str, bed: str | None) -> List[Region]:
    if bed:
        return regions_from_bed(bed)
    return regions_from_bam_contigs(bam)


def cmd_pileup(args: argparse.Namespace) -> int:
    regions = _resolve_regions(args.bam, args.bed)
    region_obs = pileup_bam(
        args.bam, args.fasta, regions,
        min_mapq=args.min_mapq, min_baseq=args.min_baseq,
        show_progress=not args.quiet,
    )
    _pileup_to_tsv(region_obs, args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    region_obs = _tsv_to_pileup(args.pileup)
    transcripts = [tr for _, tr in region_obs]
    init = HMMParams.default()
    params, history = baum_welch(
        transcripts, init,
        n_iter=args.n_iter, lam=args.lam, tol=args.tol,
        verbose=not args.quiet,
    )
    out = {
        "params": params.to_dict(),
        "history": {"logZ": list(map(float, history.logZ))},
        "version": __version__,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    region_obs = _tsv_to_pileup(args.pileup)
    pj = json.loads(Path(args.params).read_text())
    params = HMMParams.from_dict(pj["params"])
    df = call_sites(region_obs, params, tau=args.tau, n_min=args.n_min)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {args.output} ({int(df['call'].sum()) if len(df) else 0} sites called)", file=sys.stderr)
    if args.bed:
        Path(args.bed).write_text(df_to_bed(df))
        print(f"wrote {args.bed}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end: pileup -> train -> call, all in memory."""
    regions = _resolve_regions(args.bam, args.bed)
    region_obs = pileup_bam(
        args.bam, args.fasta, regions,
        min_mapq=args.min_mapq, min_baseq=args.min_baseq,
        show_progress=not args.quiet,
    )

    if args.params:
        pj = json.loads(Path(args.params).read_text())
        params = HMMParams.from_dict(pj["params"])
        if not args.quiet:
            print(f"loaded pretrained parameters from {args.params}", file=sys.stderr)
    else:
        transcripts = [tr for _, tr in region_obs]
        if not transcripts:
            print("no regions had any pileup data; nothing to train on.", file=sys.stderr)
            return 1
        params, _ = baum_welch(
            transcripts, HMMParams.default(),
            n_iter=args.n_iter, lam=args.lam, tol=args.tol,
            verbose=not args.quiet,
        )
        if args.save_params:
            Path(args.save_params).write_text(
                json.dumps({"params": params.to_dict(), "version": __version__}, indent=2)
            )
            if not args.quiet:
                print(f"saved fitted parameters to {args.save_params}", file=sys.stderr)

    df = call_sites(region_obs, params, tau=args.tau, n_min=args.n_min)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {args.output} ({int(df['call'].sum()) if len(df) else 0} sites called)", file=sys.stderr)
    if args.out_bed:
        Path(args.out_bed).write_text(df_to_bed(df))
        print(f"wrote {args.out_bed}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def _add_pileup_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bam", required=True, help="coordinate-sorted, indexed BAM")
    p.add_argument("--fasta", required=True, help="indexed reference FASTA")
    p.add_argument("--bed", default=None,
                   help="BED of regions to score; one HMM chain per row "
                        "(if omitted, every BAM contig under 2 Mb is used)")
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--min-baseq", type=int, default=20)


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--lam", type=float, default=1.0, help="Dirichlet pseudocount")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="relative log-likelihood convergence tolerance")


def _add_call_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tau", type=float, default=0.8,
                   help="posterior threshold for the M state")
    p.add_argument("--n-min", type=int, default=10,
                   help="minimum informative coverage")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hmm-m6a",
        description="Per-nucleotide HMM for m6A detection from GLORI-seq.",
    )
    p.add_argument("--version", action="version", version=f"hmm-m6a {__version__}")
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pileup", help="BAM + FASTA -> per-A pileup TSV")
    _add_pileup_args(sp)
    sp.add_argument("-o", "--output", required=True, help="output TSV path")
    sp.set_defaults(func=cmd_pileup)

    st = sub.add_parser("train", help="pileup TSV -> Baum-Welch JSON parameters")
    st.add_argument("--pileup", required=True)
    _add_train_args(st)
    st.add_argument("-o", "--output", required=True, help="output JSON path")
    st.set_defaults(func=cmd_train)

    sc = sub.add_parser("call", help="pileup TSV + params -> per-site calls")
    sc.add_argument("--pileup", required=True)
    sc.add_argument("--params", required=True)
    _add_call_args(sc)
    sc.add_argument("-o", "--output", required=True, help="output TSV path")
    sc.add_argument("--bed", default=None, help="optional BED6 of called sites")
    sc.set_defaults(func=cmd_call)

    sr = sub.add_parser("run", help="end-to-end: BAM -> calls (one command)")
    _add_pileup_args(sr)
    _add_train_args(sr)
    _add_call_args(sr)
    sr.add_argument("--params", default=None,
                    help="JSON of pretrained parameters; skips training if given")
    sr.add_argument("--save-params", default=None,
                    help="write the fitted parameters here (only when training)")
    sr.add_argument("-o", "--output", required=True, help="output TSV of all per-A scores")
    sr.add_argument("--out-bed", default=None, help="optional BED6 of called sites")
    sr.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
