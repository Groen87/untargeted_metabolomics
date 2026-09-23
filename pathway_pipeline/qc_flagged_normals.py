#!/usr/bin/env python3
"""QC diagnostics for the flagging stage: where do flagged NORMALS concentrate?

Reads the flagging outputs (``pathway_flags.csv``, ``sample_decisions.csv``)
plus the original feature matrix (for the Classification/Oordeel columns) and
reports:

1. Concentration: how many flagged (sample, pathway) pairs among normals come
   from the top-k normal samples. A few samples carrying most of the flags =
   mislabeled or QC-bad reference samples; a flat spread = the thresholds or
   the pathway set are the problem.
2. Per-normal sample detail: flagged-pathway count, top pathway, top excess.
3. Pathway miscalibration: pathways whose normal flag-rate far exceeds the
   configured percentile expectation (e.g. >3% at the 99th percentile) -- these
   inflate sample-level decisions and deserve individual inspection.

Usage:
    python -m pathway_pipeline.qc_flagged_normals \
        --flags outputs/pathway_pipeline/pathway_flags.csv \
        --input data/merged_data_with_classification.csv \
        [--threshold-percentile 99.0] [--top 15]
"""

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_COLUMN = "Classification"
OORDEEL_COLUMN = "Oordeel targeted"


def main():
    parser = argparse.ArgumentParser(
        description="Report where flagged normals concentrate (samples and pathways).")
    parser.add_argument("--flags", required=True,
                        help="Path to pathway_flags.csv from the pipeline.")
    parser.add_argument("--input", required=True,
                        help="Path to the original feature matrix CSV "
                             "(for the Classification / Oordeel targeted columns).")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Percentile used for the flag thresholds (default 99).")
    parser.add_argument("--top", type=int, default=15,
                        help="Rows to show per table (default 15).")
    args = parser.parse_args()

    flags = pd.read_csv(args.flags)
    logger.info(f"Loaded {len(flags)} (sample, pathway) pairs "
                f"({int(flags['flagged'].sum())} flagged) from {args.flags}")

    meta = pd.read_csv(args.input, index_col=0, low_memory=False)
    cls = pd.to_numeric(meta[CLASSIFICATION_COLUMN], errors="coerce")
    oor = pd.to_numeric(meta[OORDEEL_COLUMN], errors="coerce")
    normal_ids = meta.index[(cls == 0) & (oor == 0)]
    normal_set = set(normal_ids)
    logger.info(f"{len(normal_set)} normals in the input metadata.")

    flags["_is_normal"] = flags["sample_id"].isin(normal_set)
    normal_flags = flags[flags["_is_normal"] & flags["flagged"]]
    n_normal_pairs = int(flags["_is_normal"].sum())
    logger.info(f"{len(normal_flags)} flagged pairs among normals "
                f"({len(normal_flags) / max(n_normal_pairs, 1):.1%} of "
                f"{n_normal_pairs} normal pairs; expected ~"
                f"{100 - args.threshold_percentile:.0f}% at the "
                f"{args.threshold_percentile}th percentile).")

    if normal_flags.empty:
        logger.info("No flagged normals; nothing to inspect.")
        return

    # ------------------------------------------------------------------
    # 1. Concentration of flags across normal samples
    # ------------------------------------------------------------------
    per_sample = (normal_flags.groupby("sample_id")
                  .agg(n_flagged=("flagged", "sum"),
                       top_excess=("excess", "max"))
                  .sort_values("n_flagged", ascending=False))

    top = per_sample.head(args.top)
    rest = per_sample.iloc[args.top:]
    top_share = top["n_flagged"].sum() / len(normal_flags)
    logger.info(f"\nTop {len(top)} normal samples carry "
                f"{top['n_flagged'].sum()} of {len(normal_flags)} flags "
                f"({top_share:.0%} of all flagged-normal pairs).")
    logger.info("Flagged-pathway counts per normal sample (top "
                f"{args.top}):")
    for sample_id, row in top.iterrows():
        logger.info(f"  {sample_id}: {int(row['n_flagged'])} pathways "
                     f"(max excess {row['top_excess']:.1f})")
    if len(rest):
        logger.info(f"Remaining {len(rest)} flagged normals average "
                    f"{rest['n_flagged'].mean():.1f} flagged pathways each.")
    else:
        logger.info("All flagged normals are in the top list.")

    # ------------------------------------------------------------------
    # 2. Concentration across pathways (miscalibration check)
    # ------------------------------------------------------------------
    normal_pairs = flags[flags["_is_normal"]]
    per_pathway = (normal_pairs.groupby(["smp_id", "pathway_name"])
                   .agg(n_flagged=("flagged", "sum"),
                        n_pairs=("flagged", "size"),
                        max_excess=("excess", "max"))
                   .assign(flag_rate=lambda d: d["n_flagged"] / d["n_pairs"])
                   .sort_values("n_flagged", ascending=False))

    expected = 1.0 - args.threshold_percentile / 100.0
    miscalibrated = per_pathway[per_pathway["flag_rate"] > 3 * expected]
    logger.info(f"\n{len(miscalibrated)} of {len(per_pathway)} pathways "
                f"flag more than {3 * expected:.0%} of normals "
                f"(>3x the expected {expected:.0%}):")
    for (smp_id, name), row in miscalibrated.head(args.top).iterrows():
        logger.info(f"  {smp_id} ({name}): {int(row['n_flagged'])} of "
                    f"{int(row['n_pairs'])} normals flagged "
                    f"({row['flag_rate']:.1%}, max excess "
                    f"{row['max_excess']:.1f})")
    if len(miscalibrated) > args.top:
        logger.info(f"  ... and {len(miscalibrated) - args.top} more.")

    well_calibrated = per_pathway[per_pathway["flag_rate"] <= 3 * expected]
    logger.info(f"{len(well_calibrated)} pathways stay within 3x the "
                f"expected normal flag-rate.")

    # ------------------------------------------------------------------
    # 3. Overlap: are flagged normals also top-excess samples overall?
    # ------------------------------------------------------------------
    sample_max_excess = (flags.groupby("sample_id")["excess"]
                         .max().sort_values(ascending=False))
    flagged_normals = set(per_sample.index)
    top10_overall = set(sample_max_excess.head(10).index)
    overlap = flagged_normals & top10_overall
    logger.info(f"\n{len(overlap)} of the flagged normals are among the 10 "
                f"highest-excess samples overall: "
                f"{sorted(overlap) if overlap else 'none'}")

    logger.info("\nInterpretation: if a few normals carry most flags "
                "(section 1), inspect those samples (batch, run order, "
                "missingness) or re-label them -- cleaning them tightens "
                "every pathway threshold. If flags spread flatly, the "
                "pathway set or the threshold rule is the problem instead "
                "(see section 2).")


if __name__ == "__main__":
    main()
