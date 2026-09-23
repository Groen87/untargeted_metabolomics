#!/usr/bin/env python3
"""QC diagnostics for the z-score stage.

Reads the pipeline outputs (``metabolite_zscores.csv`` plus the original
feature matrix for the Classification/Oordeel columns) and reports where
extreme z-scores concentrate:

1. Calibration check: the normals' per-feature median and IQR must be ~0 and
   ~1 by construction; large deviations point to a wrong normal subset.
2. Extremes by feature: features with the most |z| > threshold among normals
   (unstable features).
3. Extremes by sample: normal samples with the most |z| > threshold
   (samples worth a QC look).

Usage:
    python -m pathway_pipeline.qc_extremes \
        --zscores outputs/pathway_pipeline/metabolite_zscores.csv \
        --input data/merged_data_with_classification.csv \
        [--threshold 10.0] [--top 10]
"""

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Report where extreme z-scores concentrate (feature vs sample).")
    parser.add_argument("--zscores", required=True,
                        help="Path to metabolite_zscores.csv from the pipeline.")
    parser.add_argument("--input", required=True,
                        help="Path to the original feature matrix CSV "
                             "(for the Classification / Oordeel targeted columns).")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="|z| above this counts as extreme (default 10).")
    parser.add_argument("--top", type=int, default=10,
                        help="Rows to show per table (default 10).")
    args = parser.parse_args()

    z = pd.read_csv(args.zscores, index_col=0)
    z = z.apply(pd.to_numeric, errors="coerce")
    logger.info(f"Loaded {z.shape[0]} samples x {z.shape[1]} features from {args.zscores}")

    meta = pd.read_csv(args.input, index_col=0)
    cls = pd.to_numeric(meta["Classification"], errors="coerce")
    oor = pd.to_numeric(meta["Oordeel targeted"], errors="coerce")
    normal_ids = meta.index[(cls == 0) & (oor == 0)].intersection(z.index)
    logger.info(f"{len(normal_ids)} of {len(z.index)} samples are normals "
                f"(Classification 0 AND Oordeel 0).")

    zn = z.loc[normal_ids]

    # ------------------------------------------------------------------
    # 1. Calibration check (normals centered by construction)
    # ------------------------------------------------------------------
    print("\n=== Calibration check (normals) ===")
    medians = zn.median()
    iqrs = zn.quantile(0.75) - zn.quantile(0.25)
    print(f"per-feature median of z: min {medians.min():.3f}, "
          f"max {medians.max():.3f} (should be ~0)")
    print(f"per-feature IQR of z:     min {iqrs.min():.3f}, "
          f"max {iqrs.max():.3f} (should be ~1)")
    worst_med = medians.abs().nlargest(args.top)
    if (medians.abs() > 0.1).any():
        print("features with |median| > 0.1 (upstream issue?):")
        print(worst_med.to_string())
    else:
        print("all per-feature medians within +/-0.1: OK")

    # ------------------------------------------------------------------
    # 2/3. Extremes by feature and by sample
    # ------------------------------------------------------------------
    extreme = zn.abs() > args.threshold
    n_extreme = int(extreme.values.sum())
    print(f"\n=== Extremes: |z| > {args.threshold} among normals ===")
    print(f"total extreme cells: {n_extreme} "
          f"({n_extreme / zn.size:.4%} of the normal z-matrix)")

    by_feature = extreme.sum().sort_values(ascending=False)
    by_feature = by_feature[by_feature > 0].head(args.top)
    if len(by_feature):
        print(f"\nfeatures with most extremes (concentration here = unstable feature):")
        print(by_feature.to_string())
    else:
        print("\nno feature has any extreme cell")

    by_sample = extreme.sum(axis=1).sort_values(ascending=False)
    by_sample = by_sample[by_sample > 0].head(args.top)
    if len(by_sample):
        print(f"\nnormal samples with most extremes (concentration here = QC suspect):")
        print(by_sample.to_string())
    else:
        print("\nno normal sample has any extreme cell")

    print("\nInterpretation:")
    print("  extremes scattered across samples AND features -> heavy tails, harmless")
    print("  extremes concentrated in a few samples        -> inspect those samples")
    print("  extremes concentrated in a few features       -> unstable features")


if __name__ == "__main__":
    main()
