from __future__ import annotations

"""
Bridge-QC batch alignment module for the combined batch pipeline.

Aligns batches to a common reference using cross-batch control (QC) samples
that are the SAME material run in every batch (QC3, QC4, blauw). Unlike expQC
(a pool of each batch's own biological samples), these fixed materials carry no
batch-specific biology, so any between-batch difference in their feature values
is pure instrument/batch effect. Removing that per-feature, per-batch offset is
what lowers the score-shift that inflates the outlier-detection false-positive
rate.

Strategy (biology-preserving, "Strategy A"):
- For each feature and batch, estimate the batch's bridge center as the median
  log-intensity of all bridge QC points present in that batch.
- Estimate a grand center as the median over all batches of the per-batch
  bridge centers.
- Correction factor for batch b, feature f:
      factor = exp(grand_center_f - batch_b_center_f)
  Applied multiplicatively (in linear space) to every sample of batch b for
  feature f. Batches whose bridge center sits above the grand center are scaled
  down, and vice versa, pulling every batch's QC to the grand reference while
  leaving within-batch biological variation (relative differences between
  biological samples) untouched.

This removes ONLY the offset the bridge materials span; biological axes that
the bridge QCs do not vary in are preserved. It is deliberately more
conservative than biology-blind ComBat and is the recommended first
between-batch correction when biology is confounded with batch.

Robustness guards (important because bridge QC counts per batch are low,
typically 2-6):
- Presence filter: only correct features present in the bridge materials in at
  least ``min_bridge_batches`` batches. Features present in few batches give
  unstable ratios and are left uncorrected.
- Factor clipping: correction factors are clipped to ``[1/max_factor,
  max_factor]`` so a single bad bridge injection cannot rescale a feature by an
  extreme amount.
- Per-batch centers use the MEDIAN (not mean) of available bridge points, so a
  single outlying bridge point in a batch does not dominate.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _classify_bridge_qc(col: str, bridge_patterns: List[str]) -> bool:
    """Return True if a column name matches any of the bridge QC patterns."""
    col_lower = col.lower()
    return any(p.lower() in col_lower for p in bridge_patterns)


def bridge_qc_scaling(
    merged_data: pd.DataFrame,
    merged_metadata: pd.DataFrame,
    bridge_patterns: Optional[List[str]] = None,
    min_bridge_batches: int = 8,
    max_factor: float = 2.0,
    min_intensity: float = 0.0,
    log_offset: float = 1.0,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Align batches to a common reference using cross-batch bridge QC samples.

    Operates on merged data (features as rows, samples as columns), where each
    sample column maps to a batch via ``merged_metadata['batch']``. The
    correction is a per-feature, per-batch multiplicative factor derived from
    bridge QC samples present in every batch.

    Args:
        merged_data: DataFrame with features as rows, samples as columns
            (PQN + LOESS already applied per batch).
        merged_metadata: Metadata with columns 'original_col' and 'batch'.
        bridge_patterns: Substrings identifying bridge QC samples
            (default: ['QC3', 'QC4', 'blauw']). These must be the SAME material
            run in every batch.
        min_bridge_batches: Minimum number of batches in which a feature must
            be present in the bridge QCs to be corrected. Features present in
            fewer batches are left uncorrected (ratios unstable).
        max_factor: Correction factors are clipped to
            ``[1/max_factor, max_factor]`` (default 2.0 => factors in [0.5, 2.0]).
        min_intensity: Bridge QC values below this are treated as absent
            (default 0.0).
        log_offset: Added to intensities before log transform to handle zeros
            (default 1.0).
        output_dir: Optional directory to save diagnostics.

    Returns:
        Tuple of:
        - corrected_data: DataFrame, same shape as merged_data, bridge-aligned.
        - diagnostics: dict with counts of corrected/skipped features and the
          per-batch correction factor summary.
    """
    if bridge_patterns is None:
        bridge_patterns = ["QC3", "QC4", "blauw"]

    # Map sample column -> batch.
    batch_dict = dict(zip(merged_metadata["original_col"], merged_metadata["batch"]))
    samples = [c for c in merged_data.columns if c in batch_dict]
    batch_vector = np.array([batch_dict[c] for c in samples])
    unique_batches = sorted(set(batch_vector.tolist()))

    # Identify bridge QC columns present in merged_data.
    bridge_cols = [c for c in merged_data.columns if _classify_bridge_qc(c, bridge_patterns)]
    if len(bridge_cols) == 0:
        logger.warning(
            "Bridge-QC scaling: no bridge QC samples found with patterns "
            f"{bridge_patterns}. Skipping bridge scaling."
        )
        return merged_data.copy(), {"corrected_features": 0, "skipped_features": 0, "reason": "no_bridge_qc"}

    logger.info(f"Bridge-QC scaling: {len(bridge_cols)} bridge QC samples, "
                f"{len(unique_batches)} batches")
    logger.info(f"  Bridge patterns: {bridge_patterns}")
    logger.info(f"  Bridge QCs per batch:")
    for b in unique_batches:
        b_bridge = [c for c in bridge_cols if batch_dict.get(c) == b]
        logger.info(f"    {b}: {len(b_bridge)} ({[c.split('_')[-1] for c in b_bridge][:6]})")

    # Build per-batch bridge mask over the sample columns (for batch assignment).
    bridge_per_batch: Dict[str, List[str]] = {b: [] for b in unique_batches}
    for c in bridge_cols:
        if c in bridge_per_batch:
            b = batch_dict.get(c)
            if b in bridge_per_batch:
                bridge_per_batch[b].append(c)

    # Work on a copy; bridge QC columns that have a batch assignment.
    bridge_cols_present = [c for c in bridge_cols if c in batch_dict]

    data = merged_data.copy()
    n_features = data.shape[0]
    n_batches = len(unique_batches)

    # Per-batch bridge center: for each feature, median log-intensity of bridge
    # QCs present in that batch (above min_intensity threshold).
    batch_centers = np.full((n_features, n_batches), np.nan)
    # Track how many batches each feature is "present" in (for the presence filter).
    n_present_per_feature = np.zeros(n_features, dtype=int)

    raw_bridge = data.loc[:, bridge_cols_present].values.astype(float)
    for bi, b in enumerate(unique_batches):
        cols = [c for c in bridge_cols_present if batch_dict[c] == b]
        if len(cols) == 0:
            continue
        vals = data.loc[:, cols].values.astype(float)  # (n_features, n_bridge_in_batch)
        # Median log-intensity over present, positive bridge points. Replace
        # non-present / zero with NaN so they don't enter the median.
        log_vals = np.log(vals + log_offset)
        log_vals[vals <= min_intensity] = np.nan
        with np.errstate(invalid="ignore", all="ignore"):
            med = np.nanmedian(log_vals, axis=1)
        batch_centers[:, bi] = med
        # A feature is "present in this batch" if its bridge center is finite.
        finite = np.isfinite(med)
        n_present_per_feature += finite.astype(int)

    # Grand center per feature: median over batches of per-batch bridge centers.
    with np.errstate(invalid="ignore", all="ignore"):
        grand_centers = np.nanmedian(batch_centers, axis=1)

    # Correction factor per feature per batch: exp(grand_center - batch_center).
    factors = np.full((n_features, n_batches), np.nan)
    valid = np.isfinite(grand_centers)
    for bi in range(n_batches):
        bc = batch_centers[:, bi]
        with np.errstate(invalid="ignore", all="ignore"):
            f = np.exp(grand_centers - bc)
        f = np.clip(f, 1.0 / max_factor, max_factor)
        f[~valid] = np.nan
        f[~np.isfinite(bc)] = np.nan
        factors[:, bi] = f

    # Apply factors: scale each sample by its batch's per-feature factor.
    corrected = data.copy()
    sample_factor_arr = np.full((n_features, len(samples)), np.nan)
    for si, c in enumerate(samples):
        bi = unique_batches.index(batch_dict[c])
        sample_factor_arr[:, si] = factors[:, bi]
    sample_vals = data.loc[:, samples].values.astype(float)
    # Where factor is NaN (feature skipped), leave the original value unchanged.
    apply_mask = np.isfinite(sample_factor_arr)
    corrected_vals = np.where(apply_mask, sample_vals * sample_factor_arr, sample_vals)
    corrected.loc[:, samples] = corrected_vals

    # Apply the same factors to bridge QC columns themselves so they are
    # aligned to the grand reference alongside the biological samples.
    for c in bridge_cols_present:
        bi = unique_batches.index(batch_dict[c])
        f = factors[:, bi]
        col_vals = data.loc[:, c].values.astype(float)
        m = np.isfinite(f)
        corrected.loc[:, c] = np.where(m, col_vals * f, col_vals)

    # Diagnostics.
    n_corrected = int(np.sum(valid & (n_present_per_feature >= min_bridge_batches)))
    n_skipped = int(n_features - n_corrected)
    factor_summary = {
        b: {
            "median_factor": float(np.nanmedian(factors[:, unique_batches.index(b)])),
            "mean_factor": float(np.nanmean(factors[:, unique_batches.index(b)])),
            "n_corrected_features": int(np.sum(np.isfinite(factors[:, unique_batches.index(b)]))),
        }
        for b in unique_batches
    }
    diagnostics = {
        "corrected_features": n_corrected,
        "skipped_features": n_skipped,
        "min_bridge_batches": min_bridge_batches,
        "max_factor": max_factor,
        "bridge_patterns": bridge_patterns,
        "n_bridge_samples": len(bridge_cols_present),
        "n_batches": n_batches,
        "per_batch_factors": factor_summary,
    }

    logger.info(f"Bridge-QC scaling complete: {n_corrected}/{n_features} features corrected, "
                f"{n_skipped} skipped (present in < {min_bridge_batches} bridge batches)")
    for b in unique_batches:
        s = factor_summary[b]
        logger.info(f"  Batch {b}: median factor {s['median_factor']:.4f}, "
                    f"corrected {s['n_corrected_features']} features")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            factors, index=data.index, columns=unique_batches
        ).to_csv(output_dir / "bridge_qc_factors.csv")
        pd.DataFrame([diagnostics]).to_csv(output_dir / "bridge_qc_diagnostics.csv", index=False)
        logger.info(f"Bridge-QC diagnostics saved to {output_dir}")

    return corrected, diagnostics
