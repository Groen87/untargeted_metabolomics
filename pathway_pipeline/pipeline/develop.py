"""Development-stage QC: every check here is label-blind.

Used while the configuration is being developed. Nothing in this module
may read the IMD labels or the group assignments; every diagnostic is
computed on the calibration reference (development normals) alone, or on
measurement/chemistry properties. That separation is what makes the
development loop scientifically valid: iterate as much as you like here,
no disease signal leaks into any choice.

Checks:

1. Calibration verification: the reference normals' z-score medians must
   be ~0 and their IQRs ~1 by construction; deviations point to a wrong
   reference subset or a stale reference.
2. Noise-floor features: a value spike plus a razor-thin reference IQR --
   these manufacture huge z-scores from trivial absolute shifts.
3. Diverging duplicate features per HMDB metabolite: when two features
   share a metabolite, their reference IQRs should be similar; a >2x
   divergence means the averaged z is dominated by the noisiest twin.
4. Threshold stability: bootstrap resampling of the reference normals to
   check how much the per-pathway 99th-percentile thresholds wobble; an
   unstable threshold needs more reference samples or a wider percentile.
5. Redundant pathways: pathways whose flag sets would be identical (or
   near-identical) because they share most scored metabolites -- pruning
   them is label-blind and reduces multiplicity.

The module writes a ``development_qc.csv`` and logs a summary; it never
changes the configuration itself. Acting on its output (adding an
override, a demotion, changing a threshold) is a human decision and creates
a new frozen configuration version.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def verify_calibration(zscores: pd.DataFrame,
                       reference_mask: pd.Series) -> pd.DataFrame:
    """Check that the reference normals' z-scores are standardized."""
    ref = zscores[reference_mask.reindex(zscores.index, fill_value=False)]
    if ref.empty:
        return pd.DataFrame(columns=["feature", "median_abs_z",
                                     "iqr_z", "n_values"])
    med = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    report = pd.DataFrame({
        "feature": zscores.columns,
        "median_abs_z": med.abs().to_numpy(),
        "iqr_z": iqr.to_numpy(),
        "n_values": ref.notna().sum().to_numpy(),
    })
    n_bad = int(((report["median_abs_z"] > 0.05) | (report["iqr_z"] < 0.8)
                 | (report["iqr_z"] > 1.2)).sum())
    logger.info(f"Calibration check: {n_bad} of {len(report)} features "
                f"deviate from median |z| ~ 0 / IQR ~ 1 among reference "
                f"normals.")
    return report


def noise_floor_features(features: pd.DataFrame,
                          reference_mask: pd.Series,
                          spike_fraction: float = 0.5,
                          max_iqr: float = 0.08) -> pd.DataFrame:
    """Features with a spiked value and razor-thin reference IQR.

    Args:
        features: raw log10 feature matrix (all samples).
        reference_mask: boolean Series marking the calibration reference.
        spike_fraction: most-common-value fraction that counts as a spike.
        max_iqr: reference IQR below which a feature is noise-floor suspect.
    """
    ref_ids = reference_mask.index[reference_mask]
    sub = features.loc[features.index.isin(ref_ids)]
    sub = sub.apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in sub.columns:
        vals = sub[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        uniq, counts = np.unique(vals, return_counts=True)
        spike = float(counts.max() / len(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        if spike >= spike_fraction and iqr <= max_iqr:
            rows.append({
                "feature": col,
                "spike_fraction": spike,
                "spike_value": float(uniq[counts.argmax()]),
                "reference_iqr": iqr,
            })
    report = pd.DataFrame(rows)
    logger.info(f"Noise-floor check: {len(report)} features show "
                f"spike >= {spike_fraction:.0%} with reference IQR "
                f"<= {max_iqr} log10.")
    if len(report):
        for r in report.itertuples():
            logger.info(f"  {r.feature}: spike {r.spike_fraction:.0%}, "
                        f"IQR {r.reference_iqr:.3f}")
    return report


def diverging_duplicates(reference_stats: pd.DataFrame,
                         feature_to_hmdb: pd.DataFrame,
                         max_ratio: float = 2.0) -> pd.DataFrame:
    """Metabolites whose dataset features' reference IQRs diverge.

    A >2x IQR divergence between two features sharing an HMDB ID means the
    plain or even weighted average is dominated by the noisiest twin --
    or the two features are not really the same compound (a mapping
    collision, to be confirmed chemically and pinned with an override).
    """
    scale_map = dict(zip(reference_stats["feature"],
                         reference_stats["scale"]))
    links = feature_to_hmdb[feature_to_hmdb["hmdb_id"].notna()].copy()
    links["scale"] = links["feature"].map(scale_map)
    links = links.dropna(subset=["scale"])
    grouped = (links.groupby("hmdb_id")
               .agg(n_features=("feature", "nunique"),
                    scale_min=("scale", "min"),
                    scale_max=("scale", "max"),
                    features=("feature", lambda s: "; ".join(sorted(set(s)))))
               .reset_index())
    diverging = grouped[(grouped["n_features"] > 1)
                        & (grouped["scale_max"] > max_ratio * grouped["scale_min"])]
    logger.info(f"Duplicate-feature check: {len(diverging)} HMDB metabolites "
                f"have features with reference IQRs diverging more than "
                f"{max_ratio}x.")
    for r in diverging.itertuples():
        logger.info(f"  {r.hmdb_id}: IQR {r.scale_min:.3f} vs "
                    f"{r.scale_max:.3f} | {r.features}")
    return diverging


def threshold_stability(zscores: pd.DataFrame,
                        pathway_scores: pd.DataFrame,
                        reference_mask: pd.Series,
                        percentile: float = 99.0,
                        n_bootstrap: int = 200,
                        seed: int = 20260923) -> pd.DataFrame:
    """Bootstrap the reference to measure per-pathway threshold wobble.

    Resamples the reference normals with replacement, recomputes each
    pathway's percentile threshold each time, and reports the relative
    spread (IQR of the bootstrap thresholds / median threshold). A pathway
    whose threshold is unstable will produce borderline flags that flip on
    reference resampling; flagging it needs either more reference samples
    or a wider percentile.
    """
    ref_ids = np.asarray(reference_mask.index[reference_mask])
    if len(ref_ids) < 10 or pathway_scores.empty:
        return pd.DataFrame(columns=["smp_id", "median_threshold",
                                     "threshold_iqr", "rel_wobble"])
    rng = np.random.default_rng(seed)
    scores_wide = pathway_scores.pivot(index="sample_id", columns="smp_id",
                                       values="z_stouffer_abs")
    boot_thresholds = []
    for _ in range(n_bootstrap):
        sample_ids = rng.choice(ref_ids, size=len(ref_ids), replace=True)
        # Deduplicate: a duplicated row would double-count in quantiles.
        unique_ids = pd.unique(sample_ids)
        sub = scores_wide.loc[scores_wide.index.isin(unique_ids)]
        boot_thresholds.append(
            sub.quantile(percentile / 100.0).rename("threshold"))
    boot = pd.concat(boot_thresholds, axis=1).T
    med = boot.median()
    iqr = boot.quantile(0.75) - boot.quantile(0.25)
    report = pd.DataFrame({
        "smp_id": boot.columns,
        "median_threshold": med.to_numpy(),
        "threshold_iqr": iqr.to_numpy(),
        "rel_wobble": (iqr / med.replace(0, np.nan)).to_numpy(),
    }).sort_values("rel_wobble", ascending=False)
    if len(report):
        n_unstable = int((report["rel_wobble"] > 0.5).sum())
        logger.info(f"Threshold stability: {n_unstable} of {len(report)} "
                    f"pathways have bootstrap threshold wobble > 50% of "
                    f"the threshold (n_bootstrap={n_bootstrap}).")
        for r in report[report["rel_wobble"] > 0.5].head(10).itertuples():
            logger.info(f"  {r.smp_id}: threshold {r.median_threshold:.2f} "
                        f"(wobble {r.rel_wobble:.2f})")
    return report


def redundant_pathways(pathway_flags: pd.DataFrame,
                       feature_to_pathway: pd.DataFrame,
                       min_jaccard: float = 0.8) -> pd.DataFrame:
    """Pathway pairs whose scored-metabolite sets nearly coincide.

    PathBank disease pathways share metabolites heavily; five pathways
    driven by the same three metabolites produce five identical flags and
    inflate the flagged-pathway count. Pruning near-duplicates is
    label-blind (uses only pathway composition) and reduces multiplicity.
    """
    if pathway_flags.empty or feature_to_pathway.empty:
        return pd.DataFrame(columns=["pathway_a", "pathway_b",
                                     "jaccard", "n_metabolites"])
    metabolite_sets = (feature_to_pathway
                       .groupby("smp_id")["hmdb_id"]
                       .agg(lambda s: set(s)).to_dict())
    smp_ids = sorted(metabolite_sets)
    rows = []
    for i in range(len(smp_ids)):
        for j in range(i + 1, len(smp_ids)):
            a, b = metabolite_sets[smp_ids[i]], metabolite_sets[smp_ids[j]]
            if not a or not b:
                continue
            union = a | b
            jac = len(a & b) / len(union)
            if jac >= min_jaccard:
                rows.append({
                    "pathway_a": smp_ids[i],
                    "pathway_b": smp_ids[j],
                    "jaccard": jac,
                    "n_metabolites": len(union),
                })
    report = pd.DataFrame(rows)
    logger.info(f"Pathway redundancy: {len(report)} pathway pairs with "
                f"Jaccard >= {min_jaccard} in their scored metabolite sets.")
    return report


def run_development_qc(zscores: pd.DataFrame,
                      reference_stats: pd.DataFrame,
                      reference_mask: pd.Series,
                      features: pd.DataFrame,
                      feature_to_hmdb: pd.DataFrame,
                      pathway_scores: pd.DataFrame,
                      pathway_flags: pd.DataFrame,
                      feature_to_pathway: pd.DataFrame,
                      percentile: float = 99.0,
                      n_bootstrap: int = 200,
                      output_csv: str = None) -> Dict[str, pd.DataFrame]:
    """Run every label-blind development check and collect the reports.

    Args:
        zscores: calibrated z-score matrix (all samples).
        reference_stats: per-feature reference table (feature, scale, ...).
        reference_mask: boolean Series marking the calibration reference.
        features: raw log10 feature matrix (all samples, scored columns).
        feature_to_hmdb: feature -> HMDB mapping with match methods.
        pathway_scores: Stouffer scores (sample, pathway) long table.
        pathway_flags: per (sample, pathway) flags with excess.
        feature_to_pathway: (feature, pathway) links.
        percentile: flagging percentile for the stability bootstrap.
        n_bootstrap: bootstrap resamples for threshold stability.
        output_csv: optional path to write the noise-floor feature list.

    Returns:
        Dict with the check reports (calibration, noise floor, duplicates,
        stability, redundancy).
    """
    calibration_report = verify_calibration(zscores, reference_mask)
    noise_report = noise_floor_features(features, reference_mask)
    duplicate_report = diverging_duplicates(reference_stats, feature_to_hmdb)
    stability_report = threshold_stability(
        zscores, pathway_scores, reference_mask,
        percentile=percentile, n_bootstrap=n_bootstrap)
    redundancy_report = redundant_pathways(pathway_flags, feature_to_pathway)

    if output_csv and len(noise_report):
        noise_report.to_csv(output_csv, index=False)
        logger.info(f"Wrote noise-floor feature list to {output_csv}")
    return {
        "calibration": calibration_report,
        "noise_floor": noise_report,
        "duplicates": duplicate_report,
        "stability": stability_report,
        "redundancy": redundancy_report,
    }
