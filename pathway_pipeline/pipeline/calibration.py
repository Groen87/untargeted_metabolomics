"""Label-blind calibration core: cohort split and reference hygiene.

Scientific contract of this module:

- :func:`stratified_split` assigns every sample once to a development or a
  validation half (deterministic, stratified by group). Validation samples
  must never influence any calibration choice downstream.
- :func:`leave_one_out_hygiene` replaces hand-picked reference exclusions.
  A candidate normal is excluded when, scored against ALL OTHER candidate
  normals (so it cannot mask its own disturbance), its deepest metabolite
  z-score exceeds a pre-specified absolute threshold.

No disease label is used anywhere in this module. The group labels produced
by :func:`assign_groups` exist only to stratify the split and to report
evaluation metrics; they never enter a threshold, a scale, or an exclusion
decision.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

CLASSIFICATION_COLUMN = "Classification"
OORDEEL_COLUMN = "Oordeel targeted"


def assign_groups(metadata: pd.DataFrame) -> pd.Series:
    """Assign each sample a reporting group: normal, imd, or other.

    Normals are ``Classification == 0`` AND ``Oordeel targeted == 0``; IMD
    samples are ``Classification == 1`` AND ``Oordeel targeted == 1``. The
    group is reporting/stratification-only evidence and is never used to
    set thresholds.
    """
    group = pd.Series("other", index=metadata.index)
    if {CLASSIFICATION_COLUMN, OORDEEL_COLUMN}.issubset(metadata.columns):
        cls = pd.to_numeric(metadata[CLASSIFICATION_COLUMN], errors="coerce")
        oor = pd.to_numeric(metadata[OORDEEL_COLUMN], errors="coerce")
        group[(cls == 1) & (oor == 1)] = "imd"
        group[(cls == 0) & (oor == 0)] = "normal"
    else:
        logger.warning("Metadata lacks the label columns; every sample is "
                       "grouped 'other'.")
    return group


def stratified_split(groups: pd.Series,
                     seed: int = 20260923,
                     validation_fraction: float = 0.5) -> pd.Series:
    """Deterministically assign each sample to development or validation.

    Within every group the assignment is a seeded random permutation, so
    both halves contain the same group proportions and the split is exactly
    reproducible from the seed. Every sample is assigned exactly once.

    Args:
        groups: sample_id -> group label (from :func:`assign_groups`).
        seed: seed of the permutation (part of the frozen configuration).
        validation_fraction: fraction of each group assigned to validation.

    Returns:
        Boolean Series indexed like ``groups``; True for validation samples.
    """
    rng = np.random.default_rng(seed)
    is_validation = pd.Series(False, index=groups.index)
    for g in sorted(groups.unique()):
        idx = np.asarray(groups.index[groups == g])
        if len(idx) == 0:
            continue
        n_val = int(round(len(idx) * validation_fraction))
        perm = rng.permutation(len(idx))
        is_validation.loc[idx[perm[:n_val]]] = True
    n_val = int(is_validation.sum())
    logger.info(f"Cohort split (seed {seed}): {len(groups) - n_val} "
                f"development / {n_val} validation samples "
                f"({validation_fraction:.0%} per group, stratified).")
    return is_validation


def leave_one_out_hygiene(features: pd.DataFrame,
                          normal_mask: pd.Series,
                          max_depth: float = 10.0,
                          max_rounds: int = 10
                          ) -> Tuple[pd.Series, pd.DataFrame]:
    """Exclude candidate normals whose leave-one-out depth is too deep.

    For every candidate normal *i*, robust reference statistics (median and
    IQR per feature) are computed over the OTHER current candidates only,
    and *i*'s z-scores are evaluated against that peer reference. A sample
    therefore cannot mask its own disturbance by inflating the reference
    spread. The sample is excluded from the calibration reference when its
    deepest ``max |z|`` exceeds ``max_depth`` (a pre-specified absolute
    threshold, not a data-derived quantile).

    The check runs in rounds: after each round the excluded candidates also
    leave everyone else's peer set, the depths are recomputed on the
    tightened reference, and the process repeats until no new candidate
    crosses ``max_depth``. Removing one deep outlier tightens the peer IQR,
    which can reveal borderline candidates a single pass would have hidden.

    Excluded samples keep their scores in every output; they only leave the
    reference used to calibrate z-scores, pathway thresholds, and nulls.

    Args:
        features: sample x feature matrix (log10 values, already restricted
            to the scored feature set).
        normal_mask: boolean Series marking the candidate normals.
        max_depth: pre-specified exclusion threshold on the LOO max |z|.
        max_rounds: maximum number of iterative exclusion rounds.

    Returns:
        Tuple ``(clean_mask, report)``: the reference mask with excluded
        candidates set to False, and a per-candidate report with columns
        ``sample_id``, ``loo_max_z``, ``excluded``.
    """
    ids = list(normal_mask.index[normal_mask])
    if len(ids) == 0:
        return normal_mask.copy(), pd.DataFrame(
            columns=["sample_id", "loo_max_z", "excluded"])

    numeric = features.apply(pd.to_numeric, errors="coerce")
    remaining = list(ids)
    excluded_ids = []
    depths = {}
    for round_no in range(1, max_rounds + 1):
        newly = []
        for sid in remaining:
            peers = numeric.loc[remaining].drop(sid, errors="ignore")
            med = peers.median()
            scale = peers.quantile(0.75) - peers.quantile(0.25)
            scale = scale.replace(0, np.nan)
            z = (numeric.loc[sid] - med) / scale
            if z.notna().any():
                depths[sid] = float(np.nanmax(np.abs(z.to_numpy())))
            else:
                depths[sid] = float("nan")
            if depths[sid] > max_depth:
                newly.append(sid)
        if not newly:
            break
        remaining = [s for s in remaining if s not in newly]
        excluded_ids.extend(newly)
        logger.debug(f"Reference hygiene round {round_no}: excluded "
                     f"{len(newly)} candidate(s), {len(remaining)} remain.")

    report = pd.DataFrame({
        "sample_id": ids,
        "loo_max_z": [depths.get(s, float("nan")) for s in ids],
        "excluded": [s in excluded_ids for s in ids],
    })
    clean = normal_mask.copy()
    if excluded_ids:
        clean.loc[excluded_ids] = False

    n_excluded = len(excluded_ids)
    if n_excluded:
        bad = report.loc[report["excluded"], ["sample_id", "loo_max_z"]]
        bad = bad.sort_values("loo_max_z", ascending=False)
        detail = ", ".join(f"{r.sample_id} "
                           f"(|z| {r.loo_max_z:.1f})" for r in bad.itertuples())
        logger.info(f"Reference hygiene: excluded {n_excluded} of {len(ids)} "
                    f"candidate normals (iterative LOO max |z| > "
                    f"{max_depth}): {detail}.")
    else:
        logger.info(f"Reference hygiene: all {len(ids)} candidate normals "
                    f"stay in the reference (LOO max |z| <= {max_depth}).")
    return clean, report
