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
NON_TREATED_COLUMN = "Non-treated"


def assign_groups(metadata: pd.DataFrame,
                  normal_mask: pd.Series = None,
                  normal_classification: int = 0,
                  normal_oordeel: int = 0,
                  untreated_imd_only: bool = False) -> pd.Series:
    """Assign each sample a reporting group: normal, imd, or other.

    Normals are the configured reference set (``Classification ==
    normal_classification`` AND ``Oordeel targeted == normal_oordeel``;
    defaults 0/0); IMD samples are ``Classification == 1`` AND
    ``Oordeel targeted == 1``. With ``untreated_imd_only`` (and the
    ``Non-treated`` column present in the metadata), an IMD sample counts
    as ``imd`` only when ``Non-treated == 1`` (untreated, a true IMD
    presentation); treated IMDs (``Non-treated == 0`` or empty) fall to
    ``other``, since treatment normalizes the metabolome and they are not
    the phenotype the screen must catch. The group is
    reporting/stratification-only evidence and is never used to set
    thresholds.

    Args:
        metadata: frame holding the label columns (when present).
        normal_mask: boolean Series marking the normal reference; when
            given it overrides the label rule for the 'normal' group.
        normal_classification: classification value marking normals.
        normal_oordeel: oordeel value marking normals.
        untreated_imd_only: demote treated IMDs (Non-treated != 1) to
            'other'; inert when the Non-treated column is absent.
    """
    group = pd.Series("other", index=metadata.index)
    if {CLASSIFICATION_COLUMN, OORDEEL_COLUMN}.issubset(metadata.columns):
        cls = pd.to_numeric(metadata[CLASSIFICATION_COLUMN], errors="coerce")
        oor = pd.to_numeric(metadata[OORDEEL_COLUMN], errors="coerce")
        imd_mask = (cls == 1) & (oor == 1)
        if untreated_imd_only and NON_TREATED_COLUMN in metadata.columns:
            untreated = pd.to_numeric(metadata[NON_TREATED_COLUMN],
                                      errors="coerce").fillna(0)
            imd_mask = imd_mask & (untreated == 1)
        group[imd_mask] = "imd"
        if normal_mask is None:
            group[(cls == normal_classification)
                  & (oor == normal_oordeel)] = "normal"
    if normal_mask is not None:
        group[normal_mask.reindex(metadata.index, fill_value=False)] = "normal"
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
                          max_depth: float = 20.0,
                          max_rounds: int = 10,
                          max_excluded_fraction: float = None
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

    A pre-declared ``max_excluded_fraction`` guards against a miscalibrated
    ``max_depth`` cascading: when the exclusion fraction exceeds the cap,
    the hygiene stops at the cap (deepest candidates first) and logs a
    WARNING that the reference is suspect -- inspect the LOO depth
    distribution and reconsider ``max_depth`` before trusting any flag.

    Excluded samples keep their scores in every output; they only leave the
    reference used to calibrate z-scores, pathway thresholds, and nulls.

    Args:
        features: sample x feature matrix (log10 values, already restricted
            to the scored feature set).
        normal_mask: boolean Series marking the candidate normals.
        max_depth: pre-specified exclusion threshold on the LOO max |z|.
        max_rounds: maximum number of iterative exclusion rounds.
        max_excluded_fraction: optional cap on the fraction of candidates the
            hygiene may exclude (e.g. 0.10 = at most 10%); None disables it.

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
    max_excluded = None
    if max_excluded_fraction is not None:
        max_excluded = int(np.floor(max_excluded_fraction * len(ids)))
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
        # Enforce the pre-declared cap within the round: keep only the
        # deepest candidates up to the remaining capacity.
        hit_cap = (max_excluded is not None
                   and len(excluded_ids) + len(newly) > max_excluded)
        if hit_cap:
            capacity = max(0, max_excluded - len(excluded_ids))
            newly = sorted(newly, key=lambda s: depths.get(s, float("nan")),
                           reverse=True)[:capacity]
        remaining = [s for s in remaining if s not in newly]
        excluded_ids.extend(newly)
        logger.debug(f"Reference hygiene round {round_no}: excluded "
                     f"{len(newly)} candidate(s), {len(remaining)} remain.")
        if hit_cap:
            logger.warning(
                f"Reference hygiene hit the exclusion cap "
                f"({len(excluded_ids)} of {len(ids)} candidates >= "
                f"{max_excluded_fraction:.0%}): the max_depth threshold is "
                f"likely miscalibrated against the LOO depth distribution. "
                f"Inspect reference_hygiene.csv and reconsider max_depth "
                f"before trusting any flag.")
            break

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
                    f"candidate normals ({n_excluded / len(ids):.0%}; "
                    f"iterative LOO max |z| > "
                    f"{max_depth}): {detail}.")
    else:
        logger.info(f"Reference hygiene: all {len(ids)} candidate normals "
                    f"stay in the reference (LOO max |z| <= {max_depth}).")
    return clean, report
