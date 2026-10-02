"""Validation-stage evaluation: label-aware, one-shot, frozen-config only.

This module is the ONLY place in the package where the IMD labels may be
read for metric computation, and it exists to be run ONCE per frozen
configuration version on the validation half of the cohort. Running it
during development (on the development half) is allowed for sanity checks;
running it on the validation half and then changing the configuration
invalidates the validation -- the act itself, not any single number.

Metrics:

- Sensitivity (IMD samples flagged) and specificity (normals not flagged)
  with exact (Clopper-Pearson) confidence intervals.
- ROC-AUC of the continuous anomaly score (max pathway excess) with a
  bootstrap CI, IMD vs normal.
- Flag-resolution evidence: flags per flagged sample, so a reviewer can
  see what a flag actually consists of.
- Group breakdown for transparency (the 'other' group is reported but is
  not part of the primary metrics).

Everything reads from the pipeline outputs of the frozen run; nothing here
feeds back into any threshold.
"""

import logging
from math import comb
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), exact sum."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(k + 1):
        total += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def _solve_lower(n: int, k: int, target: float) -> float:
    """Solve P(X >= k | p) == target for p (tail increasing in p)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = 1.0 - _binom_cdf(k - 1, n, mid)
        if tail < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _solve_upper(n: int, k: int, target: float) -> float:
    """Solve P(X <= k | p) == target for p (CDF decreasing in p)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        cdf = _binom_cdf(k, n, mid)
        if cdf > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact_binomial_ci(k: int, n: int,
                      alpha: float = 0.05) -> Tuple[float, float]:
    """Clopper-Pearson exact CI for the success proportion k/n.

    lower: p with P(X >= k | p) == alpha/2  (tail increasing in p)
    upper: p with P(X <= k | p) == alpha/2  (CDF decreasing in p)
    """
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else _solve_lower(n, k, alpha / 2)
    hi = 1.0 if k >= n else _solve_upper(n, k, alpha / 2)
    return lo, hi


def _auc_of(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC of score x against binary label y."""
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def bootstrap_auc(labelled: pd.DataFrame, score_col: str,
                  n_bootstrap: int = 2000,
                  seed: int = 20260923) -> Tuple[float, float, float]:
    """Point AUC with a bootstrap percentile CI (no SciPy dependency)."""
    y = (labelled["group"] == "imd").to_numpy()
    x = labelled[score_col].to_numpy(dtype=float)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    if len(x) == 0 or y.sum() == 0 or (~y).sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = _auc_of(x, y)
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = _auc_of(x[idx], y[idx])
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize_evaluation(sample_decisions: pd.DataFrame,
                         pathway_flags: pd.DataFrame,
                         half: str = "validation",
                         validation_mask: Optional[pd.Series] = None,
                         bootstrap: int = 2000,
                         seed: int = 20260923) -> Dict[str, object]:
    """Compute the frozen-run evaluation metrics on the given half.

    Args:
        sample_decisions: per-sample decisions with ``sample_id``,
            ``flagged``, ``top_excess``, and ``group``.
        pathway_flags: per (sample, pathway) flags from the frozen run.
        half: label for logging ('validation' or 'development').
        validation_mask: optional boolean Series marking the half to keep;
            None evaluates all supplied samples.
        bootstrap: bootstrap resamples for the AUC CI.
        seed: bootstrap seed (part of the frozen config).

    Returns:
        Dict with the metrics and report tables.
    """
    decisions = sample_decisions.copy()
    if validation_mask is not None:
        keep = decisions["sample_id"].map(
            validation_mask.reindex(decisions["sample_id"], fill_value=False))
        decisions = decisions[keep.to_numpy()]

    normals = decisions[decisions["group"] == "normal"]
    imds = decisions[decisions["group"] == "imd"]
    others = decisions[decisions["group"] == "other"]

    n_normal, n_imd, n_other = len(normals), len(imds), len(others)
    sens_k = int(imds["flagged"].astype(bool).sum()) if n_imd else 0
    spec_k = int((~normals["flagged"].astype(bool)).sum()) if n_normal else 0

    sens = sens_k / n_imd if n_imd else float("nan")
    spec = spec_k / n_normal if n_normal else float("nan")
    sens_lo, sens_hi = exact_binomial_ci(sens_k, n_imd) if n_imd else (float("nan"),) * 2
    spec_lo, spec_hi = exact_binomial_ci(spec_k, n_normal) if n_normal else (float("nan"),) * 2

    score_col = ("max_excess" if "max_excess" in decisions.columns
                 else "top_excess")
    labelled = decisions[decisions["group"].isin(["imd", "normal"])]
    auc = auc_lo = auc_hi = float("nan")
    if (score_col in labelled.columns and not labelled.empty
            and labelled[score_col].notna().any()):
        auc, auc_lo, auc_hi = bootstrap_auc(labelled, score_col,
                                             n_bootstrap=bootstrap, seed=seed)

    flags_per_sample = (pathway_flags[pathway_flags["flagged"]]
                        .groupby("sample_id")["smp_id"].nunique())
    flagged = decisions[decisions["flagged"].astype(bool)].copy()
    if len(flagged):
        flagged["n_flagged_pathways"] = (
            flagged["sample_id"].map(flags_per_sample).fillna(0).astype(int))
        evidence = (flagged.groupby("group")["n_flagged_pathways"]
                    .agg(["mean", "median", "max"]).round(2))
    else:
        evidence = pd.DataFrame()

    group_summary = pd.DataFrame({
        "group": ["normal", "imd", "other"],
        "n": [n_normal, n_imd, n_other],
        "n_flagged": [int(normals["flagged"].astype(bool).sum()) if n_normal else 0,
                      sens_k,
                      int(others["flagged"].astype(bool).sum()) if n_other else 0],
    })

    logger.info("=" * 70)
    logger.info(f"FROZEN-CONFIG EVALUATION ({half} half)")
    logger.info("=" * 70)
    if n_imd:
        logger.info(f"Sensitivity (IMD flagged): {sens_k}/{n_imd} = "
                    f"{sens:.1%} (95% CI {sens_lo:.1%}-{sens_hi:.1%})")
    else:
        logger.info("No IMD samples in this half.")
    if n_normal:
        logger.info(f"Specificity (normals clean): {spec_k}/{n_normal} = "
                    f"{spec:.1%} (95% CI {spec_lo:.1%}-{spec_hi:.1%})")
    else:
        logger.info("No normal samples in this half.")
    if np.isfinite(auc):
        logger.info(f"ROC-AUC ({score_col}, IMD vs normal): {auc:.3f} "
                    f"(95% CI {auc_lo:.3f}-{auc_hi:.3f})")
    for g, row in evidence.iterrows():
        logger.info(f"Flag evidence, group '{g}': mean flags/sample "
                    f"{row['mean']}, median {int(row['median'])}, "
                    f"max {int(row['max'])}")
    for _, r in group_summary.iterrows():
        logger.info(f"Group '{r['group']}': {int(r['n_flagged'])} of "
                    f"{int(r['n'])} flagged.")

    missed = imds[~imds["flagged"].astype(bool)] if n_imd else imds
    if len(missed):
        logger.info(f"Missed IMD samples ({len(missed)}):")
        evidence_cols = [c for c in ("n_flagged_pathways", "sample_p_value",
                                     "top_pathway_name", "top_excess",
                                     "max_metabolite_z", "top_metabolite",
                                     "n_flagged_biomarkers", "top_biomarker",
                                     "top_disease")
                         if c in missed.columns]
        for _, m in missed.iterrows():
            parts = [f"{c}={m[c]}" for c in evidence_cols]
            logger.info(f"  {m['sample_id']}: " + ", ".join(parts))
    elif n_imd:
        logger.info("No missed IMD samples in this half.")

    return {
        "half": half,
        "sensitivity": sens, "sensitivity_ci": (sens_lo, sens_hi),
        "specificity": spec, "specificity_ci": (spec_lo, spec_hi),
        "auc": auc, "auc_ci": (auc_lo, auc_hi),
        "n_normal": n_normal, "n_imd": n_imd,
        "sensitivity_n": sens_k, "specificity_n": spec_k,
        "group_summary": group_summary,
        "flag_evidence": evidence,
    }
