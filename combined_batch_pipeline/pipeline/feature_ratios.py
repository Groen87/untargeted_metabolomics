"""
Feature-ratio builder for the combined batch pipeline.

Adds clinically-informative metabolite ratios (e.g. acylcarnitine ratios) as
new feature rows, computed AFTER log10 transformation and BEFORE
RobustScaler. Ratios are computed on the raw concentration scale (the log
values are exponentiated back) so that sums in the numerator/denominator are
correct -- log(A+B) != log(A)+log(B), so sums cannot be combined on the log
scale directly. The final ratio feature is itself log10-transformed so it
sits in the same space as the other (log10) features and is then robustly
scaled together with them.

Configuration (config.yaml)::

    feature_ratios:
      - name: "ratio_C3_C2"            # output feature name
        numerator: ["Propionylcarnitine.HMDB0000842"]
        denominator: ["L-Acetylcarnitine.HMDB0000201"]
      - name: "ratio_C0_long"
        numerator: ["L-carnitine.HMDB0000062"]
        denominator:
          - "Palmitoylcarnitine.HMDB0000222"
          - "Stearoylcarnitine.HMDB0000848"

Numerator and denominator are each lists of feature names; the entries in each
list are summed on the raw concentration scale before the ratio is taken. A
single-entry list is just that one metabolite. Feature names must match the
DataFrame index EXACTLY (case-sensitive) as they appear after feature
filtering -- e.g. "Propionylcarnitine.HMDB0000842".
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_feature_ratios(
    data: pd.DataFrame,
    ratios: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Add ratio feature rows to a log10-transformed, feature-rows DataFrame.

    The input ``data`` has features as rows (index = feature names) and
    samples as columns, with values already on the log10 scale (i.e.
    ``np.log10(raw_concentration)``). For each configured ratio:

      1. Recover raw concentrations: ``raw = 10 ** log_value``.
      2. Sum the raw concentrations within the numerator list and within the
         denominator list (correct for ratio-of-sums such as
         L-carnitine / (Palmitoylcarnitine + Stearoylcarnitine)).
      3. Take the concentration ratio ``num / den``.
      4. log10-transform the ratio so the new feature is on the same log10
         scale as the existing features, ready for RobustScaler.

    Features named in a ratio that are absent from the index are skipped with
    a warning; if ALL of a ratio's numerator OR all of its denominator
    features are missing, the ratio is skipped entirely. Zero/NaN-safe: a
    denominator sum of zero yields NaN for that sample (preserved as a missing
    value, consistent with how the pipeline handles other features).

    Args:
        data: log10-transformed DataFrame, features as rows, samples as columns.
        ratios: list of ratio specs, each with keys ``name``,
            ``numerator`` (list of feature names), ``denominator`` (list of
            feature names).

    Returns:
        The input DataFrame with the new ratio feature rows appended (a copy;
        the input is not modified in place). If ``ratios`` is empty or None,
        the input is returned unchanged.
    """
    if not ratios:
        return data

    if data.empty:
        logger.warning("Feature-ratio builder: data is empty; nothing to do.")
        return data

    available = set(data.index.astype(str))
    out = data.copy()
    added = 0

    for spec in ratios:
        name = spec.get('name')
        num_names = spec.get('numerator') or []
        den_names = spec.get('denominator') or []
        if not name or not num_names or not den_names:
            logger.warning(
                f"Feature-ratio spec missing name/numerator/denominator: {spec!r}; skipped."
            )
            continue

        # Coerce to lists of strings.
        if isinstance(num_names, str):
            num_names = [num_names]
        if isinstance(den_names, str):
            den_names = [den_names]
        num_names = [str(n) for n in num_names]
        den_names = [str(n) for n in den_names]

        num_present = [n for n in num_names if n in available]
        den_present = [n for n in den_names if n in available]
        num_missing = [n for n in num_names if n not in available]
        den_missing = [n for n in den_names if n not in available]

        if num_missing:
            logger.warning(
                f"Ratio {name!r}: numerator feature(s) not found in data, "
                f"skipped for this ratio: {num_missing}"
            )
        if den_missing:
            logger.warning(
                f"Ratio {name!r}: denominator feature(s) not found in data, "
                f"skipped for this ratio: {den_missing}"
            )
        if not num_present or not den_present:
            logger.warning(
                f"Ratio {name!r}: no usable numerator or denominator features "
                f"after missing check; ratio skipped."
            )
            continue

        if name in available:
            logger.warning(
                f"Ratio {name!r}: a feature with this name already exists in "
                f"the data; the ratio will overwrite it."
            )

        # Recover raw concentrations from the log10 values and sum within each
        # group. Clip log values to avoid 10**inf / 10**-inf blow-ups; NaNs in
        # the source features propagate to NaN in the sum (correct: a missing
        # metabolite should not silently contribute zero to a sum).
        def _raw_sum(names: List[str]) -> pd.Series:
            raw = 10.0 ** out.loc[names]
            return raw.sum(axis=0)

        num_sum = _raw_sum(num_present)
        den_sum = _raw_sum(den_present)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_raw = num_sum / den_sum
            # 0/0 -> NaN, x/0 -> inf -> log10(inf) -> inf; clip negatives off.
            ratio_log = np.log10(ratio_raw.replace(0.0, np.nan))

        # Align to column index; ensure a Series named for the new row.
        ratio_series = pd.Series(ratio_log, index=out.columns, name=name)
        out = out.drop(index=[name], errors='ignore')
        # Use to_frame().T (not DataFrame([series])) so duplicate sample
        # column labels (duplicate injections) are preserved without
        # triggering a unique-index check.
        row = ratio_series.to_frame().T
        row.index = [name]
        out = pd.concat([out, row])

        added += 1
        logger.info(
            f"Added ratio feature {name!r} = "
            f"({' + '.join(num_present)}) / ({' + '.join(den_present)}); "
            f"{len(num_present)} numerator, {len(den_present)} denominator "
            f"feature(s) used; {int(ratio_series.notna().sum())}/{len(ratio_series)} "
            f"samples non-NaN."
        )

    logger.info(f"Feature-ratio builder: added {added} ratio feature(s); "
                f"features now {out.shape[0]}.")
    return out
