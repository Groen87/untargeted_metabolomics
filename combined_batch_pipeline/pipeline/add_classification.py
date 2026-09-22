#!/usr/bin/env python3
"""Merge classification labels into the combined batch metabolomics data.

Reads an outlier/classification file (``data/data outlier model.csv``) and a
merged data file (``data/merged_data.csv``), joins the classification columns
onto the merged data by sample identifier, and writes
``data/merged_data_with_classification.csv``.

The outlier file's ``Monster`` column is matched against the merged file's
``Sample`` column. The resulting ``Oordeel targeted``, ``Classification`` and
``Non-treated`` columns are placed immediately after the ``Sample`` column.

Usage:
    python combined_batch_pipeline/pipeline/add_classification.py
    python combined_batch_pipeline/pipeline/add_classification.py \
        --outlier-file data/data_outlier_model.csv \
        --merged-file data/merged_data.csv \
        --output-file data/merged_data_with_classification.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Candidate base directories whose `data/` folder is searched for the default
# input/output files. Ordered by preference: current working directory, the
# script's parent directory (combined_batch_pipeline/, sibling of the data
# folder), and the repository root.
_CANDIDATE_BASES = [
    Path.cwd(),
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parents[2],
]


def _resolve(path: str) -> str:
    """Resolve a relative path against candidate base directories.

    Searches each candidate base for the path; returns the first existing
    match. Absolute paths are returned unchanged. If no candidate contains
    the path, the path relative to the first existing `data/` folder (or the
    first candidate) is returned so the eventual error message stays helpful.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)

    candidates = [base / p for base in _CANDIDATE_BASES]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    # Nothing matched yet: prefer a base that has a `data/` folder so the
    # output (and a clear FileNotFoundError) points where the user keeps data.
    for base in _CANDIDATE_BASES:
        if (base / "data").is_dir():
            return str(base / p)
    return str(_CANDIDATE_BASES[0] / p)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, retrying with latin1 if utf-8 decoding fails."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def add_classification(
    outlier_file: str = "data/data outlier model.csv",
    merged_file: str = "data/merged_data.csv",
    output_file: str = "data/merged_data_with_classification.csv",
) -> pd.DataFrame:
    outlier_df = _read_csv(Path(_resolve(outlier_file)))
    merged_df = _read_csv(Path(_resolve(merged_file)))

    result = merged_df.merge(
        outlier_df[["Monster", "Oordeel targeted", "Classification", "Non-treated"]],
        left_on="Sample",
        right_on="Monster",
        how="left",
    )

    result = result.drop(columns=["Monster"])

    cols = list(result.columns)
    new_cols = []
    for col in cols:
        if col not in ("Oordeel targeted", "Classification", "Non-treated"):
            new_cols.append(col)
            if col == "Sample":
                new_cols.append("Oordeel targeted")
                new_cols.append("Classification")
                new_cols.append("Non-treated")

    result = result[new_cols]

    output_path = Path(_resolve(output_file))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Merge complete! Result saved to {output_file}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge classification labels into the merged metabolomics data."
    )
    parser.add_argument(
        "--outlier-file",
        default="data/data outlier model.csv",
        help="CSV with classification info (columns: Monster, Oordeel targeted, Classification, Non-treated).",
    )
    parser.add_argument(
        "--merged-file",
        default="data/merged_data.csv",
        help="Merged data CSV (column: Sample) to add classification to.",
    )
    parser.add_argument(
        "--output-file",
        default="data/merged_data_with_classification.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    add_classification(
        outlier_file=args.outlier_file,
        merged_file=args.merged_file,
        output_file=args.output_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
