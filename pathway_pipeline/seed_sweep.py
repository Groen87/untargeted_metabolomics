#!/usr/bin/env python3
"""Label-blind split-stability sweep for the frozen pathway pipeline.

Re-runs the feature-engineering pipeline across multiple validation-split
seeds and reports how the NORMAL flag rates distribute between the
development and validation halves. This diagnoses whether the frozen
seed's validation-specificity draw is typical across splits or an
outlier -- a variance question, answerable from normal-reference evidence
only.

Protocol notes:
- ``run_evaluation`` is forced false: IMD labels are never read, no
  sensitivity/specificity is computed, and the one-shot of any frozen
  version is untouched.
- Every reported number concerns the normal reference (dev/val normal
  flag rates, per-channel flag counts, hygiene exclusions) or quantities
  the pipeline already logs label-blind in STEP 5 (group sizes).
- Choosing a seed AFTER seeing its validation metrics (seed shopping) is
  not a valid use of this output; the sweep exists to characterize split
  variance, and a reportable split change is a new frozen version with
  its own one-shot evaluation.

Usage:
    python -m pathway_pipeline.seed_sweep
    python -m pathway_pipeline.seed_sweep --n-seeds 20 --base-seed 20260923
    python -m pathway_pipeline.seed_sweep --seeds 20260923,20261923,20262923
"""

import argparse
import copy
import logging
from pathlib import Path

import pandas as pd
import yaml

from pathway_pipeline.main import run_pipeline

logger = logging.getLogger(__name__)

FORCED_SETTINGS = {
    "run_evaluation": False,
    "run_development_qc": False,
}


def _seed_values(args) -> list:
    if args.seeds:
        return [int(s) for s in str(args.seeds).split(",")]
    return [args.base_seed + 1000 * i for i in range(args.n_seeds)]


def _prepare_config(config_path: str, seed: int, out_dir: Path) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    config.update(copy.deepcopy(FORCED_SETTINGS))
    split = config.get("validation_split") or {}
    split["seed"] = seed
    split["enable"] = True
    config["validation_split"] = split
    used = out_dir / "config_used.yaml"
    with open(used, "w") as f:
        yaml.safe_dump(config, f)
    return str(used)


def _half_normal_stats(decisions: pd.DataFrame, half_mask: pd.Series,
                       prefix: str) -> dict:
    sel = decisions[decisions["sample_id"].map(half_mask).fillna(False)
                    .astype(bool) & (decisions["group"] == "normal")]
    n = len(sel)
    flagged = sel["flagged"].astype(bool)
    stats = {
        f"{prefix}_n_normals": n,
        f"{prefix}_flagged": int(flagged.sum()),
        f"{prefix}_flag_rate": float(flagged.mean()) if n else float("nan"),
    }
    for col, name in (("flagged_pathway_channel", "pathway"),
                      ("biomarker_flagged", "biomarker")):
        if col in sel.columns:
            stats[f"{prefix}_flagged_{name}"] = int(
                sel[col].fillna(False).astype(bool).sum())
    return stats


def _hygiene_excluded(out_dir: Path) -> int:
    path = out_dir / "reference_hygiene.csv"
    if not path.exists():
        return -1
    hygiene = pd.read_csv(path)
    return int(hygiene["excluded"].sum())


def run_seed_sweep(input_file: str, config_path: str,
                   output_dir: str, seeds: list) -> pd.DataFrame:
    """Run the pipeline once per split seed; return per-seed normal stats."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pipeline_logger = logging.getLogger("pathway_pipeline.main")
    pre_handlers = list(pipeline_logger.handlers)
    rows = []
    for seed in seeds:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        used_config = _prepare_config(config_path, seed, seed_dir)
        logger.info(f"Sweep run: seed={seed} -> {seed_dir}")
        result = run_pipeline(input_file=input_file,
                              output_dir=str(seed_dir),
                              config_path=used_config)
        for handler in list(pipeline_logger.handlers):
            if handler not in pre_handlers:
                pipeline_logger.removeHandler(handler)
                handler.close()
        decisions = result["sample_decisions"]
        groups = result["sample_group"]
        validation = result["validation_mask"]
        row = {"seed": seed}
        row.update(_half_normal_stats(decisions, ~validation, "dev"))
        row.update(_half_normal_stats(decisions, validation, "val"))
        row["hygiene_excluded"] = _hygiene_excluded(seed_dir)
        for group in ("normal", "imd", "other"):
            row[f"n_{group}"] = int((groups == group).sum())
        if validation is not None:
            for group in ("normal", "other"):
                row[f"n_{group}_val"] = int(
                    ((groups == group) & validation).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _log_summary(runs: pd.DataFrame) -> None:
    dev = runs["dev_flag_rate"]
    val = runs["val_flag_rate"]
    logger.info("=" * 70)
    logger.info("SPLIT-STABILITY SWEEP -- normal flag rates (label-blind)")
    logger.info("=" * 70)
    logger.info(f"Seeds: {len(runs)}  (first = frozen seed when defaults "
                "are used)")
    for col, name in (("dev_flag_rate", "dev"), ("val_flag_rate", "val")):
        s = runs[col]
        logger.info(
            f"{name} normal flag rate: mean {s.mean():.3f}, "
            f"median {s.median():.3f}, min {s.min():.3f}, max {s.max():.3f}"
        )
    n_high = int((val > 1.5 * dev).sum())
    logger.info(
        f"Seeds with val rate > 1.5x dev rate: {n_high} of {len(runs)}"
    )
    frozen = runs.iloc[0]
    rank = int((val > frozen["val_flag_rate"]).sum()) + 1
    logger.info(
        f"First-seed val rate {frozen['val_flag_rate']:.3f} ranks "
        f"{rank} of {len(runs)} by val flag rate (1 = highest)"
    )
    logger.info(
        "Interpretation: a val rate near the dev rate across most seeds "
        "means the frozen seed drew a flaggier validation half (split "
        "lottery); a stably elevated val rate across seeds means the "
        "re-grouped calibration genuinely differs between halves."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Label-blind split-stability sweep: normal flag rates "
                    "across validation-split seeds (run_evaluation forced "
                    "off; IMD labels are never read).")
    parser.add_argument("--input", default=None,
                        help="Path to the feature matrix CSV "
                             "(default: input_file from the config).")
    parser.add_argument("--config", default="pathway_pipeline/config/config.yaml",
                        help="Path to the frozen config YAML.")
    parser.add_argument("--output-dir", default="outputs/seed_sweep",
                        help="Directory for per-seed outputs and the sweep "
                             "CSV (default: outputs/seed_sweep).")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of seeds to sweep (default: 20).")
    parser.add_argument("--base-seed", type=int, default=20260923,
                        help="First seed; further seeds step by 1000 "
                             "(default: the frozen split seed 20260923).")
    parser.add_argument("--seeds", default=None,
                        help="Explicit comma-separated seed list; overrides "
                             "--n-seeds/--base-seed.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    input_file = args.input or config.get(
        "input_file", "data/merged_data_with_classification.csv")
    seeds = _seed_values(args)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    runs = run_seed_sweep(input_file=input_file, config_path=args.config,
                          output_dir=args.output_dir, seeds=seeds)
    out = Path(args.output_dir) / "seed_sweep_runs.csv"
    runs.to_csv(out, index=False)
    logger.info(f"Wrote {out}")
    _log_summary(runs)


if __name__ == "__main__":
    main()
