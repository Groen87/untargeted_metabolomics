#!/usr/bin/env python3
"""Pre-declared multi-split evaluation of the frozen pipeline (one-shot).

Scientific protocol
-------------------
This is the label-aware evaluation protocol for a NEW frozen version: the
validation-split seed list is pre-declared here BEFORE any label-aware
read, the pipeline runs once per seed, and the validation half of each
split is evaluated. The whole protocol consumes the version's single
one-shot: read the results, then change nothing and re-report.

Rationale: the dev/val 50/50 split re-stratifies with cohort changes, and
the pathway channel's max-excess decision constant is the p95 of a
max-statistic over ~105 dev normals -- the noisiest calibration number in
the pipeline. A single split therefore has high variance (the seed sweep
measured val normal flag rates from 7.5% to 26.4% across 20 pre-declared
seeds). Reporting the metric DISTRIBUTION over pre-declared splits is as
fair as a single frozen split (nothing is tuned post-hoc) and much more
informative: the mean is far more stable than any single draw, while the
range quantifies the split lottery explicitly.

Interpretation caveat: the splits share samples (correlated replicates),
so the per-seed values are NOT independent; quote mean +/- range, not a
p-value over seeds.

Outputs (default outputs/multi_split_evaluation/)
------------------------------------------------
- runs/multi_split_runs.csv          one row per seed: validation-half
                                     sensitivity/specificity/AUC (+ CIs),
                                     denominators, missed-IMD count, and
                                     the label-blind context numbers
                                     (dev/val normal flag rates, channel
                                     flags, hygiene exclusions)
- runs/multi_split_aggregate.csv     mean/median/sd/min/max per metric
- runs/missed_imd_evidence.csv       every missed IMD with its evidence,
                                     one row per (seed, sample)
- MULTI_SPLIT_REPORT.md              human-readable write-up summary
- per_seed/seed_<seed>/              full pipeline outputs for each split
                                     (includes each seed's
                                     evaluation_summary.csv and
                                     config_used.yaml for audit)

Usage
-----
    python -m pathway_pipeline.multi_split_evaluation
    python -m pathway_pipeline.multi_split_evaluation --n-seeds 10
    python -m pathway_pipeline.multi_split_evaluation --seeds 20260923,20262923
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathway_pipeline.main import run_pipeline

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = REPO_ROOT / "pathway_pipeline" / "config" / "config.yaml"

FORCED_SETTINGS = {
    "run_evaluation": True,
    "evaluate_dev_half": False,
    "run_development_qc": False,
}

METRICS = ("sensitivity", "specificity", "auc",
           "dev_flag_rate", "val_flag_rate")

MISSED_EVIDENCE_COLS = ("n_flagged_pathways", "sample_p_value",
                        "top_pathway_name", "top_excess",
                        "max_metabolite_z", "top_metabolite",
                        "n_flagged_biomarkers", "top_biomarker",
                        "top_disease")


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


def _normal_context(decisions: pd.DataFrame, half_mask: pd.Series,
                    prefix: str) -> dict:
    sel = decisions[decisions["sample_id"].map(half_mask)
                    .fillna(False).astype(bool)
                    & (decisions["group"] == "normal")]
    flagged = sel["flagged"].astype(bool)
    stats = {
        f"{prefix}_n_normals": len(sel),
        f"{prefix}_flag_rate": (float(flagged.mean())
                                if len(sel) else float("nan")),
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


def _missed_imds(decisions: pd.DataFrame, validation: pd.Series,
                 seed: int) -> pd.DataFrame:
    val = decisions["sample_id"].map(validation).fillna(False).astype(bool)
    missed = decisions[(decisions["group"] == "imd") & val
                       & ~decisions["flagged"].astype(bool)].copy()
    if missed.empty:
        cols = ["seed", "sample_id", *MISSED_EVIDENCE_COLS]
        return pd.DataFrame(columns=cols)
    missed.insert(0, "seed", seed)
    keep = ["seed", "sample_id"] + [c for c in MISSED_EVIDENCE_COLS
                                    if c in missed.columns]
    return missed[keep]


def run_multi_split_evaluation(input_file: str, config_path: str,
                               output_dir: str, seeds: list) -> pd.DataFrame:
    """Run the frozen pipeline per pre-declared seed; evaluate validation."""
    out_root = Path(output_dir)
    runs_dir = out_root / "runs"
    per_seed = out_root / "per_seed"
    runs_dir.mkdir(parents=True, exist_ok=True)
    per_seed.mkdir(parents=True, exist_ok=True)
    pipeline_logger = logging.getLogger("pathway_pipeline.main")
    pre_handlers = list(pipeline_logger.handlers)
    rows = []
    missed_frames = []
    for seed in seeds:
        seed_dir = per_seed / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        used_config = _prepare_config(config_path, seed, seed_dir)
        logger.info(f"Multi-split run: seed={seed} -> {seed_dir}")
        result = run_pipeline(input_file=input_file,
                              output_dir=str(seed_dir),
                              config_path=used_config)
        for handler in list(pipeline_logger.handlers):
            if handler not in pre_handlers:
                pipeline_logger.removeHandler(handler)
                handler.close()
        decisions = result["sample_decisions"]
        validation = result["validation_mask"]
        groups = result["sample_group"]
        evaluation = (result.get("evaluation") or {}).get("validation")
        if evaluation is None:
            raise RuntimeError(
                f"seed {seed}: validation-half evaluation missing; check "
                "that validation_split.enable is true in the config")
        row = {"seed": seed}
        for key in ("sensitivity", "specificity", "auc", "n_normal", "n_imd"):
            row[key] = evaluation[key]
        for lo_hi, key in (("sensitivity_ci", "sensitivity"),
                           ("specificity_ci", "specificity"),
                           ("auc_ci", "auc")):
            lo, hi = evaluation[lo_hi]
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        seed_missed = _missed_imds(decisions, validation, seed)
        missed_frames.append(seed_missed)
        row["n_missed_imd"] = len(seed_missed)
        row.update(_normal_context(decisions, ~validation, "dev"))
        row.update(_normal_context(decisions, validation, "val"))
        row["hygiene_excluded"] = _hygiene_excluded(seed_dir)
        for group in ("normal", "imd", "other"):
            row[f"n_{group}"] = int((groups == group).sum())
        rows.append(row)
    runs = pd.DataFrame(rows)
    missed = (pd.concat(missed_frames, ignore_index=True)
              if missed_frames else pd.DataFrame(
                  columns=["seed", "sample_id", *MISSED_EVIDENCE_COLS]))
    return runs, missed


def _aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in runs.columns:
            continue
        s = pd.to_numeric(runs[metric], errors="coerce").dropna()
        rows.append({
            "metric": metric,
            "n_seeds": len(s),
            "mean": s.mean() if len(s) else float("nan"),
            "median": s.median() if len(s) else float("nan"),
            "sd": s.std(ddof=1) if len(s) > 1 else float("nan"),
            "min": s.min() if len(s) else float("nan"),
            "max": s.max() if len(s) else float("nan"),
        })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return float_fmt % v
        return str(v)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for _, row in df.iterrows())
    return "\n".join([header, sep, body])


def _write_report(out_root: Path, seeds: list, runs: pd.DataFrame,
                  agg: pd.DataFrame, missed: pd.DataFrame) -> None:
    per_metric = []
    for metric in ("sensitivity", "specificity", "auc"):
        sel = agg[agg["metric"] == metric]
        if sel.empty:
            continue
        r = sel.iloc[0]
        per_metric.append(
            f"- **{metric}**: mean {r['mean']:.3f}, median {r['median']:.3f}, "
            f"range {r['min']:.3f}-{r['max']:.3f} "
            f"(sd {r['sd']:.3f} over {int(r['n_seeds'])} splits)")
    dev = agg[agg["metric"] == "dev_flag_rate"]
    val = agg[agg["metric"] == "val_flag_rate"]
    context = ""
    if not dev.empty and not val.empty:
        context = (
            f"\nLabel-blind context: dev-half normal flag rate "
            f"{dev.iloc[0]['mean']:.3f} (sd {dev.iloc[0]['sd']:.3f}), "
            f"val-half normal flag rate {val.iloc[0]['mean']:.3f} "
            f"(sd {val.iloc[0]['sd']:.3f}) -- the gap is the pathway "
            f"channel's split-dependent max-excess threshold.")
    missed_table = ""
    if not missed.empty:
        show = missed.copy()
        for col in ("top_excess", "max_metabolite_z", "sample_p_value"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: "%.3f" % v
                                         if pd.notna(v) else "")
        missed_table = "\n\n### Missed IMD samples\n\n" + _md_table(show)
    lines = [
        "# Multi-Split Evaluation Report",
        "",
        "Pre-declared validation-split seeds "
        f"({len(seeds)}): {', '.join(str(s) for s in seeds)}",
        "",
        "Protocol: the pipeline ran once per seed; the validation half of",
        "each split was evaluated against the frozen configuration. This",
        "is the label-aware one-shot for this frozen version: the seed",
        "list was declared before any label-aware read, and nothing was",
        "tuned after seeing these results. Splits share samples, so the",
        "per-seed values are correlated replicates -- quote mean with",
        "range, not a p-value over seeds.",
        "",
        "## Aggregate results",
        "",
        *per_metric,
        context,
        "",
        "## Per-split results",
        "",
        _md_table(runs[["seed", "n_imd", "sensitivity", "n_normal",
                        "specificity", "auc", "dev_flag_rate",
                        "val_flag_rate", "hygiene_excluded"]]),
        missed_table,
        "",
        "## Files",
        "",
        "- `runs/multi_split_runs.csv` -- per-seed metrics and context",
        "- `runs/multi_split_aggregate.csv` -- mean/median/sd/min/max "
        "per metric",
        "- `runs/missed_imd_evidence.csv` -- every missed IMD with its "
        "evidence",
        "- `per_seed/seed_<seed>/` -- full pipeline outputs per split "
        "(each contains `evaluation_summary.csv` and `config_used.yaml`)",
    ]
    (out_root / "MULTI_SPLIT_REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-declared multi-split evaluation (one-shot, "
                    "label-aware): validation-half metrics per split seed, "
                    "aggregated for reporting.")
    parser.add_argument("--input", default=None,
                        help="Path to the feature matrix CSV "
                             "(default: input_file from the config).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to the frozen config YAML (default: the "
                             "repo's pathway_pipeline/config/config.yaml).")
    parser.add_argument("--output-dir",
                        default="outputs/multi_split_evaluation",
                        help="Dedicated output folder "
                             "(default: outputs/multi_split_evaluation).")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of pre-declared seeds (default: 20).")
    parser.add_argument("--base-seed", type=int, default=20260923,
                        help="First seed; further seeds step by 1000 "
                             "(default: the frozen split seed 20260923).")
    parser.add_argument("--seeds", default=None,
                        help="Explicit comma-separated seed list; overrides "
                             "--n-seeds/--base-seed.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        candidate = Path.cwd() / config_path
        config_path = candidate if candidate.exists() else REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    input_file = args.input or config.get(
        "input_file", "data/merged_data_with_classification.csv")
    if not Path(input_file).is_absolute() and not Path(input_file).exists():
        input_file = str(REPO_ROOT / input_file)
    seeds = _seed_values(args)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Pre-declared seeds (%d): %s", len(seeds), seeds)
    logger.info("This protocol consumes the frozen version's one-shot: "
                "the seed list is fixed before the label-aware read.")
    out_root = Path(args.output_dir)
    runs, missed = run_multi_split_evaluation(
        input_file=input_file, config_path=str(config_path),
        output_dir=str(out_root), seeds=seeds)
    runs_dir = out_root / "runs"
    runs.to_csv(runs_dir / "multi_split_runs.csv", index=False)
    agg = _aggregate(runs)
    agg.to_csv(runs_dir / "multi_split_aggregate.csv", index=False)
    missed.to_csv(runs_dir / "missed_imd_evidence.csv", index=False)
    _write_report(out_root, seeds, runs, agg, missed)
    logger.info("Wrote multi_split_runs.csv, multi_split_aggregate.csv, "
                "missed_imd_evidence.csv, MULTI_SPLIT_REPORT.md to %s",
                out_root)
    logger.info("=" * 70)
    logger.info("MULTI-SPLIT EVALUATION -- validation half per split")
    logger.info("=" * 70)
    for metric in ("sensitivity", "specificity", "auc"):
        sel = agg[agg["metric"] == metric]
        if not sel.empty:
            r = sel.iloc[0]
            logger.info("%s: mean %.3f, median %.3f, range %.3f-%.3f",
                        metric, r["mean"], r["median"],
                        r["min"], r["max"])


if __name__ == "__main__":
    main()
