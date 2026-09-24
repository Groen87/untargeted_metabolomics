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
- runs/sample_flag_stability.csv     per sample across ALL splits: in how
                                     many of the pre-declared splits the
                                     sample flagged (flag_rate), per-channel
                                     counts, and modal evidence. Sorted by
                                     flag_rate -- the top rows are the
                                     robust flags. Unknown/unlabeled samples
                                     sit in group 'other' and are scored like
                                     anyone else (flagging never uses
                                     labels), so this table is where to find
                                     robustly flagged unknowns for clinical
                                     chart review.
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
    """Run the frozen pipeline per pre-declared seed; evaluate validation.

    Returns ``(runs, missed, stability, miss_freq)`` where ``runs`` holds
    per-seed validation-half metrics, ``missed`` the missed-IMD evidence
    rows, ``stability`` the per-sample flag counts across all splits, and
    ``miss_freq`` per-IMD miss counts over splits.
    """
    out_root = Path(output_dir)
    runs_dir = out_root / "runs"
    per_seed = out_root / "per_seed"
    runs_dir.mkdir(parents=True, exist_ok=True)
    per_seed.mkdir(parents=True, exist_ok=True)
    pipeline_logger = logging.getLogger("pathway_pipeline.main")
    pre_handlers = list(pipeline_logger.handlers)
    rows = []
    missed_frames = []
    per_sample_frames = []
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
        stability_cols = [c for c in ("sample_id", "group", "flagged",
                                      "validation",
                                      "flagged_pathway_channel",
                                      "biomarker_flagged", "top_excess",
                                      "top_pathway_name", "top_biomarker",
                                      "top_disease")
                          if c in decisions.columns]
        per_seed_samples = decisions[stability_cols].copy()
        per_seed_samples.insert(0, "seed", seed)
        per_sample_frames.append(per_seed_samples)
        row.update(_normal_context(decisions, ~validation, "dev"))
        row.update(_normal_context(decisions, validation, "val"))
        row["hygiene_excluded"] = _hygiene_excluded(seed_dir)
        for group in ("normal", "imd", "other"):
            row[f"n_total_{group}"] = int((groups == group).sum())
        rows.append(row)
    runs = pd.DataFrame(rows)
    missed = (pd.concat(missed_frames, ignore_index=True)
              if missed_frames else pd.DataFrame(
                  columns=["seed", "sample_id", *MISSED_EVIDENCE_COLS]))
    stability = _sample_flag_stability(per_sample_frames)
    miss_freq = _imd_miss_frequency(missed, per_sample_frames, len(seeds))
    return runs, missed, stability, miss_freq


def _modal_evidence(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]
    if values.empty:
        return ""
    return values.value_counts().index[0]


def _imd_miss_frequency(missed: pd.DataFrame, per_sample_frames: list,
                         n_seeds: int) -> pd.DataFrame:
    """Per IMD: in how many splits it landed in validation and was missed.

    Combining the missed-IMD evidence with the per-seed group/split
    membership distinguishes systematic misses (missed in most splits
    whenever in validation -- phenotype/coverage problem) from split
    lottery (missed in a small minority).
    """
    if not per_sample_frames:
        return pd.DataFrame()
    all_samples = pd.concat(per_sample_frames, ignore_index=True)
    imd = all_samples[all_samples["group"] == "imd"]
    if imd.empty:
        return pd.DataFrame(columns=["sample_id", "n_splits_validation",
                                     "n_splits_missed", "miss_rate",
                                     "mean_top_excess", "top_pathway_name",
                                     "top_biomarker"])
    imd["validation"] = imd["validation"].fillna(False).astype(bool)
    grouped = imd.groupby("sample_id", sort=False)
    freq = pd.DataFrame({
        "n_splits_validation": grouped["validation"].sum().astype(int),
        "mean_top_excess": grouped["top_excess"].mean()
        if "top_excess" in imd.columns else float("nan"),
        "top_pathway_name": grouped["top_pathway_name"].agg(_modal_evidence)
        if "top_pathway_name" in imd.columns else "",
        "top_biomarker": grouped["top_biomarker"].agg(_modal_evidence)
        if "top_biomarker" in imd.columns else "",
    })
    miss_counts = (missed[missed["sample_id"].isin(freq.index)]
                   .groupby("sample_id").size()
                   if not missed.empty else pd.Series(
                       dtype=int))
    freq["n_splits_missed"] = (freq.index.map(miss_counts).fillna(0)
                               .astype(int))
    freq["miss_rate"] = freq["n_splits_missed"] / freq["n_splits_validation"]
    freq = freq.sort_values(["miss_rate", "n_splits_missed"],
                            ascending=False)
    return freq.reset_index()


def _sample_flag_stability(per_sample_frames: list) -> pd.DataFrame:
    """Per-sample flag counts across all splits, sorted by flag_rate."""
    if not per_sample_frames:
        return pd.DataFrame()
    all_samples = pd.concat(per_sample_frames, ignore_index=True)
    all_samples["flagged"] = all_samples["flagged"].astype(bool)
    grouped = all_samples.groupby("sample_id", sort=False)
    stability = pd.DataFrame({
        "group": grouped["group"].first(),
        "n_splits": grouped.size(),
        "n_splits_flagged": grouped["flagged"].sum().astype(int),
        "n_splits_pathway_channel": grouped["flagged_pathway_channel"]
            .sum().astype(int) if "flagged_pathway_channel"
            in all_samples.columns else 0,
        "n_splits_biomarker_channel": grouped["biomarker_flagged"]
            .sum().astype(int) if "biomarker_flagged"
            in all_samples.columns else 0,
        "mean_top_excess": grouped["top_excess"].mean()
            if "top_excess" in all_samples.columns else float("nan"),
        "top_pathway_name": grouped["top_pathway_name"]
            .agg(_modal_evidence) if "top_pathway_name"
            in all_samples.columns else "",
        "top_biomarker": grouped["top_biomarker"].agg(_modal_evidence)
            if "top_biomarker" in all_samples.columns else "",
        "top_disease": grouped["top_disease"].agg(_modal_evidence)
            if "top_disease" in all_samples.columns else "",
    }).reset_index()
    stability["flag_rate"] = (stability["n_splits_flagged"]
                              / stability["n_splits"])
    stability = stability.sort_values(
        ["flag_rate", "mean_top_excess"], ascending=[False, False])
    return stability.reset_index(drop=True)


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
                  agg: pd.DataFrame, missed: pd.DataFrame,
                  stability: pd.DataFrame = None,
                  miss_freq: pd.DataFrame = None) -> None:
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
    miss_freq_table = ""
    if miss_freq is not None and not miss_freq.empty:
        show = miss_freq.copy()
        show["miss_rate"] = show["miss_rate"].map("{:.0%}".format)
        show["mean_top_excess"] = show["mean_top_excess"].map(
            lambda v: "%.2f" % v if pd.notna(v) else "")
        miss_freq_table = ("\n\n### IMD miss frequency over splits\n\n"
                           "Systematic misses (high miss_rate whenever in "
                           "validation) point at phenotype/coverage gaps; "
                           "low miss_rate is split lottery.\n\n"
                           + _md_table(show))
    missed_table = ""
    if not missed.empty:
        show = missed.copy()
        for col in ("top_excess", "max_metabolite_z", "sample_p_value"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: "%.3f" % v
                                         if pd.notna(v) else "")
        missed_table = "\n\n### Missed IMD samples\n\n" + _md_table(show)
    stability_table = ""
    if stability is not None and not stability.empty:
        unknowns = stability[stability["group"] == "other"]
        flagged_unknowns = unknowns[unknowns["flag_rate"] > 0].head(30)
        if not flagged_unknowns.empty:
            show = flagged_unknowns.copy()
            show["flag_rate"] = show["flag_rate"].map("{:.0%}".format)
            show["mean_top_excess"] = show["mean_top_excess"].map(
                lambda v: "%.2f" % v if pd.notna(v) else "")
            stability_table = ("\n\n### Robustly flagged samples without an "
                              "IMD label (chart-review candidates; top 30 "
                              "by flag rate)\n\n" + _md_table(show))
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
        miss_freq_table,
        stability_table,
        "",
        "## Files",
        "",
        "- `runs/multi_split_runs.csv` -- per-seed metrics and context",
        "- `runs/multi_split_aggregate.csv` -- mean/median/sd/min/max "
        "per metric",
        "- `runs/missed_imd_evidence.csv` -- every missed IMD with its "
        "evidence",
        "- `runs/imd_miss_frequency.csv` -- per IMD: splits in validation "
        "vs splits missed and miss rate -- separates systematic misses "
        "from split lottery",
        "- `runs/sample_flag_stability.csv` -- per sample across all "
        "splits: flag rate over splits, per-channel counts, modal "
        "evidence; sorted by flag rate. Unknowns sit in group 'other' and "
        "are scored label-blind, so robustly flagged unknowns here are "
        "the chart-review candidates.",
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
    runs, missed, stability, miss_freq = run_multi_split_evaluation(
        input_file=input_file, config_path=str(config_path),
        output_dir=str(out_root), seeds=seeds)
    runs_dir = out_root / "runs"
    runs.to_csv(runs_dir / "multi_split_runs.csv", index=False)
    agg = _aggregate(runs)
    agg.to_csv(runs_dir / "multi_split_aggregate.csv", index=False)
    missed.to_csv(runs_dir / "missed_imd_evidence.csv", index=False)
    if not stability.empty:
        stability.to_csv(runs_dir / "sample_flag_stability.csv", index=False)
    if not miss_freq.empty:
        miss_freq.to_csv(runs_dir / "imd_miss_frequency.csv", index=False)
    _write_report(out_root, seeds, runs, agg, missed, stability, miss_freq)
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
