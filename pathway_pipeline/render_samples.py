"""Render the report-only per-sample figures for arbitrary sample IDs
directly from a saved pipeline output directory -- without re-running
the pipeline.

Reads the CSVs a finished run already wrote (single run or one
``per_seed/seed_<seed>`` split of a multi-split evaluation) and rebuilds
the exact figure inputs STEP 8e used: the scored z matrix, the Stouffer
scores (re-merging the promoted IEMbase disease panels the same way
main.py does), the flags, the resolved disease panels, the scale^2
weights, and the normal mask. Nothing is recomputed from raw features
and nothing feeds back into scoring or flagging; the figures are the
same report-only functions the pipeline already calls.

Usage:
    python -m pathway_pipeline.render_samples <output_dir> --ids ID1,ID2
    python -m pathway_pipeline.render_samples <multi_split_dir> \
        --seed 20260923 --ids ID1,ID2
    python -m pathway_pipeline.render_samples <output_dir> --top-flagged 5

Outputs land in <output_dir>/figures/sample_reports/ (the same location
STEP 8e uses), so the frozen run's directory stays the single source of
truth for every figure.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pathway_pipeline.pipeline.biomarkers import (
    load_disease_biomarker_table, resolve_disease_biomarkers)
from pathway_pipeline.pipeline.visualize import (
    sample_overview_figure, sample_report_figures,
    top_pathway_waterfall_figure)

logger = logging.getLogger("pathway_pipeline.render_samples")


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _disease_resolved_from_run(run_dir: Path) -> Optional[pd.DataFrame]:
    """Rebuild the resolved disease panels from the run's own audit CSVs.

    ``disease_table_audit.csv`` holds every table row with its resolved
    feature list (semicolon-joined); ``disease_marker_name_matches.csv``
    holds the name-matched markers that joined panels without an HMDB
    code. Together they cover every member source the panel scores used,
    so the waterfall needs neither the Excel file nor a re-resolution.
    """
    audit = _read_csv(run_dir / "disease_table_audit.csv")
    if audit is None:
        return None
    cols = ["disease", "biomarker", "hmdb_id", "direction", "features"]
    resolved = audit.reindex(columns=cols)
    resolved = resolved[resolved["features"].fillna("") != ""]
    name_matches = _read_csv(run_dir / "disease_marker_name_matches.csv")
    if name_matches is not None and not name_matches.empty:
        extra = pd.DataFrame({
            "disease": name_matches["disease"],
            "biomarker": name_matches["marker"],
            "hmdb_id": "",
            "direction": name_matches["direction"],
            "features": name_matches["matched_features"],
        })
        extra = extra[extra["features"].fillna("") != ""]
        resolved = pd.concat([resolved, extra], ignore_index=True)
    if resolved.empty:
        return pd.DataFrame(columns=cols)
    return resolved


def _ratio_members_from_config(config: dict,
                               zscores: pd.DataFrame
                               ) -> pd.DataFrame:
    """Panel member terms declared as ratio_biomarkers in the run config.

    ``biomarker_channel.ratio_biomarkers`` entries joined disease panels
    as extra Stouffer member terms (RATIO:<name>); the waterfall needs
    the same terms so a panel's ratio members appear alongside its
    metabolite members.
    """
    cols = ["disease", "biomarker", "hmdb_id", "direction", "features"]
    channel = config.get("biomarker_channel") or {}
    if not isinstance(channel, dict):
        return pd.DataFrame(columns=cols)
    specs = channel.get("ratio_biomarkers") or []
    rows = []
    for spec in specs or []:
        ratio = str(spec.get("ratio", "")).strip()
        disease = str(spec.get("disease", "")).strip()
        if not ratio or not disease or ratio not in zscores.columns:
            continue
        rows.append({"disease": disease, "biomarker": ratio,
                     "hmdb_id": "", "direction": spec.get("direction"),
                     "features": ratio})
    return pd.DataFrame(rows, columns=cols)


def load_run_outputs(run_dir: Path, config_path: Optional[Path] = None
                     ) -> Optional[dict]:
    """Rebuild the STEP 8e figure inputs from a saved run directory.

    Every frame comes from the run's own CSVs (the frozen outputs) plus
    the config the run used. When the disease-panel promotion was
    active, the promoted panels are merged back into the pathway scores
    with the same replacement rule main.py applies at STEP 7b, so the
    figures see the same pathway channel the flags were computed on.
    """
    import yaml

    if config_path is None:
        config_path = run_dir / "config_used.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    zscores = _read_csv(run_dir / "metabolite_zscores.csv")
    if zscores is None:
        logger.error(f"No metabolite_zscores.csv in {run_dir}")
        return None
    if "sample_id" in zscores.columns:
        zscores = zscores.set_index("sample_id")
    else:
        zscores = zscores.set_index(zscores.columns[0])
    zscores.index = zscores.index.astype(str)
    zscores.index.name = "sample_id"

    demoted = [f for f in (config.get("demoted_features") or [])
               if f in zscores.columns]
    if demoted:
        zscores_scored = zscores.drop(columns=demoted)
    else:
        zscores_scored = zscores

    feature_to_pathway = _read_csv(run_dir / "feature_to_pathway.csv")
    feature_to_hmdb = _read_csv(run_dir / "feature_to_hmdb.csv")
    coverage = _read_csv(run_dir / "pathway_coverage.csv")
    pathway_scores = _read_csv(run_dir / "pathway_stouffer_scores.csv")
    pathway_flags = _read_csv(run_dir / "pathway_flags.csv")
    metabolite_flags = _read_csv(run_dir / "metabolite_flags.csv")
    reference_stats = _read_csv(run_dir / "reference_stats.csv")
    decisions = _read_csv(run_dir / "sample_decisions.csv")
    hygiene = _read_csv(run_dir / "reference_hygiene.csv")
    split = _read_csv(run_dir / "cohort_split.csv")
    for frame in (pathway_scores, pathway_flags, metabolite_flags,
                  decisions, hygiene, split):
        if frame is not None and "sample_id" in frame.columns:
            frame["sample_id"] = frame["sample_id"].astype(str)

    if pathway_scores is None or pathway_flags is None:
        logger.error(f"Missing pathway scores/flags CSVs in {run_dir}")
        return None

    disease_resolved = None
    promoted_diseases: List[str] = []
    panel_scores = _read_csv(run_dir / "disease_panel_scores.csv")
    if panel_scores is not None and not panel_scores.empty:
        disease_resolved = _disease_resolved_from_run(run_dir)
        if disease_resolved is None:
            channel_cfg = config.get("biomarker_channel") or {}
            table_file = (channel_cfg.get("disease_table_file")
                          if isinstance(channel_cfg, dict) else None)
            if table_file:
                table_path = Path(table_file)
                if not table_path.is_absolute():
                    for base in (run_dir, run_dir.parent, Path.cwd(),
                                 Path(__file__).resolve().parents[1]):
                        candidate = base / table_path
                        if candidate.exists():
                            table_path = candidate
                            break
                disease_table = load_disease_biomarker_table(
                    str(table_path))
                if (not disease_table.empty and feature_to_hmdb is not None
                        and coverage is not None):
                    disease_resolved, _, _ = resolve_disease_biomarkers(
                        disease_table, feature_to_hmdb, coverage)
        if disease_resolved is None:
            disease_resolved = pd.DataFrame(
                columns=["disease", "biomarker", "hmdb_id", "direction",
                         "features"])
        ratio_members = _ratio_members_from_config(config, zscores_scored)
        if not ratio_members.empty:
            disease_resolved = pd.concat(
                [disease_resolved, ratio_members], ignore_index=True)
        promoted_diseases = sorted(panel_scores["pathway_name"].unique())
        panel_smps_real = {s for s in panel_scores["smp_id"].unique()
                           if not str(s).startswith("DISEASE-")}
        replaced = pathway_scores["smp_id"].isin(panel_smps_real)
        if replaced.any():
            pathway_scores = pd.concat(
                [pathway_scores[~replaced], panel_scores], ignore_index=True)
        else:
            pathway_scores = pd.concat(
                [pathway_scores, panel_scores], ignore_index=True)

    feature_scale_weights = None
    if (bool(config.get("scale_weighted_metabolites", True))
            and reference_stats is not None
            and {"feature", "scale"}.issubset(reference_stats.columns)):
        feature_scale_weights = {
            row["feature"]: float(row["scale"]) ** 2
            for _, row in reference_stats.iterrows()}

    normal_mask = None
    if split is not None and {"sample_id", "group"}.issubset(split.columns):
        normal_mask = pd.Series(
            (split["group"] == "normal").to_numpy(), index=split["sample_id"])
    elif decisions is not None and "group" in decisions.columns:
        normal_mask = pd.Series(
            (decisions["group"] == "normal").to_numpy(),
            index=decisions["sample_id"])
    else:
        normal_mask = pd.Series(False, index=zscores.index)
    if hygiene is not None and {"sample_id", "excluded"}.issubset(
            hygiene.columns):
        excluded = set(hygiene.loc[
            hygiene["excluded"].astype(bool), "sample_id"].astype(str))
        normal_mask = normal_mask & ~normal_mask.index.astype(str).isin(
            excluded)

    return {
        "zscores_scored": zscores_scored,
        "pathway_scores": pathway_scores,
        "pathway_flags": pathway_flags,
        "metabolite_flags": metabolite_flags,
        "feature_to_pathway": feature_to_pathway,
        "disease_resolved": disease_resolved,
        "promoted_diseases": promoted_diseases,
        "feature_scale_weights": feature_scale_weights,
        "normal_mask": normal_mask,
        "decisions": decisions,
    }


def render_samples(run_dir: Path, sample_ids: List[str],
                   out_dir: Optional[Path] = None, dpi: int = 200) -> int:
    """Render every per-sample report figure for the given sample IDs."""
    frames = load_run_outputs(run_dir)
    if frames is None:
        return 1
    zscores = frames["zscores_scored"]
    known = [s for s in sample_ids if s in zscores.index]
    unknown = [s for s in sample_ids if s not in zscores.index]
    if unknown:
        logger.warning("Samples not in the z matrix (skipped): %s",
                       ", ".join(map(str, unknown)))
    if not known:
        logger.error("None of the requested samples are in %s",
                     run_dir)
        return 1
    if out_dir is None:
        out_dir = run_dir / "figures" / "sample_reports"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sid in known:
        if top_pathway_waterfall_figure(
                sample_id=sid,
                zscores=zscores,
                feature_to_pathway=frames["feature_to_pathway"],
                disease_resolved=frames["disease_resolved"],
                pathway_scores=frames["pathway_scores"],
                pathway_flags=frames["pathway_flags"],
                feature_scale_weights=frames["feature_scale_weights"],
                out_dir=out_dir,
                dpi=dpi):
            written += 1
        if sample_overview_figure(
                sample_id=sid,
                zscores=zscores,
                pathway_scores=frames["pathway_scores"],
                pathway_flags=frames["pathway_flags"],
                promoted_diseases=frames["promoted_diseases"],
                normal_mask=frames["normal_mask"],
                out_dir=out_dir,
                dpi=dpi):
            written += 1
        written += sum(sample_report_figures(
            sample_id=sid,
            zscores=zscores,
            pathway_scores=frames["pathway_scores"],
            pathway_flags=frames["pathway_flags"],
            metabolite_flags=frames["metabolite_flags"],
            normal_mask=frames["normal_mask"],
            out_dir=out_dir,
            dpi=dpi).values())
    logger.info("Rendered %d figure file(s) for %d sample(s) into %s",
                written, len(known), out_dir)
    return 0


def _top_flagged_ids(decisions: pd.DataFrame, n: int) -> List[str]:
    if decisions is None or decisions.empty or "top_excess" not in (
            decisions.columns):
        return []
    top = decisions.sort_values("top_excess", ascending=False).head(n)
    return [str(s) for s in top["sample_id"]]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render per-sample report figures from saved pipeline "
                    "outputs (no pipeline re-run; report-only).")
    parser.add_argument("output_dir", type=Path,
                        help="saved run output directory (single run or a "
                             "multi-split root with --seed)")
    parser.add_argument("--ids", type=str, default="",
                        help="comma-separated sample IDs to render")
    parser.add_argument("--seed", type=int, default=None,
                        help="render from per_seed/seed_<seed> of a "
                             "multi-split evaluation directory")
    parser.add_argument("--top-flagged", type=int, default=0,
                        help="render the top-N flagged samples instead of "
                             "explicit IDs")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: "
                             "<output_dir>/figures/sample_reports)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--config", type=Path, default=None,
                        help="config file the run used when "
                             "config_used.yaml is absent")
    parser.add_argument("--list-samples", action="store_true",
                        help="print the run's flagged sample IDs and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s %(name)s: %(message)s")

    run_dir = args.output_dir
    if args.seed is not None:
        run_dir = run_dir / "per_seed" / f"seed_{args.seed}"
    if not run_dir.exists():
        logger.error("Output directory not found: %s", run_dir)
        return 1

    if args.list_samples:
        decisions = _read_csv(run_dir / "sample_decisions.csv")
        if decisions is None:
            logger.error("No sample_decisions.csv in %s", run_dir)
            return 1
        if "flagged" in decisions.columns:
            decisions = decisions[decisions["flagged"].astype(bool)]
        for sid in decisions["sample_id"]:
            print(sid)
        return 0

    frames = load_run_outputs(
        run_dir, config_path=args.config)
    if frames is None:
        return 1
    sample_ids: List[str] = []
    if args.ids:
        sample_ids = [s.strip() for s in str(args.ids).split(",") if s.strip()]
    elif args.top_flagged > 0:
        sample_ids = _top_flagged_ids(frames["decisions"], args.top_flagged)
        if not sample_ids:
            logger.error("No flagged samples found in %s", run_dir)
            return 1
    else:
        parser.error("give --ids or --top-flagged")

    out_dir = args.out_dir
    if out_dir is None and args.seed is not None:
        out_dir = (args.output_dir / "figures" / f"seed_{args.seed}"
                   / "sample_reports")
    return render_samples(run_dir, sample_ids, out_dir=out_dir, dpi=args.dpi)


if __name__ == "__main__":
    sys.exit(main())
