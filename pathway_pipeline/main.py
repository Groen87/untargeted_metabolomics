#!/usr/bin/env python3
"""Main entry point for the pathway pipeline.

Stages:

1. Load the feature matrix CSV.
2. Build the HMDB name index from the HMDB XML (``hmdb_metabolites.xml``).
3. Match every feature column to one or more HMDB accessions
   (HMDB tag -> exact name -> loose name).
4. Load the PathBank all-metabolites CSV (``pathbank_all_metabolites.csv``),
   keeping only rows for the configured species, and keep pathways with
   sufficient feature coverage (optionally curated by extraction-chemistry
   keywords, ``exclude_pathway_keywords``).
5. Split the cohort into frozen development/validation halves and clean
   the calibration reference with leave-one-out hygiene.
6-8. Z-score metabolites against the development normals, restrict and
   de-duplicate pathways (prune_redundant_pathways), compute per-pathway
   Stouffer scores, flag samples against per-pathway normal percentile
   thresholds, and OR in the literature-curated biomarker attachment
   channel (STEP 8d).
9. Label-blind development QC on the calibration reference.
10. One-shot label-aware evaluation (only with ``run_evaluation``).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from pathway_pipeline.config.config import Config
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathbank_pathways,
    match_features_to_hmdb,
    prefer_tagged_features,
    link_features_to_pathways,
    pathway_coverage,
    filter_pathways_by_keywords,
    ambiguous_feature_report,
)
from pathway_pipeline.pipeline.pathway_stats import (
    CLASSIFICATION_COLUMN,
    OORDEEL_COLUMN,
    classify_samples,
    flag_metabolite_scores,
    summarize_metabolite_flags,
    compute_metabolite_zscores,
    derive_ratio_features,
    filter_pathways_for_scoring,
    prune_redundant_pathways,
    compute_stouffer_scores,
    flag_pathway_scores,
    summarize_sample_flags,
)
from pathway_pipeline.pipeline.calibration import (
    assign_groups,
    leave_one_out_hygiene,
    stratified_split,
)
from pathway_pipeline.pipeline.biomarkers import (
    load_biomarker_attachments,
    resolve_biomarker_attachments,
    flag_biomarker_attachments,
    load_disease_biomarker_table,
    resolve_disease_biomarkers,
    flag_disease_biomarkers,
)
from pathway_pipeline.pipeline.develop import run_development_qc
from pathway_pipeline.pipeline.evaluate import summarize_evaluation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """Add file logging alongside the stream handler."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "pathway_pipeline.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


def _log_section(title: str) -> None:
    logger.info(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_feature_matrix(input_file: str,
                         non_feature_columns,
                         patient_id_column: str = None,
                         age_column: str = None,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the input CSV and split it into (features, metadata, ages).

    Keeps every configured non-feature column in a separate metadata frame
    and, when age_column is set and present, returns a numeric ages Series.
    """
    df = pd.read_csv(input_file,
                      index_col=0 if patient_id_column is None else None,
                      low_memory=False)
    if patient_id_column is not None:
        df = df.set_index(patient_id_column)
    logger.info(f"Loaded {input_file}: {df.shape[0]} samples x {df.shape[1]} columns")

    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.any():
        dup_rows = (np.flatnonzero(dup_mask) + 2).tolist()
        logger.warning(f"Dropping {int(dup_mask.sum())} duplicate sample "
                       f"rows (keeping the first occurrence of each "
                       f"sample ID); CSV rows: {dup_rows}")
        df = df.loc[~dup_mask]

    nf = list(non_feature_columns)
    if age_column and age_column not in nf:
        nf = nf + [age_column]
    metadata = df[[c for c in non_feature_columns if c in df.columns]].copy()
    feature_cols = [c for c in df.columns if c not in nf]
    features = df[feature_cols].copy()

    ages = None
    if age_column and age_column in df.columns:
        ages = pd.to_numeric(df[age_column], errors="coerce")
        ages.index = df.index
        logger.info(f"Age column '{age_column}': {int(ages.notna().sum())} usable ages")

    label_columns = [c for c in (CLASSIFICATION_COLUMN, OORDEEL_COLUMN)
                     if c in df.columns]
    if label_columns:
        unlabeled = df[label_columns].isna().any(axis=1)
        if unlabeled.any():
            dropped_ids = df.index[unlabeled].tolist()
            logger.warning(f"Dropping {len(dropped_ids)} samples with a NaN in "
                           f"{' or '.join(label_columns)}; first 10: "
                           f"{dropped_ids[:10]}")
            features = features.loc[~unlabeled]
            metadata = metadata.loc[~unlabeled]
            if ages is not None:
                ages = ages.loc[~unlabeled]
        logger.info(f"{len(features)} samples remain after the label filter")
    else:
        logger.warning("Input has no Classification/Oordeel targeted columns; "
                      "the label filter is skipped.")

    logger.info(f"{len(feature_cols)} feature columns, {metadata.shape[1]} metadata columns")
    return features, metadata, ages


def run_pipeline(input_file: str,
                  output_dir: str = "outputs/pathway_pipeline",
                  config_path: str = None) -> dict:
    """Run the feature-engineering stage of the pathway pipeline."""
    config = Config(config_path) if config_path else Config()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out)

    _log_section("PATHWAY PIPELINE -- FEATURE ENGINEERING")
    logger.info(f"Input: {input_file}\nOutput: {out}")

    # ------------------------------------------------------------------
    # Stage 1: preprocessing -- feature -> HMDB -> pathway
    # ------------------------------------------------------------------
    _log_section("STEP 1: Load feature matrix")
    features, metadata, ages = load_feature_matrix(
        input_file,
        non_feature_columns=config.get_list(
            "non_feature_columns", ["Oordeel targeted", "Classification"]
        ),
        patient_id_column=config.get("patient_id_column", None),
        age_column=config.get("age_column", None),
    )

    # ------------------------------------------------------------------
    # STEP 1b: derive configured ratio features (log-difference of the
    # log10 areas = log-ratio). Diagnostic ratios (e.g. acylcarnitine
    # C8/C2) isolate an enzyme block from carnitine-status effects that
    # move the individual species together. The spec list is frozen
    # config, declared from textbook chemistry -- never from flag
    # performance. Derived ratios are NOT metabolites: they stay out of
    # HMDB matching and pathway mapping, and join the scored set the way
    # attached biomarkers do (metabolite z-scores and flags).
    # ------------------------------------------------------------------
    _log_section("STEP 1b: Derive ratio features")
    ratio_specs = config.get("ratio_features", None) or []
    ratio_features = []
    if ratio_specs:
        features, ratio_audit = derive_ratio_features(features, ratio_specs)
        ratio_features = sorted(
            ratio_audit.loc[ratio_audit["status"] == "derived", "name"])
        bad = ratio_audit[ratio_audit["status"] != "derived"]
        if len(bad):
            logger.warning(
                f"{len(bad)} ratio feature spec(s) could not be derived: "
                + "; ".join(f"{r['name']} ({r['status']})"
                            for _, r in bad.iterrows()))
        logger.info(
            f"Derived {len(ratio_features)} ratio feature(s): "
            + (", ".join(ratio_features) if ratio_features else "none"))
        out.mkdir(parents=True, exist_ok=True)
        ratio_audit.to_csv(out / "ratio_feature_audit.csv", index=False)
        logger.info(f"Wrote ratio_feature_audit.csv to {out}")
    else:
        logger.info("No ratio features configured.")

    _log_section("STEP 2: Build HMDB name index")
    hmdb_xml = config.get("hmdb_xml_file", "data/hmdb_metabolites.xml")
    min_name_length = int(config.get("min_name_length", 3))
    use_hmdb_cache = bool(config.get("use_hmdb_cache", True))
    name_index = build_name_index(hmdb_xml, min_name_length=min_name_length,
                                  use_cache=use_hmdb_cache)
    if not name_index:
        logger.warning("HMDB name index is empty (XML missing or unreadable). "
                       "Only HMDB-tagged features will be matched.")

    _log_section("STEP 3: Match features to HMDB accessions")
    metabolite_columns = [c for c in features.columns
                          if c not in set(ratio_features)]
    feature_to_hmdb = match_features_to_hmdb(
        feature_columns=metabolite_columns,
        name_index=name_index,
        min_name_length=min_name_length,
        overrides=config.get("feature_hmdb_overrides", None),
    )
    ambiguous = ambiguous_feature_report(
        feature_to_hmdb,
        exclude=config.get_list("demoted_features"))
    if len(ambiguous):
        logger.warning(
            f"{len(ambiguous)} feature name(s) match multiple HMDB accessions "
            "(chemically ambiguous; review for demotion or override): "
            + "; ".join(f"{feat} -> {ids}"
                        for feat, ids in ambiguous.items()))
    # Identity curation (label-blind, chemistry-based): .HMDB-tagged
    # features are standard-confirmed upstream, so within a resolved
    # metabolite they supersede plain-named twins (co-eluting
    # interlopers caught by name matching). Runs before linking so no
    # downstream stage ever sees the superseded twins.
    superseded_features = pd.DataFrame(
        columns=["feature", "hmdb_id", "superseded_by"])
    if bool(config.get("prefer_tagged_features", True)):
        feature_to_hmdb, superseded_features = prefer_tagged_features(
            feature_to_hmdb, feature_columns=metabolite_columns)
        if bool(config.get("save_mapping_outputs", True)) \
                and len(superseded_features):
            superseded_features.to_csv(out / "superseded_features.csv",
                                       index=False)
            logger.info(f"Wrote superseded_features.csv to {out}")

    _log_section("STEP 4: Load PathBank pathways and link features to pathways")
    pathbank_file = config.get("pathbank_file", "data/pathbank_all_metabolites.csv")
    pathbank_species = config.get("pathbank_species", "Homo sapiens")
    pathway_names_file = config.get("pathbank_pathway_names_file",
                                    "data/pathbank_pathways.csv")
    pathways = load_pathbank_pathways(pathbank_file,
                                      species=pathbank_species,
                                      pathway_names_file=pathway_names_file)
    feature_to_pathway = link_features_to_pathways(feature_to_hmdb, pathways)
    min_coverage = float(config.get("min_pathway_coverage", 0.20))
    coverage = pathway_coverage(feature_to_pathway, pathways,
                                min_coverage=min_coverage)
    # Chemistry-based pathway curation (label-blind): drop whole pathway
    # classes the sample preparation cannot recover (e.g. complex lipids
    # under phase extraction). The keyword list lives in the frozen config.
    exclude_keywords = config.get_list("exclude_pathway_keywords")
    coverage = filter_pathways_by_keywords(coverage, exclude_keywords)

    if bool(config.get("save_mapping_outputs", True)):
        feature_to_hmdb.to_csv(out / "feature_to_hmdb.csv", index=False)
        feature_to_pathway.to_csv(out / "feature_to_pathway.csv", index=False)
        coverage.to_csv(out / "pathway_coverage.csv", index=False)
        logger.info(f"Wrote feature_to_hmdb.csv, feature_to_pathway.csv, "
                     f"pathway_coverage.csv to {out}")

    n_matched = int(feature_to_hmdb.loc[feature_to_hmdb["hmdb_id"].notna(), "feature"].nunique())
    logger.info(f"Matched {n_matched} of {len(features.columns)} features to HMDB; "
                f"{coverage.shape[0]} pathways with >= {min_coverage:.0%} of their "
                f"metabolites mapped to dataset features.")

    if not coverage.empty:
        top = coverage.head(10)[["pathway_name", "n_metabolites",
                                 "n_matched_metabolites", "coverage"]]
        logger.info("Top 10 pathways by coverage:")
        for _, r in top.iterrows():
            logger.info(f"  {r['pathway_name']}: "
                        f"{int(r['n_matched_metabolites'])}/{int(r['n_metabolites'])} "
                        f"metabolites ({r['coverage']:.1%})")

    # ------------------------------------------------------------------
    # Stage 2: z-scores against the normal reference
    # ------------------------------------------------------------------
    if not bool(config.get("run_zscores", True)):
        logger.info("run_zscores is false; stopping after the mapping outputs.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
        }

    _log_section("STEP 5: Compute metabolite z-scores (normals as reference)")
    labeled_normal_mask = classify_samples(
        metadata,
        normal_classification=int(config.get("normal_classification", 0)),
        normal_oordeel=int(config.get("normal_oordeel", 0)),
    )
    sample_group = assign_groups(
        metadata,
        normal_mask=labeled_normal_mask,
        normal_classification=int(config.get("normal_classification", 0)),
        normal_oordeel=int(config.get("normal_oordeel", 0)),
        untreated_imd_only=bool(
            config.get("untreated_imd_only", False)),
    )
    group_counts = sample_group.value_counts()
    logger.info(
        "Sample groups: "
        + ", ".join(f"{g}={int(group_counts.get(g, 0))}"
                    for g in ("normal", "imd", "other"))
    )
    if (bool(config.get("untreated_imd_only", False))
            and "Non-treated" in metadata.columns):
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        untreated = pd.to_numeric(metadata["Non-treated"],
                                 errors="coerce").fillna(0)
        n_demoted = int(((cls == 1) & (oor == 1) & (untreated != 1)).sum())
        logger.info(
            f"untreated_imd_only: {n_demoted} treated IMD samples "
            "reported in the 'other' group"
        )

    # ------------------------------------------------------------------
    # Cohort split (development vs validation) -- the split is part of
    # the frozen configuration and must never be re-drawn per experiment.
    # ------------------------------------------------------------------
    split_enabled = bool(config.get("validation_split.enable", True))
    validation_mask = None
    if split_enabled:
        validation_mask = stratified_split(
            sample_group,
            seed=int(config.get("validation_split.seed", 20260923)),
            validation_fraction=float(
                config.get("validation_split.fraction", 0.5)),
        )

    # Calibration uses DEVELOPMENT normals only; validation samples are
    # scored but never calibrate anything.
    dev_mask = pd.Series(True, index=metadata.index)
    if validation_mask is not None:
        dev_mask = ~validation_mask
    normal_mask = labeled_normal_mask & dev_mask

    # Automated reference hygiene: leave-one-out depth against peers
    # replaces hand-picked exclusions. The rule is pre-specified and uses
    # no disease labels.
    hygiene = None
    if bool(config.get("reference_hygiene.enable", True)):
        pathway_features = sorted(
            set(coverage["matched_features"].str.split(";").explode().dropna())
        ) if not coverage.empty else []
        max_excluded_fraction = config.get(
            "reference_hygiene.max_excluded_fraction", None)
        hygiene_mask, hygiene = leave_one_out_hygiene(
            features[pathway_features],
            normal_mask=normal_mask,
            max_depth=float(config.get("reference_hygiene.max_depth", 20.0)),
            max_excluded_fraction=(
                float(max_excluded_fraction)
                if max_excluded_fraction is not None else None),
        )
        normal_mask = hygiene_mask
    if not normal_mask.any():
        logger.error("No normal reference samples; cannot compute z-scores.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
        }

    # Only features mapped to a kept pathway can contribute to pathway scores;
    # keep the whole matrix out of scope here. Attached biomarkers (literature
    # curation) join the z-score set even when no kept PathBank pathway maps
    # them -- the biomarker channel scores them independently of pathway
    # wiring.
    pathway_features = sorted(
        set(coverage["matched_features"].str.split(";").explode().dropna())
    ) if not coverage.empty else []
    biomarker_attachments = None
    biomarker_features = []
    disease_table = None
    disease_resolved = None
    if bool(config.get("biomarker_channel.enable", True)):
        # Source 1: IEMbase-style disease biomarker table (Excel/CSV with
        # biomarker names, direction arrows, and per-metabolite HMDB
        # codes). Diseases are the grouping unit -- the workbook's SMP
        # code links a disease to a kept PathBank pathway for reporting
        # only; a disease without a kept pathway still scores.
        disease_table_file = config.get(
            "biomarker_channel.disease_table_file", None)
        if disease_table_file:
            disease_table = load_disease_biomarker_table(disease_table_file)
            if not disease_table.empty:
                disease_resolved, disease_features, disease_audit = (
                    resolve_disease_biomarkers(
                        disease_table, feature_to_hmdb, coverage)
                )
                if bool(config.get("save_mapping_outputs", True)):
                    disease_audit.to_csv(out / "disease_table_audit.csv",
                                         index=False)
                    logger.info("Wrote disease_table_audit.csv to "
                                f"{out} -- review unmatched/unscored "
                                "rows before freezing the table.")
                biomarker_features.extend(disease_features)
        # Source 2: resolved attachments CSV (smp_id/pathway_name +
        # hmdb_id already known).
        attachments_file = config.get(
            "biomarker_channel.attachments_file",
            "data/pathway_biomarker_attachments.csv")
        biomarker_attachments = load_biomarker_attachments(attachments_file)
        if not biomarker_attachments.empty:
            _, att_features = resolve_biomarker_attachments(
                biomarker_attachments, feature_to_hmdb, coverage)
            biomarker_features.extend(att_features)
    biomarker_features = sorted(set(biomarker_features))
    if biomarker_features:
        extra = sorted(set(biomarker_features) - set(pathway_features))
        if extra:
            logger.info(f"Biomarker channel: {len(set(biomarker_features))} "
                        f"feature(s) carry attached biomarkers "
                        f"({len(extra)} not pathway-mapped).")
        pathway_features = sorted(set(pathway_features)
                                  | set(biomarker_features))
    if ratio_features:
        pathway_features = sorted(set(pathway_features)
                                  | set(ratio_features))
    logger.info(f"Z-scoring {len(pathway_features)} pathway-mapped features "
                f"(of {features.shape[1]} total).")
    features_scored = features[pathway_features]

    zscores, reference_stats, dropped_features = compute_metabolite_zscores(
        features_scored,
        normal_mask=normal_mask,
        iqr_scale=bool(config.get("iqr_scale", True)),
        min_reference_scale=(
            float(config.get("min_reference_scale"))
            if config.get("min_reference_scale", None) is not None else None),
    )

    # Demoted artifact features stay in the z-score output for transparency
    # but never contribute to pathway Stouffer sums, metabolite flags, or
    # the reference calibration of downstream thresholds.
    demoted_features = [f for f in config.get_list("demoted_features")
                        if f in zscores.columns]
    unmatched = [f for f in config.get_list("demoted_features")
                 if f not in zscores.columns]
    if unmatched:
        logger.warning(f"{len(unmatched)} configured demoted feature(s) match "
                       f"no z-scored column (check the exact names): "
                       f"{unmatched}")
    if demoted_features:
        logger.info(f"Demoting {len(demoted_features)} artifact features "
                    f"(excluded from scoring, kept in the reports): "
                    f"{demoted_features}")
        zscores_scored = zscores.drop(columns=demoted_features)
    else:
        zscores_scored = zscores

    if bool(config.get("save_zscore_outputs", True)):
        zscores.to_csv(out / "metabolite_zscores.csv")
        reference_stats.to_csv(out / "reference_stats.csv", index=False)
        dropped_features.to_csv(out / "dropped_features.csv", index=False)
        logger.info(f"Wrote metabolite_zscores.csv, reference_stats.csv, "
                    f"dropped_features.csv to {out}")

    # ------------------------------------------------------------------
    # Stage 2b: restrict pathways to calibrated features
    # ------------------------------------------------------------------
    _log_section("STEP 6: Restrict pathways to calibrated features")
    min_pathway_features = int(config.get("min_pathway_features", 3))
    scored_coverage = filter_pathways_for_scoring(
        coverage,
        feature_to_pathway,
        available_features=zscores_scored.columns,
        min_pathway_features=min_pathway_features,
    )
    logger.info(f"{len(scored_coverage)} pathways remain with >= "
                f"{min_pathway_features} usable matched metabolites.")

    if bool(config.get("prune_redundant_pathways", True)) and not scored_coverage.empty:
        scored_coverage, pruned_pathways = prune_redundant_pathways(
            scored_coverage,
            min_jaccard=float(config.get("redundancy_jaccard", 0.8)),
        )
        if len(pruned_pathways) and bool(config.get("save_zscore_outputs", True)):
            pruned_pathways.to_csv(out / "pruned_pathways.csv", index=False)
            logger.info("Wrote pruned_pathways.csv to " + str(out))

    if bool(config.get("save_zscore_outputs", True)) and not scored_coverage.empty:
        scored_coverage.to_csv(out / "pathway_coverage_scored.csv", index=False)
        logger.info("Wrote pathway_coverage_scored.csv to "
                    f"{out}")

    # ------------------------------------------------------------------
    # Stage 3: pathway Stouffer scores
    # ------------------------------------------------------------------
    if not bool(config.get("run_stouffer", True)):
        logger.info("run_stouffer is false; stopping after the z-score outputs.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
            "normal_mask": normal_mask,
            "zscores": zscores,
            "reference_stats": reference_stats,
            "dropped_features": dropped_features,
            "pathway_coverage_scored": scored_coverage,
        }

    _log_section("STEP 7: Pathway Stouffer scores")
    min_metabolites = int(config.get("min_stouffer_metabolites", 3))
    max_abs_z = config.get("max_abs_z", None)
    max_abs_z = float(max_abs_z) if max_abs_z is not None else None
    # Scale^2 weights: under a constant absolute analytical error the noise
    # variance of a feature's z is (error/scale)^2, so the wider-scale
    # duplicate feature of a metabolite is the trustworthy one.
    feature_scale_weights = None
    if bool(config.get("scale_weighted_metabolites", True)) \
            and not reference_stats.empty:
        feature_scale_weights = {
            row["feature"]: float(row["scale"]) ** 2
            for _, row in reference_stats.iterrows()}
    pathway_scores, pathway_reference = compute_stouffer_scores(
        zscores_scored,
        feature_to_pathway=feature_to_pathway,
        scored_coverage=scored_coverage,
        normal_mask=normal_mask,
        min_metabolites=min_metabolites,
        max_abs_z=max_abs_z,
        feature_scale_weights=feature_scale_weights,
    )

    if bool(config.get("save_stouffer_outputs", True)):
        pathway_scores.to_csv(out / "pathway_stouffer_scores.csv", index=False)
        pathway_reference.to_csv(out / "pathway_stouffer_reference.csv", index=False)
        logger.info(f"Wrote pathway_stouffer_scores.csv, "
                    f"pathway_stouffer_reference.csv to {out}")

    if not pathway_scores.empty:
        normals_in_scores = normal_mask.reindex(
            pathway_scores["sample_id"].unique(), fill_value=False)
        n_normal = int(normals_in_scores.sum())
        logger.info(f"Stouffer scores cover "
                    f"{pathway_scores['sample_id'].nunique()} samples x "
                    f"{pathway_scores['smp_id'].nunique()} pathways "
                    f"({n_normal} normals in the reference).")

    # ------------------------------------------------------------------
    # Stage 4: flag samples against each pathway's own normal range
    # ------------------------------------------------------------------
    if not bool(config.get("run_flagging", True)):
        logger.info("run_flagging is false; stopping after the Stouffer outputs.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
            "normal_mask": normal_mask,
            "zscores": zscores,
            "reference_stats": reference_stats,
            "dropped_features": dropped_features,
            "pathway_coverage_scored": scored_coverage,
            "pathway_scores": pathway_scores,
            "pathway_reference": pathway_reference,
        }

    _log_section("STEP 8: Flag samples per pathway (normal-percentile thresholds)")
    threshold_percentile = float(config.get("flag_threshold_percentile", 99.0))
    pathway_flags = flag_pathway_scores(
        pathway_scores,
        normal_mask=normal_mask,
        threshold_percentile=threshold_percentile,
    )

    min_flagged_pathways = int(config.get("min_flagged_pathways", 1))
    max_sample_p = float(config.get("max_sample_p", 0.05))
    sample_rule = str(config.get("sample_rule", "max_excess"))
    sample_decisions = summarize_sample_flags(
        pathway_flags,
        min_flagged_pathways=min_flagged_pathways,
        per_pathway_flag_rate=1.0 - threshold_percentile / 100.0,
        max_sample_p=max_sample_p,
        normal_mask=normal_mask,
        sample_rule=sample_rule,
    )

    decisions_labeled = sample_decisions.merge(
        sample_group.rename("group"), left_on="sample_id", right_index=True,
        how="left")
    decisions_labeled["group"] = decisions_labeled["group"].fillna("other")
    if validation_mask is not None:
        decisions_labeled["validation"] = (
            decisions_labeled["sample_id"].map(validation_mask))
        decisions_labeled["validation"] = (
            decisions_labeled["validation"].fillna(False).astype(bool))
        if bool(config.get("save_flagging_outputs", True)):
            split_table = pd.DataFrame({
                "sample_id": metadata.index,
                "group": sample_group,
                "validation": validation_mask,
            })
            split_table.to_csv(out / "cohort_split.csv", index=False)
            logger.info(f"Wrote cohort_split.csv to {out}")

    # ------------------------------------------------------------------
    # STEP 8c: metabolite-level depth evidence (report-only, label-blind
    # calibration -- thresholds come from the reference normals only).
    # ------------------------------------------------------------------
    metabolite_summary = None
    if bool(config.get("run_metabolite_flags", True)):
        _log_section("STEP 8c: Metabolite-level flags (report-only)")
        metabolite_flags = flag_metabolite_scores(
            zscores_scored,
            normal_mask=normal_mask,
            threshold_percentile=float(
                config.get("metabolite_flag_percentile", 99.0)),
        )
        metabolite_summary = summarize_metabolite_flags(
            metabolite_flags, normal_mask=normal_mask)
        merged = decisions_labeled.merge(
            metabolite_summary, on="sample_id", how="left")
        for col in ("max_metabolite_z", "n_flagged_metabolites",
                    "metabolite_depth_p", "top_metabolite"):
            if col in merged.columns:
                decisions_labeled[col] = merged[col]
        if bool(config.get("save_flagging_outputs", True)):
            metabolite_flags.to_csv(out / "metabolite_flags.csv", index=False)
            logger.info(f"Wrote metabolite_flags.csv to {out}")

    # ------------------------------------------------------------------
    # STEP 8d: biomarker attachment channel (literature-curated prior
    # knowledge, parallel to the PathBank pathway channel). Attachments
    # are frozen in the configuration before any evaluation read; the
    # channel ORs into the sample decision: flagged when an attached
    # biomarker exceeds its normal-percentile threshold AND the sample's
    # maximum attached-biomarker |z| beats the biomarker-restricted depth
    # null of the reference normals. When the disease table is present,
    # DISEASES are the grouping unit and directions are honored per
    # (disease, biomarker). Label-blind by construction.
    # ------------------------------------------------------------------
    biomarker_flags = None
    ratio_biomarker_specs = config.get(
        "biomarker_channel.ratio_biomarkers", None) or []
    ratio_biomarker_tests = None
    if ratio_biomarker_specs:
        ratio_biomarker_tests = pd.DataFrame(ratio_biomarker_specs)[
            ["disease", "ratio", "direction"]]
        missing = [t for t in ratio_biomarker_tests["ratio"]
                   if t not in set(zscores_scored.columns)]
        if missing:
            logger.warning(
                f"{len(missing)} declared ratio biomarker(s) have no "
                f"z-scored column (not derived at STEP 1b?): "
                f"{sorted(set(missing))}")
    feature_scale_weights = None
    if (bool(config.get("scale_weighted_metabolites", True))
            and not reference_stats.empty):
        feature_scale_weights = {
            row["feature"]: float(row["scale"]) ** 2
            for _, row in reference_stats.iterrows()}
    if (disease_resolved is not None
            and not disease_resolved.empty
            and not zscores_scored.empty):
        _log_section("STEP 8d: Disease biomarker channel (IEMbase table)")
        biomarker_flags, biomarker_summary = flag_disease_biomarkers(
            zscores_scored,
            disease_resolved,
            feature_to_hmdb,
            normal_mask=normal_mask,
            threshold_percentile=float(
                config.get("biomarker_channel.threshold_percentile", 99.0)),
            feature_scale_weights=feature_scale_weights,
            max_sample_p=float(config.get("max_sample_p", 0.05)),
            ratio_tests=ratio_biomarker_tests,
        )
    elif (biomarker_attachments is not None
            and not biomarker_attachments.empty
            and not zscores_scored.empty):
        _log_section("STEP 8d: Biomarker attachment channel "
                     "(literature-curated)")
        resolved_attachments, _ = resolve_biomarker_attachments(
            biomarker_attachments, feature_to_hmdb, coverage)
        if resolved_attachments.empty:
            logger.warning("No biomarker attachment resolved to a kept "
                           "pathway with a matched feature; channel inert.")
        else:
            biomarker_flags, biomarker_summary = flag_biomarker_attachments(
                zscores_scored,
                resolved_attachments,
                feature_to_hmdb,
                normal_mask=normal_mask,
                threshold_percentile=float(
                    config.get("biomarker_channel.threshold_percentile", 99.0)),
                feature_scale_weights=feature_scale_weights,
            )
    if biomarker_flags is not None and not biomarker_flags.empty:
        max_sample_p = float(config.get("max_sample_p", 0.05))
        biomarker_summary["biomarker_flagged"] = (
            (biomarker_summary["n_flagged_biomarkers"].fillna(0) >= 1)
            & (biomarker_summary["biomarker_depth_p"].fillna(1.0)
               <= max_sample_p)
        )
        if bool(config.get("save_flagging_outputs", True)):
            biomarker_flags.to_csv(out / "biomarker_flags.csv",
                                   index=False)
            logger.info(f"Wrote biomarker_flags.csv to {out}")
        decisions_labeled["flagged_pathway_channel"] = decisions_labeled[
            "flagged"]
        merged = decisions_labeled.merge(
            biomarker_summary, on="sample_id", how="left")
        for col in biomarker_summary.columns:
            if col != "sample_id":
                decisions_labeled[col] = merged[col]
        decisions_labeled["flagged"] = (
            decisions_labeled["flagged_pathway_channel"].astype(bool)
            | decisions_labeled["biomarker_flagged"].fillna(False)
            .astype(bool))
        n_bio = int(biomarker_summary["biomarker_flagged"].sum())
        logger.info(f"Biomarker channel: {n_bio} of "
                    f"{len(decisions_labeled)} samples flagged through "
                    f"attached biomarkers; combined decision: "
                    f"{int(decisions_labeled['flagged'].sum())} flagged "
                    f"(pathway channel alone: "
                    f"{int(decisions_labeled['flagged_pathway_channel'].sum())}).")

        # Label-blind calibration readout: flag rates among NORMALS only
        # (evidence budget #1 -- normals stats are free; IMD labels stay
        # untouched until the frozen STEP 10 read) plus per-disease and
        # per-biomarker attribution of the flags.
        normal_ids = set(decisions_labeled.loc[
            decisions_labeled["group"] == "normal", "sample_id"])
        dev_normal = normal_ids & set(decisions_labeled.loc[
            ~decisions_labeled["validation"].fillna(False), "sample_id"])
        val_normal = normal_ids - dev_normal
        for name, ids in (("development", dev_normal),
                          ("validation", val_normal)):
            if ids:
                sub = decisions_labeled[
                    decisions_labeled["sample_id"].isin(ids)]
                logger.info(
                    f"Channel calibration check ({name} normals): "
                    f"{int(sub['biomarker_flagged'].sum())} of "
                    f"{len(sub)} flagged through the biomarker channel "
                    f"({sub['biomarker_flagged'].mean():.1%}); combined "
                    f"decision: {int(sub['flagged'].sum())} of {len(sub)} "
                    f"({sub['flagged'].mean():.1%}).")
        flagged_pairs = biomarker_flags[biomarker_flags["flagged"]]
        if not flagged_pairs.empty:
            if "disease" in flagged_pairs.columns:
                per_disease = (flagged_pairs.groupby("disease")
                               .agg(n_pairs=("flagged", "size"),
                                    n_samples=("sample_id", "nunique"))
                               .sort_values("n_pairs", ascending=False))
                logger.info("Flagged biomarker pairs per disease (top 15):")
                for disease, row in per_disease.head(15).iterrows():
                    logger.info(f"  {disease}: {int(row['n_pairs'])} pairs "
                                f"across {int(row['n_samples'])} samples")
            name_cols = (["hmdb_id", "biomarker"]
                         if "biomarker" in flagged_pairs.columns
                         else ["hmdb_id"])
            hot = (flagged_pairs.groupby(name_cols)
                   .agg(n_pairs=("flagged", "size"),
                        n_samples=("sample_id", "nunique"))
                   .sort_values("n_pairs", ascending=False))
            logger.info("Biomarkers driving the flags (top 10):")
            for key, row in hot.head(10).iterrows():
                logger.info(f"  {key}: {int(row['n_pairs'])} pairs "
                            f"across {int(row['n_samples'])} samples")

    # ------------------------------------------------------------------
    # STEP 9: label-blind development QC. Every check runs on the
    # calibration reference (development normals) or on measurement
    # properties; IMD labels are never read here.
    # ------------------------------------------------------------------
    dev_qc = None
    if bool(config.get("run_development_qc", True)):
        _log_section("STEP 9: Development QC (label-blind)")
        dev_qc = run_development_qc(
            zscores=zscores_scored,
            reference_stats=reference_stats,
            reference_mask=normal_mask,
            features=features,
            feature_to_hmdb=feature_to_hmdb,
            pathway_scores=pathway_scores,
            scored_coverage=scored_coverage,
            percentile=threshold_percentile,
            n_bootstrap=int(config.get("development_qc_bootstrap", 200)),
            output_csv=str(out / "development_qc_noise_floor.csv"),
        )

    # ------------------------------------------------------------------
    # STEP 10: evaluation. Explicit and one-shot: metrics are computed on
    # the halves defined by the frozen split, and -- unless
    # evaluate_dev_half is set -- the pipeline logs ONLY the half
    # requested. Reading these numbers and then changing the config
    # invalidates the validation half.
    # ------------------------------------------------------------------
    evaluation = None
    if bool(config.get("run_evaluation", False)):
        _log_section("STEP 10: Evaluation (label-aware, frozen config)")
        dev_half = None
        val_half = None
        if validation_mask is not None:
            dev_half = ~validation_mask
            val_half = validation_mask
        metrics = {}
        if bool(config.get("evaluate_dev_half", False)) and dev_half is not None:
            metrics["development"] = summarize_evaluation(
                decisions_labeled, pathway_flags, half="development",
                validation_mask=dev_half)
        if val_half is not None:
            metrics["validation"] = summarize_evaluation(
                decisions_labeled, pathway_flags, half="validation",
                validation_mask=val_half)
        elif dev_half is None:
            # No split: evaluate everything (single-cohort mode).
            metrics["all"] = summarize_evaluation(
                decisions_labeled, pathway_flags, half="all")
        evaluation = metrics
        if bool(config.get("save_flagging_outputs", True)):
            frames = []
            for name, m in metrics.items():
                g = m["group_summary"].copy()
                g["half"] = name
                frames.append(g)
            if frames:
                pd.concat(frames).to_csv(
                    out / "evaluation_summary.csv", index=False)
                logger.info(f"Wrote evaluation_summary.csv to {out}")

    if bool(config.get("save_flagging_outputs", True)):
        pathway_flags.to_csv(out / "pathway_flags.csv", index=False)
        decisions_labeled.to_csv(out / "sample_decisions.csv", index=False)
        logger.info(f"Wrote pathway_flags.csv, sample_decisions.csv to {out}")
        if hygiene is not None and not hygiene.empty:
            hygiene.to_csv(out / "reference_hygiene.csv", index=False)
            logger.info(f"Wrote reference_hygiene.csv to {out}")

    return {
        "feature_to_hmdb": feature_to_hmdb,
        "feature_to_pathway": feature_to_pathway,
        "pathway_coverage": coverage,
        "normal_mask": normal_mask,
        "validation_mask": validation_mask,
        "sample_group": sample_group,
        "zscores": zscores,
        "reference_stats": reference_stats,
        "dropped_features": dropped_features,
        "pathway_coverage_scored": scored_coverage,
        "pathway_scores": pathway_scores,
        "pathway_reference": pathway_reference,
        "pathway_flags": pathway_flags,
        "sample_decisions": decisions_labeled,
        "metabolite_summary": metabolite_summary,
        "development_qc": dev_qc,
        "evaluation": evaluation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pathway pipeline feature engineering: feature -> HMDB "
                    "-> PathBank pathway mapping with coverage filtering."
    )
    parser.add_argument("--input", default=None,
                        help="Path to the feature matrix CSV.")
    parser.add_argument("--output", default="outputs/pathway_pipeline",
                        help="Output directory (default: outputs/pathway_pipeline).")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (default: pathway_pipeline/config/config.yaml).")
    args = parser.parse_args()

    if args.input is None:
        config = Config(args.config)
        args.input = config.get("input_file", "data/merged_data_with_classification.csv")

    try:
        run_pipeline(input_file=args.input, output_dir=args.output,
                     config_path=args.config)
        logger.info("\nPathway pipeline feature engineering completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
