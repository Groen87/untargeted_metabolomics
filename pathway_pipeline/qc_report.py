#!/usr/bin/env python3
"""QC diagnostics for features, metabolites, and pathways.

Reads the pipeline outputs (zscores, flags, mapping tables, reference stats)
plus the original feature matrix and reports three audit layers:

1. FEATURE audit (reference = calibration normals): noise-floor compression
   (a spike of identical values + razor-thin IQR manufactures huge z-scores),
   duplicate features mapped to the same HMDB ID (a dead duplicate corrupts
   the per-metabolite average), and per-feature flag concentration across
   groups (a feature flagging mostly normals/others at high |z| is suspect;
   one flagging mostly IMD patients is doing its job).

2. METABOLITE audit: HMDB IDs backed by multiple dataset features with
   very different scales -- the averaged z is dominated by the noisiest one.

3. PATHWAY audit: pathways whose flags come from a single hot feature
   (concentration), and whose normal flag-rate far exceeds the expected
   percentile (miscalibration).

Usage (one line for PowerShell):
    python -m pathway_pipeline.qc_report --outputs pathway_pipeline/outputs/pathway_pipeline --input pathway_pipeline/data/merged_data_with_classification.csv [--top 20]
"""
import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_COLUMN = "Classification"
OORDEEL_COLUMN = "Oordeel targeted"


def main():
    parser = argparse.ArgumentParser(
        description="Flag problematic features, metabolites, and pathways.")
    parser.add_argument("--outputs", required=True,
                        help="Directory with the pipeline outputs.")
    parser.add_argument("--input", required=True,
                        help="Path to the original feature matrix CSV "
                             "(for the Classification / Oordeel targeted "
                             "columns and the raw feature values).")
    parser.add_argument("--exclude-ids", default=None,
                        help="Comma-separated normal sample IDs excluded "
                             "from the reference (must match "
                             "normal_exclude_ids in the config).")
    parser.add_argument("--spike-fraction", type=float, default=0.5,
                        help="A feature whose most common raw value covers "
                             "at least this fraction of the normals is a "
                             "spike candidate (default 0.5).")
    parser.add_argument("--scale-max", type=float, default=0.08,
                        help="Tight-IQR threshold in log10 units for the "
                             "noise-floor watchlist (default 0.08).")
    parser.add_argument("--top", type=int, default=20,
                        help="Rows to show per table (default 20).")
    args = parser.parse_args()

    out = pd.Series({"outputs": args.outputs})
    out_dir = args.outputs.rstrip("/\\")
    top = args.top

    zscores = pd.read_csv(f"{out_dir}/metabolite_zscores.csv", index_col=0)
    reference_stats = pd.read_csv(f"{out_dir}/reference_stats.csv")
    feature_to_hmdb = pd.read_csv(f"{out_dir}/feature_to_hmdb.csv")
    feature_to_pathway = pd.read_csv(f"{out_dir}/feature_to_pathway.csv")
    metabolite_flags = pd.read_csv(f"{out_dir}/metabolite_flags.csv")
    pathway_flags = pd.read_csv(f"{out_dir}/pathway_flags.csv")
    decisions = pd.read_csv(f"{out_dir}/sample_decisions.csv")

    meta = pd.read_csv(args.input, index_col=0, low_memory=False)
    cls = pd.to_numeric(meta[CLASSIFICATION_COLUMN], errors="coerce")
    oor = pd.to_numeric(meta[OORDEEL_COLUMN], errors="coerce")
    labeled_normal = meta.index[(cls == 0) & (oor == 0)]
    exclude = set()
    if args.exclude_ids:
        exclude = {s.strip() for s in args.exclude_ids.split(",") if s.strip()}
    index_str = pd.Series(meta.index.astype(str).values, index=meta.index)
    exclude_labels = set(index_str[index_str.isin(exclude)].index)
    reference_ids = [s for s in labeled_normal if s not in exclude_labels]
    logger.info(f"Reference: {len(reference_ids)} calibration normals "
                f"({len(exclude_labels)} excluded), "
                f"{len(meta) - len(labeled_normal)} non-normal samples.")

    group = pd.Series("other", index=meta.index)
    group[labeled_normal] = "normal"
    is_imd = (cls == 1) & (oor == 1)
    group[meta.index[is_imd]] = "imd"

    # Raw values for the mapped features (reference rows only).
    matched_features = (feature_to_hmdb.loc[
        feature_to_hmdb["hmdb_id"].notna(), "feature"].unique())
    raw = meta.loc[reference_ids, [f for f in matched_features
                                   if f in meta.columns]]

    # ==================================================================
    # SECTION 1: feature audit (noise-floor compression)
    # ==================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 1: Noise-floor features "
                "(spike + tight IQR in the reference normals)")
    logger.info("=" * 70)
    rows = []
    for f in raw.columns:
        values = raw[f].to_numpy()
        values_ok = values[~np.isnan(values)]
        if len(values_ok) == 0:
            continue
        vals, counts = np.unique(values_ok, return_counts=True)
        spike_fraction = counts.max() / len(values_ok)
        stat = reference_stats[reference_stats["feature"] == f]
        scale = float(stat["scale"].iloc[0]) if len(stat) else np.nan
        median = float(stat["median"].iloc[0]) if len(stat) else np.nan
        rows.append({
            "feature": f, "spike_fraction": spike_fraction,
            "spike_value": vals[counts.argmax()],
            "scale": scale, "median": median,
            "n_distinct": len(vals),
        })
    feat_report = pd.DataFrame(rows)
    spike_list = feat_report[
        (feat_report["spike_fraction"] >= args.spike_fraction)
        & (feat_report["scale"] <= args.scale_max)
    ].sort_values(["spike_fraction", "scale"], ascending=[False, True])
    logger.info(f"{len(spike_list)} of {len(feat_report)} mapped features "
                f"show noise-floor compression "
                f"(spike >= {args.spike_fraction:.0%} of normals "
                f"IQR <= {args.scale_max} log10).")
    if len(spike_list):
        logger.info(f"Top {min(top, len(spike_list))} (worst first):")
        for _, r in spike_list.head(top).iterrows():
            logger.info(
                f"  {r['feature']}: {r['spike_fraction']:.0%} of normals at "
                f"one value (median {r['median']:.2f}, IQR {r['scale']:.3f}, "
                f"{int(r['n_distinct'])} distinct values)")

    # ==================================================================
    # SECTION 2: duplicate features per HMDB ID with diverging scales
    # ==================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: metabolites backed by multiple dataset features")
    logger.info("=" * 70)
    scale_map = dict(zip(reference_stats["feature"], reference_stats["scale"]))
    links = feature_to_hmdb[feature_to_hmdb["hmdb_id"].notna()].copy()
    links["scale"] = links["feature"].map(scale_map)
    dup = (links.groupby("hmdb_id")
           .agg(n_features=("feature", "nunique"),
                scale_min=("scale", "min"),
                scale_max=("scale", "max"),
                features=("feature", lambda s: "; ".join(sorted(s))))
           .reset_index())
    diverging = dup[(dup["n_features"] > 1)
                    & (dup["scale_max"] > 2 * dup["scale_min"])]
    logger.info(f"{len(dup[dup['n_features'] > 1])} HMDB metabolites have "
                f"more than one dataset feature; {len(diverging)} have "
                f"diverging scales (max IQR > 2x min IQR) -- the averaged "
                f"z is dominated by the noisiest feature.")
    if len(diverging):
        logger.info(f"Top {min(top, len(diverging))}:")
        for _, r in diverging.sort_values(
                "scale_max", ascending=False).head(top).iterrows():
            logger.info(f"  {r['hmdb_id']}: IQR {r['scale_min']:.3f} vs "
                        f"{r['scale_max']:.3f} | {r['features']}")

    # ==================================================================
    # SECTION 3: per-feature flag concentration across groups
    # ==================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: features driving metabolite flags, by group")
    logger.info("=" * 70)
    mflags = metabolite_flags[metabolite_flags["flagged"]].copy()
    mflags["group"] = mflags["sample_id"].map(group)
    n_by_group = mflags["group"].value_counts().to_dict()
    n_norm_ref = sum(1 for s in reference_ids)
    logger.info(f"{len(mflags)} flagged (sample, metabolite) pairs: "
                f"{n_by_group.get('imd', 0)} in imd patients, "
                f"{n_by_group.get('normal', 0)} in normals "
                f"(all {n_norm_ref} scored normals, not just the reference), "
                f"{n_by_group.get('other', 0)} in other samples.")
    feature_of = dict(zip(feature_to_hmdb["feature"],
                          zip(feature_to_hmdb["hmdb_id"],
                              feature_to_hmdb["match_method"])))
    mflags["feature"] = mflags["metabolite"]
    per_feature = (mflags.groupby(["metabolite", "group"])["abs_z"].max()
                   .unstack(fill_value=0.0))
    per_feature["n_total"] = mflags.groupby("metabolite")["flagged"].sum()
    per_feature["max_abs_z"] = mflags.groupby("metabolite")["abs_z"].max()
    normal_heavy = per_feature[
        (per_feature.get("normal", 0) >= 2)
        & (per_feature.get("normal", 0)
           >= per_feature.get("imd", 0))].sort_values(
        ["n_total", "max_abs_z"], ascending=False)
    logger.info(f"{len(normal_heavy)} metabolites flag >= 2 normals and "
                f"at least as many normals as IMD patients -- "
                f"artifact-suspicion list.")
    if len(normal_heavy):
        logger.info(f"Top {min(top, len(normal_heavy))}:")
        for name, r in normal_heavy.head(top).iterrows():
            logger.info(
                f"  {name}: {int(r.get('imd', 0))} imd / "
                f"{int(r.get('normal', 0))} normal / "
                f"{int(r.get('other', 0))} other flags "
                f"(max |z| {r['max_abs_z']:.1f})")

    # ==================================================================
    # SECTION 4: pathway audit -- flags driven by single hot features
    # ==================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: pathway flag concentration and calibration")
    logger.info("=" * 70)
    pflags = pathway_flags[pathway_flags["flagged"]].copy()
    pflags["group"] = pflags["sample_id"].map(group)
    # how many of a pathway's metabolites carry the flags
    per_pathway = (pflags.groupby(["smp_id", "pathway_name"])
                   .agg(n_flagged_pairs=("flagged", "sum"),
                        mean_excess=("excess", "mean")).reset_index())
    # features per pathway and their max |z| among flagged samples
    zscore_abs = zscores.abs()
    pw_feature_lists = {}
    for smp, g in feature_to_pathway.groupby("smp_id"):
        pw_feature_lists[smp] = sorted(g["feature"].unique())
    concentrated = []
    for _, r in per_pathway.iterrows():
        feats = pw_feature_lists.get(r["smp_id"], [])
        feats = [f for f in feats if f in zscore_abs.columns]
        if not feats or r["n_flagged_pairs"] == 0:
            continue
        flagged_samples = pflags[
            (pflags["smp_id"] == r["smp_id"])]["sample_id"].unique()
        sub = zscore_abs.loc[flagged_samples, feats]
        if len(sub) == 0:
            continue
        hot = sub.max().nlargest(1)
        share = float(sub[hot.index[0]].max() / sub.max(axis=1).mean()) \
            if len(feats) > 1 else 1.0
        concentrated.append({
            "smp_id": r["smp_id"], "pathway_name": r["pathway_name"],
            "n_flagged_pairs": r["n_flagged_pairs"],
            "mean_excess": r["mean_excess"],
            "n_features": len(feats),
            "hottest_feature": hot.index[0],
            "hottest_z": float(hot.iloc[0]),
        })
    conc_df = pd.DataFrame(concentrated)
    n_scored = pathway_flags["smp_id"].nunique()
    per_pathway["n_normals_flagged"] = [
        int(((pflags["smp_id"] == r["smp_id"])
             & (pflags["group"] == "normal")).sum())
        for _, r in per_pathway.iterrows()]
    logger.info(f"{n_scored} pathways scored; "
                f"{per_pathway['n_flagged_pairs'].sum()} flagged pairs "
                f"across {len(per_pathway)} pathways with >= 1 flag.")
    hot_norm = per_pathway[per_pathway["n_normals_flagged"] >= 3]
    logger.info(f"{len(hot_norm)} pathways flag >= 3 normals "
                f"(possible miscalibration or artifact-driven flags).")
    if len(hot_norm):
        logger.info(f"Top {min(top, len(hot_norm))}:")
        for _, r in (hot_norm.sort_values("n_normals_flagged",
                                          ascending=False).head(top)
                     .iterrows()):
            logger.info(f"  {r['pathway_name']} ({r['smp_id']}): "
                        f"{int(r['n_normals_flagged'])} normals flagged, "
                        f"mean excess {r['mean_excess']:.2f}")
    if len(conc_df):
        single_feat = conc_df[conc_df["n_features"] >= 3].sort_values(
            "n_flagged_pairs", ascending=False)
        logger.info(f"Pathway flags per pathway with the single hottest "
                    f"feature (top {min(top, len(single_feat))} by flagged "
                    f"pairs):")
        for _, r in single_feat.head(top).iterrows():
            logger.info(f"  {r['pathway_name']} ({r['smp_id']}): "
                        f"{int(r['n_flagged_pairs'])} flags over "
                        f"{int(r['n_features'])} features, hottest "
                        f"{r['hottest_feature']} (|z| {r['hottest_z']:.1f})")

    logger.info("")
    logger.info("Interpretation guide: Section 1 lists measurement-suspect "
               "features (spike+tight IQR = z-magnifiers); confirm with raw "
               "distributions and batch position before excluding any. "
               "Section 2 lists metabolites whose averaged z is corrupted "
               "by a diverging duplicate feature. Section 3 lists "
               "metabolites flagging normals as often as patients "
               "(artifact suspicion) -- metabolites flagging mostly IMD "
               "patients are working as intended. Section 4 lists "
               "miscalibrated or single-feature-driven pathways.")


if __name__ == "__main__":
    main()
