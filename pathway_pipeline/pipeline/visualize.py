"""Report-only Seaborn visualizations of existing pipeline outputs.

Two audiences, two figure families:

1. Cohort overview figures (:func:`overview_figures`): how the whole
   cohort scores and which pathways drive the flags -- the paper's
   result figures.
2. Per-sample report figures (:func:`sample_report_figures`): how one
   flagged sample's evidence stacks up against the normal reference
   -- the paper's "how a sample is flagged" walkthrough figure.

Label discipline
----------------
Figures follow the pipeline's evidence-budget protocol. By default the
figures are LABEL-BLIND: samples are colored by flag status, never by
IMD group, so producing them during development cannot leak disease
signal into any choice. Pass ``use_labels=True`` ONLY for the paper's
final, label-aware figures of a frozen version -- that read is covered
by the version's one-shot, same as STEP 10.

Robustness
----------
Every figure is wrapped so a plotting failure can never break the
pipeline run: the exception is logged and the run continues. A missing
input frame or an empty frame produces no figure, not a crash.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper")
FLAG_PALETTE = {True: "#d73027", False: "#4575b4"}
GROUP_PALETTE = {"normal": "#4575b4", "imd": "#d73027", "other": "#878787"}
GROUP_ORDER = ["normal", "imd", "other"]


def _save(fig, out: Path, dpi: int, close: bool = True) -> bool:
    try:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    except Exception as exc:
        logger.warning("Failed to write %s: %s", out, exc)
        if close:
            plt.close(fig)
        return False
    if close:
        plt.close(fig)
    return True


def _canon_bool(s: pd.Series) -> pd.Series:
    """Coerce a flag-like column (bool / 0-1 / string) to real booleans."""
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s == 1
    return s.astype(str).str.strip().str.lower().isin(
        ["true", "flagged", "yes", "y", "1"])


def _flag_status(decisions: pd.DataFrame) -> pd.Series:
    if "flagged" not in decisions.columns:
        return pd.Series(False, index=decisions.index)
    return _canon_bool(decisions["flagged"]).fillna(False)


def _color_key(decisions: pd.DataFrame, use_labels: bool) -> Optional[str]:
    if use_labels and "group" in decisions.columns:
        return "group"
    return "flag_status"


def _hue_decisions(decisions: pd.DataFrame, use_labels: bool) -> pd.DataFrame:
    d = decisions.copy()
    d["flag_status"] = _flag_status(d)
    if use_labels and "group" in d.columns:
        d["group"] = d["group"].fillna("other").astype(str)
    return d


def _top_categories(names: pd.Series, values: pd.Series, top: int) -> List[str]:
    """Category order by mean value, highest first (for ordering plots)."""
    df = pd.DataFrame({"name": names.astype(str),
                       "value": pd.to_numeric(values, errors="coerce")})
    return list(df.groupby("name")["value"].mean()
                .sort_values(ascending=False).head(top).index)


def overview_figures(decisions: pd.DataFrame,
                     pathway_flags: pd.DataFrame,
                     pathway_scores: pd.DataFrame,
                     normal_mask: pd.Series,
                     out_dir: Path,
                     dpi: int = 200,
                     use_labels: bool = False,
                     max_pathways: int = 30) -> Dict[str, bool]:
    """Cohort-level result figures from existing pipeline outputs.

    Args:
        decisions: ``sample_decisions`` (one row per sample; at least
            ``sample_id`` and ``flagged``; ``top_excess`` and
            ``sample_p_value`` when available).
        pathway_flags: one row per (sample, pathway) with
            ``z_stouffer_abs``, ``excess``, ``flagged``, ``pathway_name``.
        pathway_scores: Stouffer scores per (sample, pathway) with
            ``z_stouffer``, ``z_stouffer_abs``, ``n_metabolites_used``.
        normal_mask: boolean Series (sample_id -> is-normal reference);
            used for the reference overlay figures.
        out_dir: directory for the PNG files (created on demand).
        dpi: raster resolution.
        use_labels: color by IMD group instead of flag status. Label-blind
            by default; enabling it is a label-aware read of the frozen
            version's data.
        max_pathways: cap on pathways shown per figure.

    Returns:
        Dict mapping figure file name -> written flag.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, bool] = {}
    if decisions is None or len(decisions) == 0:
        logger.info("Overview figures skipped: no sample decisions.")
        return written

    d = _hue_decisions(decisions, use_labels)
    hue = _color_key(d, use_labels)
    palette = GROUP_PALETTE if hue == "group" else FLAG_PALETTE

    # Fig 1: distribution of the continuous anomaly score by flag status.
    try:
        if "top_excess" in d.columns and d["top_excess"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=d, x="top_excess", hue=hue, element="step",
                         fill=False, stat="density", common_norm=False,
                         palette=palette, ax=ax)
            ax.axvline(1.0, color="0.3", ls="--", lw=1)
            ax.set_xlabel("Maximum pathway excess (score / pathway threshold)")
            ax.set_ylabel("Sample density")
            ax.set_title("Anomaly score by " + hue.replace("_", " "))
            written["fig1_anomaly_score_distribution.png"] = _save(
                fig, out_dir / "fig1_anomaly_score_distribution.png", dpi)
    except Exception as exc:
        logger.warning("fig1 (anomaly score distribution) failed: %s", exc)

    # Fig 2: pathway excess heatmap across the cohort.
    try:
        if pathway_flags is not None and len(pathway_flags) > 0 \
                and {"sample_id", "pathway_name", "excess"}.issubset(
                    pathway_flags.columns):
            sub = pathway_flags[pathway_flags["excess"].notna()]
            top_pw = _top_categories(sub["pathway_name"], sub["excess"],
                                    max_pathways)
            sub = sub[sub["pathway_name"].isin(top_pw)]
            mat = (sub.pivot_table(index="pathway_name", columns="sample_id",
                                   values="excess", aggfunc="mean")
                   .reindex(top_pw))
            mat = mat.loc[:, mat.notna().sum().sort_values(
                ascending=False).index]
            if mat.shape[0] and mat.shape[1]:
                fig, ax = plt.subplots(
                    figsize=(1.5 + 0.055 * mat.shape[1],
                             4 + 0.2 * mat.shape[0]))
                sns.heatmap(mat, cmap="rocket_r", vmin=0, center=1.0,
                            linewidths=0.1, linecolor="0.9",
                            cbar_kws={"label": "Pathway excess"}, ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("Pathway excess per sample "
                             "(>1 exceeds the normal p99)")
                plt.setp(ax.get_xticklabels(), rotation=90, fontsize=4)
                plt.setp(ax.get_yticklabels(), fontsize=5)
                written["fig2_pathway_excess_heatmap.png"] = _save(
                    fig, out_dir / "fig2_pathway_excess_heatmap.png", dpi)
    except Exception as exc:
        logger.warning("fig2 (pathway excess heatmap) failed: %s", exc)

    # Fig 3: which pathways drive the flags.
    try:
        if pathway_flags is not None and len(pathway_flags) > 0 \
                and {"pathway_name", "flagged"}.issubset(pathway_flags.columns):
            fl = pathway_flags[_canon_bool(pathway_flags["flagged"])]
            counts = fl["pathway_name"].value_counts().head(max_pathways)
            if len(counts):
                fig, ax = plt.subplots(
                    figsize=(7, 0.25 * len(counts) + 1.5))
                sns.barplot(x=counts.values, y=counts.index, color="#d73027",
                            ax=ax)
                ax.set_xlabel("Flagged (sample, pathway) pairs")
                ax.set_ylabel("")
                ax.set_title("Pathways driving the flags")
                written["fig3_flagged_pathway_counts.png"] = _save(
                    fig, out_dir / "fig3_flagged_pathway_counts.png", dpi)
    except Exception as exc:
        logger.warning("fig3 (flagged pathway counts) failed: %s", exc)

    # Fig 4: depth vs breadth evidence axes per sample.
    try:
        if {"top_excess", "sample_p_value"}.issubset(d.columns):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(data=d, x="top_excess", y="sample_p_value",
                            hue=hue, palette=palette, s=28, alpha=0.85,
                            ax=ax)
            ax.set_xscale("log")
            ax.axhline(0.05, color="0.3", ls="--", lw=1)
            ax.axvline(1.0, color="0.3", ls="--", lw=1)
            ax.set_xlabel("Maximum pathway excess (log scale)")
            ax.set_ylabel("Depth p-value")
            ax.set_title("Depth vs breadth evidence per sample")
            written["fig4_excess_vs_depth.png"] = _save(
                fig, out_dir / "fig4_excess_vs_depth.png", dpi)
    except Exception as exc:
        logger.warning("fig4 (excess vs depth) failed: %s", exc)

    # Fig 5: per-pathway disturbance across all samples (boxplot), with the
    # normal reference range overlaid -- shows which pathways are disturbed
    # cohort-wide and how wide their normal calibration is.
    try:
        if pathway_scores is not None and len(pathway_scores) > 0 \
                and {"pathway_name", "z_stouffer_abs",
                     "n_metabolites_used"}.issubset(pathway_scores.columns):
            pw = pathway_scores[
                pathway_scores["n_metabolites_used"] >= 3].copy()
            top_pw = _top_categories(pw["pathway_name"],
                                     pw["z_stouffer_abs"], max_pathways)
            pw = pw[pw["pathway_name"].isin(top_pw)]
            if len(pw):
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=pw, x="z_stouffer_abs", y="pathway_name",
                            order=top_pw, color="#a6bddb", fliersize=1, ax=ax)
                ax.set_xlabel("Absolute Stouffer score")
                ax.set_ylabel("")
                ax.set_title("Per-pathway disturbance across the cohort")
                written["fig5_pathway_disturbance_boxplot.png"] = _save(
                    fig, out_dir / "fig5_pathway_disturbance_boxplot.png",
                    dpi)
    except Exception as exc:
        logger.warning("fig5 (pathway disturbance boxplot) failed: %s", exc)

    logger.info("Overview figures: %d written to %s",
                sum(written.values()), out_dir)
    return written


def sample_report_figures(sample_id: str,
                          zscores: pd.DataFrame,
                          pathway_scores: pd.DataFrame,
                          pathway_flags: pd.DataFrame,
                          metabolite_flags: pd.DataFrame,
                          normal_mask: pd.Series,
                          out_dir: Path,
                          dpi: int = 200,
                          max_pathways: int = 25,
                          max_metabolites: int = 40) -> Dict[str, bool]:
    """Per-sample evidence figures: how one sample is flagged.

    Panel A ranks the sample's pathways by excess against the normal
    reference (flagged ones highlighted). Panel B compares the sample's
    absolute Stouffer score distribution to the normal reference
    distribution. Panel C shows the sample's most disturbed metabolite
    z-scores against the reference. Panel D compares signed (direction)
    vs absolute pathway scores -- the direction-aware evidence.

    Args:
        sample_id: the sample to report on.
        zscores: per-sample metabolite z-scores (samples x features).
        pathway_scores: Stouffer scores per (sample, pathway).
        pathway_flags: flag table per (sample, pathway), with ``excess``
            and ``flagged``.
        metabolite_flags: per (sample, metabolite) flags with ``abs_z``,
            ``excess``, ``flagged``.
        normal_mask: boolean Series (sample_id -> is-normal reference).
        out_dir: directory for the PNG files.
        dpi: raster resolution.
        max_pathways: cap on pathways shown.
        max_metabolites: cap on metabolites shown.

    Returns:
        Dict mapping figure file name -> written flag.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, bool] = {}
    if zscores is None or sample_id not in zscores.index:
        logger.info("Sample report for %s skipped: not in the z matrix.",
                    sample_id)
        return written

    # Normal reference ids shared by all panels.
    ref = normal_mask[~normal_mask.index.duplicated(keep="first")] \
        if normal_mask is not None else pd.Series(dtype=bool)
    ref_ids = list(ref[ref].index)

    # Panel A: pathway excess ranking vs the flag threshold.
    try:
        if pathway_flags is not None and len(pathway_flags) > 0:
            sel = (pathway_flags["sample_id"] == sample_id).fillna(False) \
                if "sample_id" in pathway_flags.columns else None
            if sel is not None:
                rows = pathway_flags[sel].dropna(subset=["excess"]) \
                    if "excess" in pathway_flags.columns else \
                    pd.DataFrame()
                if len(rows):
                    rows = rows.sort_values("excess", ascending=False) \
                        .head(max_pathways)
                    rows = rows.assign(
                        pathway=rows["pathway_name"].astype(str),
                        is_flagged=_canon_bool(rows["flagged"])
                        if "flagged" in rows.columns else False)
                    fig, ax = plt.subplots(
                        figsize=(7, 0.25 * len(rows) + 1.5))
                    sns.barplot(data=rows, x="excess", y="pathway",
                                hue="is_flagged", dodge=False,
                                palette=FLAG_PALETTE, ax=ax)
                    ax.axvline(1.0, color="0.3", ls="--", lw=1)
                    ax.set_xlabel("Pathway excess (score / threshold)")
                    ax.set_ylabel("")
                    ax.set_title(f"{sample_id}: pathway excess ranking")
                    ax.get_legend().set_title("flagged")
                    written[f"{sample_id}_panelA_pathway_excess.png"] = _save(
                        fig, out_dir / f"{sample_id}_panelA_pathway_excess.png",
                        dpi)
    except Exception as exc:
        logger.warning("Sample report panel A failed for %s: %s",
                       sample_id, exc)

    # Panel B: sample vs normal reference distribution of absolute Stouffer.
    try:
        if pathway_scores is not None and len(pathway_scores) > 0 \
                and {"sample_id", "z_stouffer_abs"}.issubset(
                    pathway_scores.columns):
            ref_scores = pathway_scores[
                pathway_scores["sample_id"].isin(ref_ids)] \
                if ref_ids else pd.DataFrame()
            samp_scores = pathway_scores[
                pathway_scores["sample_id"] == sample_id]
            if len(samp_scores) and len(ref_scores):
                plot = pd.concat([
                    ref_scores.assign(source="normal reference"),
                    samp_scores.assign(source="this sample"),
                ], ignore_index=True)
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.kdeplot(data=plot, x="z_stouffer_abs", hue="source",
                            common_norm=False, fill=True, alpha=0.4,
                            palette={"normal reference": "#a6bddb",
                                     "this sample": "#d73027"}, ax=ax)
                ax.set_xlabel("Absolute Stouffer score")
                ax.set_ylabel("Density")
                ax.set_title(f"{sample_id}: pathway disturbance "
                             "vs normal reference")
                written[f"{sample_id}_panelB_stouffer_vs_reference.png"] = \
                    _save(fig, out_dir /
                          f"{sample_id}_panelB_stouffer_vs_reference.png",
                          dpi)
    except Exception as exc:
        logger.warning("Sample report panel B failed for %s: %s",
                       sample_id, exc)

    # Panel C: most disturbed metabolites, sample z vs reference z.
    try:
        if metabolite_flags is not None and len(metabolite_flags) > 0 \
                and {"sample_id", "metabolite", "abs_z"}.issubset(
                    metabolite_flags.columns):
            sel = metabolite_flags["sample_id"] == sample_id
            rows = metabolite_flags[sel].dropna(subset=["abs_z"]) \
                .sort_values("abs_z", ascending=False).head(max_metabolites)
            if len(rows):
                ref_abs = metabolite_flags[
                    metabolite_flags["sample_id"].isin(ref_ids)] \
                    if ref_ids else pd.DataFrame()
                ref_stat = (ref_abs.groupby("metabolite")["abs_z"]
                            .median().rename("reference_median_abs_z")
                            if len(ref_abs) else None)
                rows = rows.merge(ref_stat, left_on="metabolite",
                                  right_index=True, how="left") \
                    if ref_stat is not None else rows
                plot = rows.melt(
                    id_vars=["metabolite"],
                    value_vars=["abs_z", "reference_median_abs_z"]
                    if ref_stat is not None else ["abs_z"],
                    var_name="series", value_name="value")
                plot["series"] = plot["series"].map(
                    {"abs_z": "this sample",
                     "reference_median_abs_z": "normal median"})
                fig, ax = plt.subplots(
                    figsize=(7, 0.25 * len(rows) + 1.5))
                sns.barplot(data=plot, x="value", y="metabolite",
                            hue="series", palette={
                                "this sample": "#d73027",
                                "normal median": "#a6bddb"}, ax=ax)
                ax.set_xlabel("Absolute z-score")
                ax.set_ylabel("")
                ax.set_title(f"{sample_id}: top disturbed metabolites "
                             "vs normal reference")
                written[f"{sample_id}_panelC_metabolite_z.png"] = _save(
                    fig, out_dir / f"{sample_id}_panelC_metabolite_z.png",
                    dpi)
    except Exception as exc:
        logger.warning("Sample report panel C failed for %s: %s",
                       sample_id, exc)

    # Panel D: signed vs absolute Stouffer scatter -- direction evidence.
    try:
        if pathway_scores is not None and len(pathway_scores) > 0 \
                and {"sample_id", "z_stouffer", "z_stouffer_abs"}.issubset(
                    pathway_scores.columns):
            samp = pathway_scores[
                pathway_scores["sample_id"] == sample_id].dropna(
                subset=["z_stouffer", "z_stouffer_abs"])
            if len(samp):
                samp = samp.sort_values("z_stouffer_abs",
                                        ascending=False).head(max_pathways)
                flagged_names = set()
                if pathway_flags is not None and len(pathway_flags) > 0:
                    fl = pathway_flags[
                        (pathway_flags["sample_id"] == sample_id)
                        & _canon_bool(pathway_flags["flagged"])]
                    flagged_names = set(fl["pathway_name"].astype(str))
                samp = samp.assign(
                    is_flagged=samp["pathway_name"].astype(str)
                    .isin(flagged_names))
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.scatterplot(data=samp, x="z_stouffer",
                                y="z_stouffer_abs", hue="is_flagged",
                                palette=FLAG_PALETTE, s=30, alpha=0.85,
                                ax=ax)
                ax.axhline(ax.get_ylim()[0], lw=0)
                ax.axvline(0.0, color="0.3", ls="--", lw=1)
                ax.set_xlabel("Signed Stouffer score (direction)")
                ax.set_ylabel("Absolute Stouffer score (magnitude)")
                ax.set_title(f"{sample_id}: pathway direction vs magnitude")
                written[f"{sample_id}_panelD_signed_vs_abs.png"] = _save(
                    fig, out_dir / f"{sample_id}_panelD_signed_vs_abs.png",
                    dpi)
    except Exception as exc:
        logger.warning("Sample report panel D failed for %s: %s",
                       sample_id, exc)

    logger.info("Sample report figures for %s: %d written to %s",
                sample_id, sum(written.values()), out_dir)
    return written


def sample_overview_figure(sample_id: str,
                           zscores: pd.DataFrame,
                           pathway_scores: pd.DataFrame,
                           pathway_flags: pd.DataFrame,
                           promoted_diseases,
                           normal_mask: pd.Series,
                           out_dir: Path,
                           dpi: int = 200,
                           top_disease_panels: int = 15,
                           top_pathways: int = 15,
                           top_metabolites: int = 15) -> bool:
    """One per-sample overview figure with three stacked subpanels.

    Panel A: waterfall of the IEMbase disease panels (from the disease
    table, direction-aware) ranked by the sample's pathway excess.
    Panel B: the same waterfall for the direction-blind PathBank
    pathways. Panel C: the top most aberrant metabolites as SIGNED
    z-scores -- the direction evidence at the metabolite level. The
    flag threshold (excess = 1) and z = 0 lines mark the calibration.

    Args:
        sample_id: the sample to report on.
        zscores: per-sample metabolite z-scores (samples x features).
        pathway_scores: Stouffer scores per (sample, pathway).
        pathway_flags: flag table per (sample, pathway) with
            ``excess`` and ``flagged``.
        promoted_diseases: names of the promoted IEMbase disease
            panels (the pathway channel's panel rows); everything else
            in the pathway frames is a (direction-blind) PathBank
            pathway.
        normal_mask: boolean Series (sample_id -> is-normal reference).
        out_dir: directory for the PNG file.
        dpi: raster resolution.
        top_disease_panels: cap on disease panels shown.
        top_pathways: cap on PathBank pathways shown.
        top_metabolites: cap on metabolites shown.

    Returns:
        True when the figure was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if zscores is None or sample_id not in zscores.index:
        logger.info("Sample overview for %s skipped: not in the z matrix.",
                    sample_id)
        return False

    promoted = set(promoted_diseases or [])

    samp_flags = pathway_flags[
        pathway_flags["sample_id"] == sample_id] \
        if pathway_flags is not None and len(pathway_flags) > 0 \
        else pd.DataFrame()
    samp_scores = pathway_scores[
        pathway_scores["sample_id"] == sample_id] \
        if pathway_scores is not None and len(pathway_scores) > 0 \
        else pd.DataFrame()

    panels = (samp_flags[samp_flags["pathway_name"].isin(promoted)]
              if len(samp_flags) else pd.DataFrame())
    pathbank = (samp_flags[~samp_flags["pathway_name"].isin(promoted)]
                if len(samp_flags) else pd.DataFrame())

    zrow = zscores.loc[sample_id].dropna()
    signed_z = zrow.sort_values(key=np.abs, ascending=False) \
        .head(top_metabolites) if len(zrow) else pd.Series(dtype=float)

    heights = []
    if len(panels):
        heights.append(0.30 * min(len(panels), top_disease_panels) + 0.8)
    else:
        heights.append(0.6)
    if len(pathbank):
        heights.append(0.30 * min(len(pathbank), top_pathways) + 0.8)
    else:
        heights.append(0.6)
    heights.append(0.30 * len(signed_z) + 0.8)

    fig, axes = plt.subplots(
        3, 1, figsize=(9, sum(heights) + 1.2),
        gridspec_kw={"height_ratios": heights})
    fig.suptitle(f"{sample_id}: per-sample overview", y=0.995)

    # Panel A: disease-panel waterfall (direction-aware, from the table).
    ax = axes[0]
    if len(panels):
        rows = (panels.dropna(subset=["excess"])
                .sort_values("excess", ascending=False)
                .head(top_disease_panels))
        flags = _canon_bool(rows["flagged"]) \
            if "flagged" in rows.columns else pd.Series(False, index=rows.index)
        colors = [FLAG_PALETTE.get(bool(f), "#4575b4") for f in flags]
        ax.barh(range(len(rows)), rows["excess"].to_numpy(), color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows["pathway_name"].astype(str), fontsize=6)
        ax.invert_yaxis()
        ax.axvline(1.0, color="0.3", ls="--", lw=1)
    else:
        ax.text(0.5, 0.5, "no disease panels scored for this sample",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_title("IEMbase disease panels (direction-aware) — excess", loc="left", fontsize=9)

    # Panel B: PathBank pathway waterfall (direction-blind).
    ax = axes[1]
    if len(pathbank):
        rows = (pathbank.dropna(subset=["excess"])
                .sort_values("excess", ascending=False)
                .head(top_pathways))
        flags = _canon_bool(rows["flagged"]) \
            if "flagged" in rows.columns else pd.Series(False, index=rows.index)
        colors = [FLAG_PALETTE.get(bool(f), "#4575b4") for f in flags]
        ax.barh(range(len(rows)), rows["excess"].to_numpy(), color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows["pathway_name"].astype(str), fontsize=6)
        ax.invert_yaxis()
        ax.axvline(1.0, color="0.3", ls="--", lw=1)
    else:
        ax.text(0.5, 0.5, "no PathBank pathways scored for this sample",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_title("PathBank pathways (direction-blind) — excess", loc="left", fontsize=9)

    # Panel C: top metabolite SIGNED z-scores (direction evidence).
    ax = axes[2]
    if len(signed_z):
        colors = ["#d73027" if v > 0 else "#4575b4" for v in signed_z]
        ax.barh(range(len(signed_z)), signed_z.to_numpy(), color=colors)
        ax.set_yticks(range(len(signed_z)))
        ax.set_yticklabels(signed_z.index.astype(str), fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0.0, color="0.3", lw=1)
    else:
        ax.text(0.5, 0.5, "no metabolite z-scores for this sample",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_title(f"Top {len(signed_z)} aberrant metabolites — signed z (red = up, blue = down)", loc="left", fontsize=9)

    out = out_dir / f"{sample_id}_overview.png"
    ok = _save(fig, out, dpi)
    if ok:
        logger.info("Sample overview figure written to %s", out)
    return ok


def multisplit_figures(runs: pd.DataFrame,
                       stability: pd.DataFrame,
                       out_dir: Path,
                       dpi: int = 200) -> Dict[str, bool]:
    """Figures for the pre-declared multi-split evaluation outputs.

    Fig 1: per-split validation metrics (sensitivity/specificity/AUC)
    with the aggregate mean -- the split-lottery exhibit. Fig 2: the
    per-sample flag-rate histogram by group -- flag robustness across
    splits (label-aware; only produce for the frozen one-shot write-up).
    Fig 3: dev vs val normal flag rate per split -- the threshold-
    optimism exhibit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, bool] = {}
    if runs is None or len(runs) == 0:
        logger.info("Multi-split figures skipped: no runs table.")
        return written

    metrics = [m for m in ("sensitivity", "specificity", "auc")
               if m in runs.columns]
    if metrics:
        try:
            plot = runs.melt(id_vars=["seed"], value_vars=metrics,
                             var_name="metric", value_name="value")
            plot["metric"] = plot["metric"].str.capitalize()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.stripplot(data=plot, x="metric", y="value",
                          color="0.4", size=5, alpha=0.7, jitter=0.15,
                          ax=ax)
            means = plot.groupby("metric")["value"].mean()
            for i, m in enumerate(sorted(plot["metric"].unique())):
                if m in means and pd.notna(means[m]):
                    ax.hlines(means[m], i - 0.25, i + 0.25,
                              color="#d73027", lw=2)
            ax.set_ylim(0.5, 1.02)
            ax.set_xlabel("")
            ax.set_ylabel("Validation-half metric")
            ax.set_title("Per-split validation metrics "
                         "(red line = mean over pre-declared splits)")
            written["fig1_multisplit_metrics.png"] = _save(
                fig, out_dir / "fig1_multisplit_metrics.png", dpi)
        except Exception as exc:
            logger.warning("Multi-split fig1 failed: %s", exc)

    if {"dev_flag_rate", "val_flag_rate"}.issubset(runs.columns):
        try:
            plot = runs.melt(id_vars=["seed"],
                             value_vars=["dev_flag_rate", "val_flag_rate"],
                             var_name="half", value_name="flag_rate")
            plot["half"] = plot["half"].map(
                {"dev_flag_rate": "development normals",
                 "val_flag_rate": "validation normals"})
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.stripplot(data=plot, x="half", y="flag_rate",
                          hue="half", palette=["#4575b4", "#d73027"],
                          size=5, alpha=0.75, jitter=0.12,
                          legend=False, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Normal flag rate")
            ax.set_title("Normal flag rate per split: "
                         "dev (calibrated) vs val (out-of-sample)")
            written["fig2_multisplit_dev_val_flagrate.png"] = _save(
                fig, out_dir / "fig2_multisplit_dev_val_flagrate.png", dpi)
        except Exception as exc:
            logger.warning("Multi-split fig2 failed: %s", exc)

    if stability is not None and len(stability) > 0 \
            and {"group", "flag_rate"}.issubset(stability.columns):
        try:
            plot = stability.copy()
            plot["group"] = plot["group"].fillna("other").astype(str)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=plot, x="flag_rate", hue="group",
                         hue_order=[g for g in GROUP_ORDER
                                    if g in set(plot["group"])],
                         palette=GROUP_PALETTE, bins=10, multiple="stack",
                         ax=ax)
            ax.set_xlabel("Flag rate across splits (per sample)")
            ax.set_ylabel("Samples")
            ax.set_title("Flag stability across pre-declared splits")
            written["fig3_multisplit_flag_stability.png"] = _save(
                fig, out_dir / "fig3_multisplit_flag_stability.png", dpi)
        except Exception as exc:
            logger.warning("Multi-split fig3 failed: %s", exc)

    logger.info("Multi-split figures: %d written to %s",
                sum(written.values()), out_dir)
    return written
