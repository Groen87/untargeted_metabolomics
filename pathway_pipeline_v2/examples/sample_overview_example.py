"""Render an example per-sample overview figure with synthetic IEM data.

Demonstrates :func:`~pathway_pipeline_v2.pipeline.visualize.sample_overview_figure`
without a pipeline run: builds one synthetic patient with an MCADD-like
biochemical pattern (medium-chain acylcarnitines up, acetylcarnitine down),
a matching direction-aware disease panel, and background PathBank pathways.
The output PNG is the per-sample walkthrough figure of the paper: the
disease-panel waterfall, the PathBank waterfall, and the top signed
metabolite z-scores.

Usage:
    python -m pathway_pipeline_v2.examples.sample_overview_example \
        --output-dir outputs/sample_overview_example
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pathway_pipeline_v2.pipeline.visualize import sample_overview_figure

SAMPLE_ID = "patient_007"

PANEL_NAMES = [
    "Medium-chain acyl-CoA dehydrogenase deficiency",
    "Very long-chain acyl-CoA dehydrogenase deficiency",
    "Carnitine transporter deficiency (CTD)",
    "Glutaric aciduria type 1",
    "Propionic acidemia",
    "Isovaleric acidemia",
    "Methylmalonic acidemia",
    "Carnitine-acylcarnitine translocase deficiency",
    "Short-chain acyl-CoA dehydrogenase deficiency",
    "Mitochondrial acetoacetyl-CoA thiolase deficiency",
    "Phenylketonuria",
    "Maple syrup urine disease",
    "Ornithine transcarbamylase deficiency",
    "Tyrosinemia type 1",
    "Homocystinuria",
]

PATHBANK_NAMES = [
    "Fatty Acid Biosynthesis",
    "Fatty Acid Metabolism",
    "Mitochondrial Beta-Oxidation",
    "Carnitine Shuttle",
    "Valine, Leucine and Isoleucine Metabolism",
    "Lysine Degradation",
    "Tryptophan Metabolism",
    "Ketone Body Metabolism",
    "Glycine and Serine Metabolism",
    "Arginine and Proline Metabolism",
    "Aspirin Pathway",
    "Glycolysis and Gluconeogenesis",
    "Purine Metabolism",
    "Caffeine Metabolism",
    "Urea Cycle",
]

# MCADD-like signed z pattern: chain-short acylcarnitines up, C2 down,
# dicarboxylic acids up (the usual secondary pattern).
METAB_Z = {
    "Octanoylcarnitine (C8)": 12.4,
    "cis-5-Decenoylcarnitine (C10:1)": 5.8,
    "Decanoylcarnitine (C10)": 4.9,
    "Hexanoylcarnitine (C6)": 4.1,
    "L-Acetylcarnitine (C2)": -3.2,
    "Suberic acid": 3.4,
    "Adipic acid": 2.9,
    "Octenedioic acid": 2.6,
    "L-Carnitine (C0)": -1.8,
    "Glutaric acid": 1.9,
    "Sebacic acid": 1.7,
    "L-Anthranilic acid": 1.4,
    "Creatinine": -1.2,
    "L-Alanine": 0.9,
    "Pyruvic acid": 0.7,
    "L-Lactic acid": 0.5,
    "Adenosine triphosphate": -0.4,
    "L-Phenylalanine": 0.3,
    "Urea": -0.2,
}


def _flags(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    excess_panels = np.concatenate([
        rng.uniform(1.6, 2.6, 1),
        rng.uniform(0.95, 1.25, 2),
        rng.uniform(0.2, 0.8, len(PANEL_NAMES) - 3)])
    for name, ex in zip(PANEL_NAMES, excess_panels):
        rows.append({
            "sample_id": SAMPLE_ID, "smp_id": name, "pathway_name": name,
            "n_metabolites_used": 4,
            "z_stouffer": 1.2 * ex, "z_stouffer_abs": 1.2 * ex,
            "threshold": 1.2, "excess": ex, "flagged": bool(ex > 1.0)})
    excess_pb = np.concatenate([
        rng.uniform(1.4, 2.4, 3),
        rng.uniform(0.3, 0.95, len(PATHBANK_NAMES) - 3)])
    rng.shuffle(excess_pb)
    for name, ex in zip(PATHBANK_NAMES, excess_pb):
        rows.append({
            "sample_id": SAMPLE_ID, "smp_id": name, "pathway_name": name,
            "n_metabolites_used": 6,
            "z_stouffer": 1.2 * ex, "z_stouffer_abs": 1.2 * ex,
            "threshold": 1.2, "excess": ex, "flagged": bool(ex > 1.0)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Render the example per-sample overview figure.")
    parser.add_argument("--output-dir",
                        default="outputs/sample_overview_example",
                        help="Directory for the PNG (default: "
                             "outputs/sample_overview_example).")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    zscores = pd.DataFrame([METAB_Z], index=[SAMPLE_ID])
    pathway_flags = _flags(rng)
    pathway_scores = pathway_flags[
        ["sample_id", "smp_id", "pathway_name", "n_metabolites_used",
         "z_stouffer", "z_stouffer_abs"]]
    normal_mask = pd.Series([True], index=["reference_sample"])

    out_dir = Path(args.output_dir)
    ok = sample_overview_figure(
        sample_id=SAMPLE_ID,
        zscores=zscores,
        pathway_scores=pathway_scores,
        pathway_flags=pathway_flags,
        promoted_diseases=PANEL_NAMES,
        normal_mask=normal_mask,
        out_dir=out_dir,
        dpi=args.dpi)
    if ok:
        print(f"Wrote {out_dir / (SAMPLE_ID + '_overview.png')}")
    else:
        raise SystemExit("Figure was not written.")


if __name__ == "__main__":
    main()
