"""Pathway-shift pipeline for untargeted metabolomics.

A pathway-centric alternative to the outlier-detection pipeline. Instead of
sparse PCA + per-sample outlier scoring, this pipeline:

1. Matches every feature column to an HMDB accession using the HMDB XML
   metabolite database (name/synonym index), producing a feature -> HMDB map.
2. Links those HMDB IDs to metabolic pathways via a pathways TSV
   (smp_id, pathway_name, n_compounds, hmdb_ids), producing feature -> pathway
   links and a pathway-coverage table.
3. (Optional, second stage) Computes direction-aware per-pathway shift
   statistics for each sample:
       Z_med(P)  = median(z_1 ... z_k)            (direction-aware median)
       F(P)      = (1/k) * sum 1[|z_i| > t_i]      (flagged-fraction breadth)
       Z_up(P)   = median of positive z's
       Z_down(P) = median of negative z's         (signed-extreme guard)
   where z_i are per-metabolite IQR-scaled z-scores and t_i are per-metabolite
   empirical (99th percentile of normals) thresholds.
"""
__version__ = "0.1.0"
