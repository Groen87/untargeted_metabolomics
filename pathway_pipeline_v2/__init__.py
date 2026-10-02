"""Pathway pipeline for untargeted metabolomics, built on PathBank.

Feature-engineering stage (per-pathway statistics to be added later):

1. Matches every feature column to an HMDB accession using the HMDB XML
   metabolite database (``hmdb_metabolites.xml`` name/synonym index),
   producing a feature -> HMDB map (HMDB tag -> exact name -> loose name).
2. Loads the PathBank all-metabolites CSV
   (``pathbank_all_metabolites.csv``), keeping only Metabolic and Disease
   pathways for Homo sapiens, and links the matched HMDB IDs to those
   pathways.
3. Keeps only pathways where at least ``min_pathway_coverage`` (default 20%)
   of the pathway's metabolites are mapped to features in the dataset.
"""

__version__ = "1.0.0"
