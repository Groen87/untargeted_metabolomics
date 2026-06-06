import os
import re
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import csv
from .injection_order import get_injection_order

# --- Constants ---
INTENSITY_THRESHOLD_DEFAULT = 10000
QC_RSD_THRESHOLD = 20.0  # %
QC_INTENSITY_QUANTILE = 0.25  # 25th percentile
BIO_MISSING_THRESHOLD = 0.2  # 20%
BIO_INTENSITY_QUANTILE = 0.10  # 10th percentile

# Known IMD metabolites
IMD_METABOLITES = ['L-Phenylalanine.HMDB0000159', 'Metronidazole.HMDB0015052', 'Propionylcarnitine.HMDB0000824', 'Lidocaine.HMDB0014426', 'L-Proline.HMDB0000162', 'L-Carnitine.HMDB0000062', 'L-Acetylcarnitine.HMDB0000201', 'Hypoxanthine.HMDB0000157', 'L-Valine.HMDB0000883', 'L-Tryptophan.HMDB0000929', 'Creatinine.HMDB0000562', 'L-Isoleucine.HMDB0000172', 'Ceftriaxone.HMDB0015343', 'Acetaminophen.HMDB0001859', 'L-Isoleucine.HMDB0000172', 'Uric acid.HMDB0000289', 'D-Glucose.HMDB0000122', 'Creatine.HMDB0000064', 'Midazolam.HMDB0014821', 'L-Glutamine.HMDB0000641', 'Aciclovir.HMDB0014925', 'Levetiracetam.HMDB0015333', 'Sulfamethoxazole.HMDB0015150', 'L-Glutamic acid.HMDB0000148', 'Pyroglutamic acid.HMDB0000267', 'Carbamazepine.HMDB0014704', 'N-methyl-4-pyridone-3-carboxamide(4PY).HMDB0004194', 'Octanoylcarnitine.HMDB0000791', 'Acetaminophen.HMDB0001859', 'Ceftriaxone.HMDB0015343', 'Hippuric acid.HMDB0000714', 'Trigonelline(TRIG).HMDB0000875', 'N-a-Acetyl-L-arginine.HMDB0004620', 'L-Alanine.HMDB0000161', 'Chenodeoxycholic acid glycine conjugate.HMDB0000637', 'L-Valine.HMDB0000883', '4-Hydroxyproline.HMDB0000725', 'L-Threonine.HMDB0000167', 'Decanoylcarnitine.HMDB0000651', '2-Octenoylcarnitine.HMDB0013324', 'Pipecolic acid.HMDB0000070', 'Pipecolic acid.HMDB0000070', 'Heptanoylcarnitine.HMDB0013238', 'Hexanoylcarnitine.HMDB0000756', 'Isovalerylcarnitine.HMDB0000688', 'Tiglylcarnitine.HMDB0002366', 'N-methyl-4-pyridone-3-carboxamide(4PY).HMDB0004194', 'Butyrylcarnitine.HMDB0002013', '4-Trimethylammoniobutanoic acid.HMDB0001161', 'Butyrylcarnitine.HMDB0002013', 'Citrulline.HMDB0000904', 'Isovalerylcarnitine.HMDB0000688', 'Cortisol.HMDB0000063', 'Glycocholic acid.HMDB0000138', 'Glutamylphenylalanine.HMDB0029156', 'Sebacic acid.HMDB0000792', 'Kynurenic acid.HMDB0000715', 'Methylmalonylcarnitine.HMDB0013133', 'Ceftriaxone.HMDB0015343', '3-Hydroxyhexanoylcarnitine.HMDB0013131', 'L-Valine.HMDB0000883', 'L-Valine.HMDB0000883', '3-Hydroxybutyrylcarnitine.HMDB0013127', '3-Hydroxybutyrylcarnitine.HMDB0013127', 'L-Valine.HMDB0000883', '1-Methylnicotinamide.HMDB0000699', 'Tetradecenoylcarnitine.HMDB0258884', 'D-Glucose.HMDB0000122', 'Adenosine.HMDB0000050', 'Ceftriaxone.HMDB0015343', '4-Guanidinobutanoic acid.HMDB0003464', 'N-Acetyl-L-phenylalanine.HMDB0000512', 'Xanthine.HMDB0000292', 'Isovalerylcarnitine.HMDB0000688', 'N-alpha-Acetyl-L-lysine.HMDB0000446', '3-Methylhistidine.HMDB0000479', 'Suberic acid.HMDB0000893', 'Methylmalonylcarnitine.HMDB0013133', 'Malonylcarnitine.HMDB0002095', 'Taurine.HMDB0000251', 'Trimethoprim.HMDB0014583', 'Dodecanoylcarnitine.HMDB0002250', '2-Octenoylcarnitine.HMDB0013324', 'L-Arginine.HMDB0000517', '4-Pyridoxic acid.HMDB0000017', 'Guanosine.HMDB0000133', 'Adipic acid.HMDB0000448', 'Pyridoxamine.HMDB0001431', '3-Hydroxyisovalerylcarnitine.HMDB0061189', 'Phenylalanylphenylalanine.HMDB0013302', '3-Methylglutarylcarnitine.HMDB0000552', '3-Hydroxybutyrylcarnitine.HMDB0013127', '3-Methylglutarylcarnitine.HMDB0000552', 'Pyridoxine.HMDB0000239', 'norlidocaine.HMDB0060656', 'Lamotrigine.HMDB0014695', 'Uridine.HMDB0000296', 'Pyridoxal.HMDB0001545', 'Niacinamide.HMDB0001406', 'Inosine.HMDB0000195', '3-hydroxyoctanoyl carnitine.HMDB0061634', 'Deoxyadenosine.HMDB0000101', 'S-Adenosylhomocysteine.HMDB0000939', '3-Hydroxysebacic acid.HMDB0000350', 'AICA-riboside.HMDB0062179', '4-Trimethylammoniobutanoic acid.HMDB0001161', '2-Methylbutyrylglycine.HMDB0000339', 'Alanylproline.HMDB0028695', '3-Methylglutaconic acid.HMDB0000522', 'L-Valine.HMDB0000883', 'Chenodeoxycholic acid glycine conjugate.HMDB0000637', 'N-alpha-Acetyl-L-lysine.HMDB0000446', 'Mevalonolactone.HMDB0006024', '2-oxopropylpiperidine-2-carboxylic acid (2-OPP).HMDB0341532', 'N-Acetyl-L-alanine.HMDB0000766', 'norlidocaine.HMDB0060656', 'gamma-Aminobutyric acid.HMDB0000112', 'Adenine.HMDB0000034', 'N-Acetyl-L-alanine.HMDB0000766', 'norlidocaine.HMDB0060656', 'Isobutyrylglycine.HMDB0000730', 'Methylglutaric acid.HMDB0000752', 'Urocanic acid.HMDB0000301', 'Chenodeoxycholic acid glycine conjugate.HMDB0000637', 'norlidocaine.HMDB0060656', 'norclobazam.HMDB0060970', 'Succinyladenosine.HMDB0000912', 'Cholic acid.HMDB0000619', '2-Octenoylcarnitine.HMDB0013324', 'Glutarylcarnitine.HMDB0013130', '3-Hydroxyanthranilic acid.HMDB0001476', 'N-Lactoylphenylalanine.HMDB0062175', 'Oleoylcarnitine.HMDB0005065', 'Guanidoacetic acid.HMDB0000128', 'Homo-L-arginine.HMDB0000670', '2-Hydroxybutyric acid.HMDB0000008', 'S-(2-Carboxypropyl)cysteine.HMDB0030411', 'L-Lysine.HMDB0000182', 'Tiglylglycine.HMDB0000959', 'Sebacic acid.HMDB0000792', '3-hydroxydecanoyl carnitine.HMDB0061636', 'L-Phenylalanine.HMDB0000159', 'Clobazam.HMDB0014493', 'Dihydrothymine.HMDB0000079', 'L-Glutamic acid.HMDB0000148', 'Nicotinamide N-oxide(NNO).HMDB0002730', 'Hawkinsin.HMDB0002354', 'Sebacic acid.HMDB0000792', 'Hydroxykynurenine.HMDB0000732', 'Fluconazole.HMDB0014342', 'Aminoadipic acid.HMDB0000510', 'Pimelic acid.HMDB0000857', 'Hexanoylglycine.HMDB0000701', '3-Hydroxyisovaleric acid.HMDB0000754', 'Homocitrulline.HMDB0000679', '3-Hydroxyhexanoylcarnitine.HMDB0013131', 'Tetradecanoylcarnitine.HMDB0005066', 'L-Isoleucine.HMDB0000172', 'L-Histidine.HMDB0000177', '3-hydroxydecanoyl carnitine.HMDB0061636', 'L-Glutamine.HMDB0000641', 'Nicotinuric acid(NUA).HMDB0003269', 'Glycocholic acid.HMDB0000138', 'Glycocholic acid.HMDB0000138', 'Isobutyrylglycine.HMDB0000730', 'L-Isoleucine.HMDB0000172', 'N-Acetylmannosamine.HMDB0001129', 'Pipecolic acid.HMDB0000070', 'N-alpha-Acetyl-L-lysine.HMDB0000446', 'Isovalerylcarnitine.HMDB0000688', 'Glutarylcarnitine.HMDB0013130', 'Linoleyl carnitine.HMDB0006469', 'N6N6N6-Trimethyl-L-lysine.HMDB0001325', 'norlidocaine.HMDB0060656', 'Acetoacetic acid.HMDB0000060', 'Palmitoylcarnitine.HMDB0000222', 'L-Serine.HMDB0000187', 'Hydroxyphenyllactic acid.HMDB0000755', '3-Hydroxybutyrylcarnitine.HMDB0013127', 'Formiminoglutamic acid.HMDB0000854', 'Hexanoylglycine.HMDB0000701', '2-Octenoylcarnitine.HMDB0013324', 'gamma-Aminobutyric acid.HMDB0000112', '2-Octenoylcarnitine.HMDB0013324', '6-Oxo-pipecolinic acid.HMDB0061705', 'Amoxicillin.HMDB0015193', 'Homocysteine.HMDB0000742', 'Cortisone.HMDB0002802', 'N-methyl-4-pyridone-3-carboxamide(4PY).HMDB0004194', 'Sebacic acid.HMDB0000792', 'L-Phenylalanine.HMDB0000159', 'L-Isoleucine.HMDB0000172', 'L-Aspartic acid.HMDB0000191', '3-Methyl-2-oxovaleric acid.HMDB0000491', 'Argininosuccinic acid.HMDB0000052', 'N-Acetyl-L-phenylalanine.HMDB0000512', '3-hydroxyoctanoyl carnitine.HMDB0061634', 'Citric acid.HMDB0000094', 'Pregabalin.HMDB0014375', 'L-Asparagine.HMDB0000168', '3-Hydroxyphenylacetic acid.HMDB0000440', '3-hydroxydodecanoyl carnitine.HMDB0061638', 'Glycocholic acid.HMDB0000138', '3-hydroxydodecanoyl carnitine.HMDB0061638', 'Hawkinsin.HMDB0002354', 'O-Palmitoleoylcarnitine.HMDB0240782', 'L-Asparagine.HMDB0000168', 'Dihydrouracil.HMDB0000076', 'Acetylglycine.HMDB0000532', 'alpha-Ketoisovaleric acid.HMDB0000019', 'L-Methionine.HMDB0000696', 'Pyridoxal.HMDB0001545', '3-Hydroxyanthranilic acid.HMDB0001476', 'Ceftriaxone.HMDB0015343', '3-Hydroxyanthranilic acid.HMDB0001476', 'L-Homocystine.HMDB0000676', '4-Pyridoxic acid.HMDB0000017', 'N-Acetyl-L-tyrosine.HMDB0000866', 'Octanoylcarnitine.HMDB0000791', '3-Hydroxysuberic acid.HMDB0000325', '2-oxopropylpiperidine-2-carboxylic acid (2-OPP).HMDB0341532', '4-Hydroxyphenylpyruvic acid.HMDB0000707', 'Phenyllactic acid.HMDB0000779', '3-hydroxydecanoyl carnitine.HMDB0061636', 'Vigabatrin.HMDB0015212', 'Aminoadipic acid.HMDB0000510', '6-Oxo-pipecolinic acid.HMDB0061705', 'Suberylglycine.HMDB0000953', 'Nicotinamide riboside reduced(NRH).HMDB0000855', 'Pregabalin.HMDB0014375', 'Unknown_PDEmarker.UMCG0000001', 'Orotic acid.HMDB0000226', 'Pyridoxal.HMDB0001545', 'Chenodeoxycholic acid.HMDB0000518', 'S-(2-Carboxypropyl)cysteine.HMDB0030411', 'Glycocholic acid.HMDB0000138', 'Thymine.HMDB0000262', 'L-Proline.HMDB0000162', 'Glutarylcarnitine.HMDB0013130', 'Dihydrouracil.HMDB0000076', 'L-Serine.HMDB0000187', '3-hydroxydodecanoyl carnitine.HMDB0061638', 'Argininosuccinic acid anhyride.UMCG0000002', '3-Hydroxyanthranilic acid.HMDB0001476', 'Lidocaine.HMDB0014426', '6-Oxo-pipecolinic acid.HMDB0061705', 'Adenosine.HMDB0000050', 'Saccharopine.HMDB0000279', 'Thymine.HMDB0000262', 'L-Phenylalanine.HMDB0000159', 'N-Acetyl-L-methionine.HMDB0011745', 'Inosine.HMDB0000195', 'norlidocaine.HMDB0060656', 'Vanillactic acid.HMDB0000913', 'Sebacic acid.HMDB0000792', '3-Hydroxyisovaleric acid.HMDB0000754', 'Inosine.HMDB0000195', 'N-alpha-Acetyl-L-citrulline.HMDB0000856', 'Uracil.HMDB0000300', '3-Hydroxyisovalerylcarnitine.HMDB0061189', 'N-Lactoylphenylalanine.HMDB0062175', 'Argininosuccinic acid anhyride.UMCG0000002', 'Valproic acid.HMDB0001877', '3-Methylglutarylcarnitine.HMDB0000552', 'Glutaconic acid.HMDB0000620', 'Hexadecadienoylcarnitine.HMDB0013334', 'S-(2-Carboxypropyl)cysteine.HMDB0030411', 'Levetiracetam.HMDB0015333', '2-Methylbutyrylglycine.HMDB0000339', 'Pipecolic acid.HMDB0000070', '2-oxopropylpiperidine-2-carboxylic acid (2-OPP).HMDB0341532', 'norlidocaine.HMDB0060656', 'Tiglylcarnitine.HMDB0002366', 'L-Threonine.HMDB0000167', '3-Hydroxytetradecanoyl carnitine.HMDB0061640', 'L-Tryptophan.HMDB0000929', '2-oxopropylpiperidine-2-carboxylic acid (2-OPP).HMDB0341532', 'norlidocaine.HMDB0060656', 'L-Alanine.HMDB0000161', '3-Methylglutarylcarnitine.HMDB0000552', 'L-Phenylalanine.HMDB0000159', 'Isobutyrylglycine.HMDB0000730', '2-Octenoylcarnitine.HMDB0013324', 'N-Acetylneuraminic acid.HMDB0000230', 'Oleoylcarnitine.HMDB0005065', 'Omeprazol.HMDB0001913', 'Nicotinamide riboside reduced(NRH).HMDB0000855', 'L-Carnitine.HMDB0000062', 'L-Tryptophan.HMDB0000929', 'Pregabalin.HMDB0014375', '3-hydroxydecanoyl carnitine.HMDB0061636', 'L-Kynurenine.HMDB0000684', '4-Hydroxyproline.HMDB0000725']

def detect_delimiter(file_path: str, sample_size: int = 1024) -> str:
    """Detect the delimiter (comma or tab) of a CSV/TSV file."""
    with open(file_path, 'r') as f:
        sample = f.read(sample_size)
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return '\t' if '\t' in sample else ','

def clean_sample_name(x: str) -> str:
    """
    Clean sample name from a string (path or column name).
    Preserves _1/_2 ONLY for expQC samples (case-insensitive).
    """
    if not isinstance(x, str):
        return str(x).split('.raw')[0].strip()

    filename = x.replace('\\', '/').split('/')[-1]
    base_name = filename.split('.raw')[0].split(' (')[0].strip()

    if 'expqc' in base_name.lower():
        return base_name
    return base_name.replace('_1', '').replace('_2', '').strip()

def filter_by_qc_quality(
    df: pd.DataFrame,
    qc_samples: List[str],
    rsd_threshold: float = QC_RSD_THRESHOLD,
    intensity_quantile: float = QC_INTENSITY_QUANTILE,
    imd_metabolites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter features based on QC sample quality:
    1. Present in all QC samples
    2. RSD ≤ threshold in QC samples
    3. Mean intensity ≥ quantile in QC samples
    IMD metabolites are excluded from filtering.
    """
    if not qc_samples:
        return df

    # Create IMD mask if metabolites are provided
    escaped_metabolites = [re.escape(m) for m in imd_metabolites]
    imd_mask = df['Feature'].str.contains('|'.join(escaped_metabolites), case=False, regex=True)

    # Get QC subset
    qc_df = df[qc_samples]

    # Filter 1: Present in all QC samples (use boolean mask)
    mask_all_qc = qc_df.notna().all(axis=1)
    if imd_metabolites:
        mask_all_qc = mask_all_qc | imd_mask  # IMD features always pass
    df = df[mask_all_qc]

    # Filter 2: RSD ≤ threshold in QC samples (use boolean mask)
    qc_df = df[qc_samples]  # Update qc_df after Filter 1
    qc_rsd = (qc_df.std(axis=1) / qc_df.mean(axis=1) * 100).fillna(100)
    mask_low_rsd = qc_rsd <= rsd_threshold
    if imd_metabolites:
        mask_low_rsd = mask_low_rsd | imd_mask  # IMD features always pass
    df = df[mask_low_rsd]

    # Filter 3: Mean QC intensity ≥ quantile (use boolean mask)
    qc_df = df[qc_samples]  # Update qc_df after Filter 2
    qc_mean = qc_df.mean(axis=1)
    intensity_threshold = qc_mean.quantile(intensity_quantile)
    mask_high_intensity = qc_mean >= intensity_threshold
    if imd_metabolites:
        mask_high_intensity = mask_high_intensity | imd_mask  # IMD features always pass
    df = df.loc[mask_high_intensity]

    return df

def filter_by_biological_quality(
    df: pd.DataFrame,
    bio_samples: List[str],
    missing_threshold: float = BIO_MISSING_THRESHOLD,
    intensity_quantile: float = BIO_INTENSITY_QUANTILE,
    imd_metabolites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter features based on biological sample quality:
    1. Missing values ≤ threshold in biological samples
    2. Mean intensity ≥ quantile in biological samples
    IMD metabolites are excluded from filtering.
    """
    if not bio_samples:
        return df

    # Create IMD mask if metabolites are provided
    imd_mask = False
    if imd_metabolites:
        imd_mask = df['Feature'].str.contains('|'.join(re.escape(m) for m in imd_metabolites), case=False, regex=True)

    # Filter 1: Missing values ≤ threshold
    missing_rate = df[bio_samples].isna().mean(axis=1)
    low_missing_mask = missing_rate <= missing_threshold
    if imd_metabolites:
        low_missing_mask = low_missing_mask | imd_mask  # IMD features always pass
    df = df.loc[low_missing_mask].copy()

    # Filter 2: Mean intensity ≥ quantile
    bio_mean = df[bio_samples].mean(axis=1)
    intensity_threshold = bio_mean.quantile(intensity_quantile)
    high_intensity_mask = bio_mean >= intensity_threshold
    if imd_metabolites:
        high_intensity_mask = high_intensity_mask | imd_mask  # IMD features always pass
    df = df.loc[high_intensity_mask].copy()

    return df

def process_metabolomics_data(
    batch: str,
    mode: str,
    input_file: str,
    metadata_file: str,
    output_dir: str = "output",
    intensity_threshold: int = INTENSITY_THRESHOLD_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process raw metabolomics data from Compound Discoverer output.
    Args:
        input_file: Path to CSV file with peak area data
        metadata_file: Path to Excel file with sample metadata
        output_dir: Directory to save output files
        intensity_threshold: Threshold to filter low-intensity features
    Returns:
        Tuple of (transformed_data, batch_metadata) DataFrames
    """
    start_time = datetime.now()
    print(f"\n[{start_time}] Starting metabolomics data processing")
    print(f"Input file: {input_file}")
    print(f"Metadata file: {metadata_file}\n")

    # --- STEP 0: GET INJECTION ORDER ---
    print("[STEP 0] Getting injection order from metadata...")
    injection_order = get_injection_order(metadata_file)

    # --- STEP 1: LOAD AND FILTER ---
    print("[STEP 1] Loading and filtering data...")
    delimiter = detect_delimiter(input_file)
    df = pd.read_csv(input_file, sep=delimiter, low_memory=False)

    if 'Formula' in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=['Formula'])
        print(f"✓ Removed {initial_rows - len(df)} rows without Formula\n")
    else:
        print("⚠️ Warning: No 'Formula' column found. Skipping formula-based filtering.\n")

    # --- STEP 2: PRECOMPUTE MAPPINGS ---
    print("[STEP 2] Precomputing sample mappings...")
    area_cols = [col for col in df.columns if 'Area:' in col]

    col_to_base = {}
    base_to_cols = {}
    for col in area_cols:
        sample_part = col.split('Area: ')[1].split('.raw')[0].split(' (')[0].strip()
        base_id = clean_sample_name(sample_part)
        col_to_base[col] = base_id
        base_to_cols.setdefault(base_id, []).append(col)

    # Clean injection order
    cleaned_injection_order = [clean_sample_name(s) for s in injection_order]
    ordered_base_ids = list(dict.fromkeys(cleaned_injection_order))
    ordered_base_ids = [base_id for base_id in ordered_base_ids if base_id in base_to_cols]

    missing_in_metadata = set(base_to_cols.keys()) - set(ordered_base_ids)
    if missing_in_metadata:
        print(f"⚠️ Warning: {len(missing_in_metadata)} samples in data not found in metadata: {sorted(missing_in_metadata)}")

    bio_samples = [base_id for base_id in ordered_base_ids
                  if not any(kw in base_id.lower() for kw in ['qc', 'expqc', 'blank', 'mb'])]
    qc_samples = [base_id for base_id in ordered_base_ids
                  if any(kw in base_id.lower() for kw in ['qc', 'expqc', 'blank', 'mb'])]
    print(f"✓ Mapped {len(area_cols)} area columns to {len(base_to_cols)} base samples\n")

    # --- STEP 3: VECTORIZED DATA TRANSFORMATION AND MEDIAN NORMALIZATION ---
    print("[STEP 3] Transforming data and applying median normalization (vectorized)...")
    df['feature'] = df.apply(
        lambda row: f"{row['Name']}_{row['Formula']}"
        if pd.notna(row['Name']) else f"{row['Formula']}",
        axis=1
    )

    # Initialize output DataFrame with a clean index
    transformed_df = pd.DataFrame(columns=['Feature', 'RT [min]'] + ordered_base_ids)
    transformed_df['Feature'] = df['feature'].values
    transformed_df['RT [min]'] = df['RT [min]'].values

   # Apply median normalization to each sample column before merging duplicates
    for base_id, cols in base_to_cols.items():
        if base_id not in ordered_base_ids:
            continue
        sample_data = df[cols].copy().replace('', np.nan).astype(float)

        # ONLY apply intensity threshold to NON-QC samples
        if base_id not in qc_samples:
            sample_data = sample_data.mask(sample_data <= intensity_threshold, np.nan)

        if len(cols) > 1:
            mask_all_valid = sample_data.notna().all(axis=1)
            merged = sample_data.mean(axis=1)
            merged[~mask_all_valid] = np.nan
            transformed_df[base_id] = merged
        else:
            transformed_df[base_id] = sample_data.iloc[:, 0]

    print(f"✓ Transformed {len(transformed_df)} features\n")

    # --- STEP 4: QUALITY CONTROL FILTERING ---
    print("[STEP 4] Applying quality control filters...")

    # Identify QC samples
    qc_samples = [base_id for base_id in ordered_base_ids
                  if any(kw in base_id.lower() for kw in ['qc', 'expqc', 'blank', 'mb'])]

    # Filter by QC quality (exclude IMD features from filtering)
    initial_qc_features = len(transformed_df)
    transformed_df = filter_by_qc_quality(
        transformed_df,
        qc_samples,
        imd_metabolites=IMD_METABOLITES,  # Pass IMD_METABOLITES to exclude from filtering
    )
    qc_filtered = initial_qc_features - len(transformed_df)
    print(f"✓ Removed {qc_filtered} features by QC quality filters")

    # Filter by biological quality (exclude IMD features from filtering)
    initial_bio_features = len(transformed_df)
    transformed_df = filter_by_biological_quality(
        transformed_df,
        bio_samples,
        imd_metabolites=IMD_METABOLITES,  # Pass IMD_METABOLITES to exclude from filtering
    )
    bio_filtered = initial_bio_features - len(transformed_df)
    print(f"✓ Removed {bio_filtered} features by biological quality filters")

    print(f"✓ Final feature count: {len(transformed_df)} (started with {len(df)})\n")

    # --- STEP 5: GLOBAL FILTERING ---
    print("[STEP 5] Applying global intensity filter...")
    if bio_samples:
        features_before = len(transformed_df)
        below_or_nan = (transformed_df[bio_samples] <= intensity_threshold) | transformed_df[bio_samples].isna()
        keep_features = ~below_or_nan.all(axis=1)
        transformed_df = transformed_df.loc[keep_features]
        features_removed = features_before - len(transformed_df)
        print(f"✓ Removed {features_removed} features (all samples ≤ threshold)\n")
    else:
        print("⚠️ Warning: No biological samples found for global filtering\n")

    # --- NEW STEP: FILTER COLUMNS TO KEEP ONLY 'Feature' AND 'posneg*' ---
    print("[NEW STEP] Filtering columns to keep only 'Feature' and 'posneg*'...")
    cols_to_keep = ['Feature'] + [col for col in transformed_df.columns if col.startswith('posneg')]
    if not cols_to_keep:
        raise ValueError("No columns starting with 'posneg' found. Check your sample names.")
    transformed_df = transformed_df[cols_to_keep]
    print(f"✓ Kept {len(cols_to_keep)} columns: {cols_to_keep}\n")

    # --- STEP 6: CREATE BATCH DATA ---
    print("[STEP 6] Creating batch metadata...")
    batchdata_df = pd.DataFrame([
        {
            'sample_id': base_id,
            'batch': 1,
            'group': 1 if ('qc3' in base_id.lower() and 'expqc_3' not in base_id.lower()) else
                     2 if ('qc4' in base_id.lower() and 'expqc_4' not in base_id.lower()) else 0,
            'benchmark': 1 if 'blauw' in base_id.lower() else 0
        }
        for base_id in [col for col in cols_to_keep if col != 'Feature']  # Only use filtered columns
    ])
    print(f"✓ Created batch data for {len(batchdata_df)} samples\n")

    # --- STEP 7: SAVE OUTPUT ---
    print("[STEP 7] Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)
    transformed_df.to_csv(f"{output_dir}/{batch}_{mode}_transformed.csv")
    batchdata_df.to_csv(f"{output_dir}/{batch}_{mode}_batch_data.csv")
    batchdata_df.to_csv(f"{output_dir}/batch_data.csv")
    central_dir = "/Users/j.groen/PycharmProjects/untargeted_pipeline/metabolomics_pipeline/data/pqn_normalized_batches"
    os.makedirs(central_dir, exist_ok=True)  # Ensure the directory exists
    batchdata_df.to_csv(f"{central_dir}/{batch}_{mode}_batch_data.csv")

    print(f"✓ Saved to {output_dir}\n")
    print(f"[SUMMARY] Done in {(datetime.now() - start_time).total_seconds():.1f} seconds")

    return transformed_df, batchdata_df