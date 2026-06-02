import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from pathlib import Path
from inmoose.pycombat import pycombat_norm
from sklearn.decomposition import PCA
from typing import Tuple

def run_combat_and_visualize(
    merged_data_path: str,
    merged_batch_path: str,
    output_dir: str = "combat_output",
    random_state: int = 42,
    show_plots: bool = True,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run ComBat batch correction with NaN handling (metabolomics: half min positive value).
    """
    # --- Setup ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and preprocess data ---
    data = pd.read_csv(merged_data_path, index_col=0)  # Features x Samples

    # Replace NaN with half the minimum positive value (metabolomics best practice)
    min_positive = data[data > 0].min().min()
    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
    data = data.fillna(small_value)

    batch_info = pd.read_csv(merged_batch_path)

    # Standardize to strings
    data.columns = data.columns.astype(str)
    batch_info['sample_id'] = batch_info['sample_id'].astype(str)

    # Check for common samples
    common_samples = list(set(data.columns) & set(batch_info['sample_id']))
    if not common_samples:
        raise ValueError(
            f"No common samples!\n"
            f"Data columns: {set(data.columns)}\n"
            f"Batch sample_ids: {set(batch_info['sample_id'])}"
        )

    # Filter to common samples
    data = data[common_samples]
    batch_info = batch_info[batch_info['sample_id'].isin(common_samples)]

    # Create batch vector
    batch_dict = dict(zip(batch_info['sample_id'], batch_info['batch']))
    missing_samples = [col for col in data.columns if col not in batch_dict]
    if missing_samples:
        raise ValueError(f"Missing batch assignments for: {missing_samples}")
    batch_vector = np.array([batch_dict[col] for col in data.columns], dtype=int)
    batch_vector_reordered = np.array([batch_dict[col] for col in data.T.index], dtype=int)

    # --- Run ComBat ---
    print("Running ComBat...")
    combat_result = pycombat_norm(data, batch_vector, covar_mod=np.zeros((len(batch_vector), 0)), ref_batch=None, na_cov_action="na.omit")
    corrected_data = pd.DataFrame(combat_result, index=data.index, columns=data.columns)
    corrected_data.to_csv(output_dir / "combat_corrected_data.csv")
    print(f"✓ Saved to {output_dir}/combat_corrected_data.csv")

    # --- Calculate metrics ---
    def calculate_metrics(df, batch_vec):
        b1, b2 = df.loc[:, batch_vec == 1], df.loc[:, batch_vec == 2]
        m1, m2 = np.median(b1.values), np.median(b2.values)
        return {
            'batch1_median': float(m1),
            'batch2_median': float(m2),
            'median_diff': float(abs(m1 - m2)),
            'median_cv': float(np.std([m1, m2]) / np.mean([m1, m2]) * 100),
        }

    metrics = {
        'pre_correction': calculate_metrics(data, batch_vector),
        'post_correction': calculate_metrics(corrected_data, batch_vector),
    }

    # --- Generate plots ---
    if show_plots or save_plots:
        for label, df in [("Before ComBat", data), ("After ComBat", corrected_data)]:
            # Ensure no NaN values for visualization
            if df.isna().any().any():
                min_positive = df[df > 0].min().min()
                small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                df = df.fillna(small_value)

            suffix = label.lower().replace(" ", "_")

            # UMAP
            emb = umap.UMAP(random_state=random_state).fit_transform(df.T)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=batch_vector, palette='viridis', ax=ax)
            ax.set_title(f"{label} (UMAP)")
            if save_plots: fig.savefig(output_dir / f"{suffix}_umap.png", dpi=300, bbox_inches='tight')
            if show_plots: plt.show()
            plt.close(fig)

            # PCA
            emb = PCA(n_components=2, random_state=random_state).fit_transform(df.T)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=batch_vector, palette='viridis', ax=ax)
            ax.set_title(f"{label} (PCA)")
            if save_plots: fig.savefig(output_dir / f"{suffix}_pca.png", dpi=300, bbox_inches='tight')
            if show_plots: plt.show()
            plt.close(fig)

            # Boxplot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=batch_vector, y=df.T.mean(axis=1), palette='viridis', ax=ax, showfliers=False)
            ax.set_title(f"{label} (Boxplot)")
            ax.set_xlabel('Batch')
            ax.set_ylabel('Mean Intensity per Sample')
            ax.set_xticklabels(['Batch 1', 'Batch 2'])
            if save_plots: fig.savefig(output_dir / f"{suffix}_boxplot.png", dpi=300, bbox_inches='tight')
            if show_plots: plt.show()
            plt.close(fig)

    return corrected_data, metrics