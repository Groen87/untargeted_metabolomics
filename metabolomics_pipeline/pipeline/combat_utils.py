import pandas as pd
import numpy as np
import matplotlib
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
    show_plots: bool = False,  # Set to False to avoid interactive display
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run ComBat batch correction with NaN handling (metabolomics: half min positive value).
    Saves all plots to output_dir instead of showing them interactively.
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
    batch_vector = pd.Series(
        [batch_dict[col] for col in data.columns],
        index=data.columns,
        dtype="category"
    )

    # --- Run ComBat ---
    print("Running ComBat...")
    combat_result = pycombat_norm(
        data,
        batch_vector,
        covar_mod=np.zeros((len(batch_vector), 0)),
        ref_batch=None,
        na_cov_action="na.omit"
    )
    corrected_data = pd.DataFrame(combat_result, index=data.index, columns=data.columns)
    corrected_data = corrected_data.clip(lower=0)
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
    if save_plots:
        for label, df in [("Before ComBat", data), ("After ComBat", corrected_data)]:
            # Ensure no NaN values for visualization
            if df.isna().any().any():
                min_positive = df[df > 0].min().min()
                small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                df = df.fillna(small_value)

            suffix = label.lower().replace(" ", "_")

            # UMAP
            emb = umap.UMAP(random_state=random_state, n_jobs=1).fit_transform(df.T)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=batch_vector, palette='viridis', ax=ax)
            ax.set_title(f"{label} (UMAP)")
            fig.savefig(output_dir / f"{suffix}_umap.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # PCA - Plot all samples with batch colors and QC labels
            emb = PCA(n_components=2, random_state=random_state).fit_transform(df.T)

            fig, ax = plt.subplots(figsize=(12, 10))

            # Get batch labels and colors
            all_batch_labels = [batch_dict.get(col, 'Unknown') for col in df.columns]
            unique_batches_all = sorted(set(all_batch_labels))
            palette_all = sns.color_palette('viridis', n_colors=len(unique_batches_all))
            batch_to_color_all = {b: palette_all[i] for i, b in enumerate(unique_batches_all)}
            colors_all = [batch_to_color_all.get(b, 'gray') for b in all_batch_labels]
            
            # Identify QC samples
            qc_samples_all = [col for col in df.columns if 'QC3' in col or 'QC4' in col or 'blauw' in col]
            qc_indices = [list(df.columns).index(col) for col in qc_samples_all] if qc_samples_all else []
            
            # Scatter plot for all NON-QC samples (batch colored)
            non_qc_indices = [i for i in range(len(df.columns)) if i not in qc_indices]
            if non_qc_indices:
                ax.scatter(
                    emb[non_qc_indices, 0],
                    emb[non_qc_indices, 1],
                    c=[colors_all[i] for i in non_qc_indices],
                    alpha=0.6,
                    s=40,
                    label='Biological Samples'
                )
            
            # Scatter plot for QC samples with larger markers and batch colors
            if qc_samples_all:
                qc_emb = emb[qc_indices]
                qc_colors = [colors_all[i] for i in qc_indices]
                
                # Plot QC samples with larger markers
                ax.scatter(
                    qc_emb[:, 0],
                    qc_emb[:, 1],
                    c=qc_colors,
                    alpha=0.9,
                    s=100,
                    edgecolors='black',
                    linewidth=1,
                    label='QC Samples'
                )
                
                # Add QC sample labels for clarity
                for i, sample in enumerate(qc_samples_all):
                    ax.text(
                        qc_emb[i, 0],
                        qc_emb[i, 1],
                        sample,
                        fontsize=9,
                        ha='right',
                        va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
            
            # Calculate explained variance
            explained_variance = PCA(n_components=2, random_state=random_state).fit(df.T).explained_variance_ratio_
            
            ax.set_title(f"{label} (PCA - All Samples)\nExplained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
            ax.grid(True, alpha=0.3)
            
            # Create legend for batches and sample types
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=batch_to_color_all[b], label=f"Batch {b}")
                for b in sorted(unique_batches_all)
            ]
            legend_elements.append(Patch(facecolor='gray', label='Biological Samples', alpha=0.6))
            legend_elements.append(Patch(facecolor='black', label='QC Samples', edgecolor='black', linewidth=1))
            
            ax.legend(handles=legend_elements, title="Batch / Sample Type", loc='best')

            fig.savefig(output_dir / f"{suffix}_pca.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Boxplot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                y=df.T.mean(axis=1),
                hue=batch_vector.astype(str).values,
                x = batch_vector.values,
                palette='viridis',
                ax=ax,
                showfliers=False,
                legend=False,
            )

            # Explicitly set ticks and labels
            unique_batches = sorted(set(batch_vector))
            ax.set_xticks(range(len(unique_batches)))  # Set tick positions
            ax.set_xticklabels([f'Batch {b + 1}' for b in unique_batches])  # Set labels
            ax.set_title(f"{label} (Boxplot)")
            ax.set_xlabel('Batch')
            ax.set_ylabel('Mean Intensity per Sample')
            fig.savefig(output_dir / f"{suffix}_boxplot.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # RSD for QC groups
            qc_groups = {
                "QC3": [col for col in df.columns if "QC3" in col],
                "QC4": [col for col in df.columns if "QC4" in col],
                "blauw": [col for col in df.columns if "blauw" in col],
            }

            for group_name, group_samples in qc_groups.items():
                if not group_samples:
                    print(f"No samples found for group: {group_name}")
                    continue

                group_data = df[group_samples]
                group_means = group_data.mean(axis=1)
                group_stds = group_data.std(axis=1)
                rsd = (group_stds / group_means) * 100

                # --- Improved RSD calculation with filtering ---
                # Filter 1: Only features present in ALL QC samples (no NaN)
                qc_complete_mask = group_data.notna().all(axis=1)
                rsd_complete = rsd[qc_complete_mask]
                
                # Filter 2: Only features with mean intensity above median
                # This removes low-intensity features that have artificially high RSD
                intensity_threshold = group_means[qc_complete_mask].median()
                high_intensity_mask = group_means[qc_complete_mask] >= intensity_threshold
                rsd_filtered = rsd_complete[high_intensity_mask]
                
                # Print both unfiltered and filtered RSD summaries
                print(f"\n--- {group_name} ({label}) ---")
                print(f"ALL FEATURES:")
                print(f"  Median RSD: {rsd.median():.2f}%")
                print(f"  Mean RSD: {rsd.mean():.2f}%")
                print(f"  Features with RSD > 20%: {(rsd > 20).sum()} ({(rsd > 20).mean() * 100:.1f}%)")
                print(f"  Features with RSD > 15%: {(rsd > 15).sum()} ({(rsd > 15).mean() * 100:.1f}%)")
                
                print(f"COMPLETE QC FEATURES ({qc_complete_mask.sum()} features):")
                print(f"  Median RSD: {rsd_complete.median():.2f}%")
                print(f"  Mean RSD: {rsd_complete.mean():.2f}%")
                print(f"  Features with RSD > 20%: {(rsd_complete > 20).sum()} ({(rsd_complete > 20).mean() * 100:.1f}%)")
                print(f"  Features with RSD > 15%: {(rsd_complete > 15).sum()} ({(rsd_complete > 15).mean() * 100:.1f}%)")
                
                print(f"HIGH INTENSITY + COMPLETE FEATURES ({high_intensity_mask.sum()} features):")
                print(f"  Median RSD: {rsd_filtered.median():.2f}%")
                print(f"  Mean RSD: {rsd_filtered.mean():.2f}%")
                print(f"  Features with RSD > 20%: {(rsd_filtered > 20).sum()} ({(rsd_filtered > 20).mean() * 100:.1f}%)")
                print(f"  Features with RSD > 15%: {(rsd_filtered > 15).sum()} ({(rsd_filtered > 15).mean() * 100:.1f}%)")

                # Plot and save RSD distribution
                plt.figure(figsize=(8, 4))
                sns.histplot(rsd_filtered, bins=30, kde=True)
                plt.axvline(15, color='red', linestyle='--', label='RSD = 15%')
                plt.axvline(20, color='orange', linestyle='--', label='RSD = 20%')
                plt.title(f"RSD Distribution for {group_name} ({label})\n(Complete + High Intensity Features)")
                plt.xlabel("RSD (%)")
                plt.ylabel("Number of Features")
                plt.legend()
                plt.grid(True)
                plt.savefig(output_dir / f"{suffix}_{group_name}_rsd_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()

    return corrected_data, metrics