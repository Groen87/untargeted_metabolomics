import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from pathlib import Path
from inmoose.pycombat import pycombat_norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

    batch_info = pd.read_csv(merged_batch_path)

    # Standardize to strings
    data.columns = data.columns.astype(str)
    batch_info['sample_id'] = batch_info['sample_id'].astype(str)

    # --- Identify QC samples and track complete features BEFORE gap-filling ---
    # This is CRITICAL: we need to know which features are present in ALL QC samples
    # BEFORE any gap-filling, otherwise we can't distinguish gap-filled from real features
    qc4_samples = [col for col in data.columns if 'QC4' in col]
    blauw_samples = [col for col in data.columns if 'blauw' in col]
    
    # Track which features are present in ALL QC samples (before gap-filling)
    qc_complete_features = {}
    for group_name, group_samples in [("QC4", qc4_samples), ("blauw", blauw_samples)]:
        if group_samples:
            group_data_raw = data[group_samples]
            # Features present in ALL samples of this QC group (no NaN in any sample)
            qc_complete_features[group_name] = group_data_raw.notna().all(axis=1)

    # Replace NaN with half the minimum positive value (metabolomics best practice)
    min_positive = data[data > 0].min().min()
    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
    data = data.fillna(small_value)

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

            # PCA - Plot all samples with batch colors and QC markers
            emb = PCA(n_components=2, random_state=random_state).fit_transform(df.T)

            fig, ax = plt.subplots(figsize=(12, 10))

            # Get batch labels and colors
            all_batch_labels = [batch_dict.get(col, 'Unknown') for col in df.columns]
            unique_batches_all = sorted(set(all_batch_labels))
            palette_all = sns.color_palette('viridis', n_colors=len(unique_batches_all))
            batch_to_color_all = {b: palette_all[i] for i, b in enumerate(unique_batches_all)}
            colors_all = [batch_to_color_all.get(b, 'gray') for b in all_batch_labels]
            
            # Identify QC samples by type for different markers (excluding QC3)
            qc4_samples = [col for col in df.columns if 'QC4' in col]
            blauw_samples = [col for col in df.columns if 'blauw' in col]
            
            # Get indices for all QC samples
            qc4_indices = [list(df.columns).index(col) for col in qc4_samples] if qc4_samples else []
            blauw_indices = [list(df.columns).index(col) for col in blauw_samples] if blauw_samples else []
            all_qc_indices = qc4_indices + blauw_indices
            
            # Scatter plot for all NON-QC samples (batch colored circles)
            non_qc_indices = [i for i in range(len(df.columns)) if i not in all_qc_indices]
            if non_qc_indices:
                ax.scatter(
                    emb[non_qc_indices, 0],
                    emb[non_qc_indices, 1],
                    c=[colors_all[i] for i in non_qc_indices],
                    alpha=0.6,
                    s=40,
                    marker='o',
                    label='Biological Samples'
                )
            
            # Scatter plot for QC samples with distinct markers and BRIGHT colors
            # QC4: bright blue triangles, blauw: bright yellow diamonds
            if qc4_indices:
                ax.scatter(
                    emb[qc4_indices, 0],
                    emb[qc4_indices, 1],
                    c='blue',
                    alpha=1.0,
                    s=80,
                    marker='^',
                    edgecolors='black',
                    linewidth=0.5,
                    label='QC4'
                )
            
            if blauw_indices:
                ax.scatter(
                    emb[blauw_indices, 0],
                    emb[blauw_indices, 1],
                    c='gold',
                    alpha=1.0,
                    s=80,
                    marker='D',
                    edgecolors='black',
                    linewidth=0.5,
                    label='blauw QC'
                )
            
            # Calculate explained variance
            explained_variance = PCA(n_components=2, random_state=random_state).fit(df.T).explained_variance_ratio_
            
            ax.set_title(f"{label} (PCA - All Samples)\nExplained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
            ax.grid(True, alpha=0.3)
            
            # Create legend for batches and QC types
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Patch(facecolor=batch_to_color_all[b], label=f"Batch {b}")
                for b in sorted(unique_batches_all)
            ]
            
            # Add markers for sample types with their actual colors
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                           markeredgecolor='black', markersize=8, alpha=0.6, 
                                           label='Biological', linestyle='None'))
            if qc4_indices:
                legend_elements.append(Line2D([0], [0], marker='^', color='blue', markersize=8,
                                               markeredgecolor='black', markeredgewidth=0.5,
                                               label='QC4', linestyle='None'))
            if blauw_indices:
                legend_elements.append(Line2D([0], [0], marker='D', color='gold', markersize=8,
                                               markeredgecolor='black', markeredgewidth=0.5,
                                               label='blauw QC', linestyle='None'))
            
            ax.legend(handles=legend_elements, title="Batch / QC Type", loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)

            fig.savefig(output_dir / f"{suffix}_pca.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        # --- PCA for QC samples ONLY with filtered features ---
        # Filter to features present in >=90% of QC samples to remove gap-filled features
        qc_samples_for_filter = qc4_samples + blauw_samples
        if len(qc_samples_for_filter) >= 2:  # Need at least 2 QC samples
            qc_data_for_filter = df[qc_samples_for_filter]
            
            # Calculate percentage of QC samples where each feature is present (non-NaN)
            feature_presence = qc_data_for_filter.notna().mean(axis=1)
            
            # Filter: features present in >=90% of QC samples
            high_presence_mask = feature_presence >= 0.90
            filtered_features = high_presence_mask[high_presence_mask].index
            
            if len(filtered_features) > 0:
                df_qc_filtered = df.loc[filtered_features, qc_samples_for_filter]
                
                # Handle NaN values
                if df_qc_filtered.isna().any().any():
                    min_positive = df_qc_filtered[df_qc_filtered > 0].min().min()
                    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                    df_qc_filtered = df_qc_filtered.fillna(small_value)
                
                # Standardize and perform PCA
                scaler_qc = StandardScaler()
                df_qc_scaled = scaler_qc.fit_transform(df_qc_filtered.T)
                
                pca_qc = PCA(n_components=2, random_state=random_state)
                pca_qc_result = pca_qc.fit_transform(df_qc_scaled)
                
                # Create figure for QC-only PCA
                fig_qc, ax_qc = plt.subplots(figsize=(10, 8))
                
                # Plot each QC type with its bright color
                if qc4_samples:
                    qc4_idx = [qc_samples_for_filter.index(s) for s in qc4_samples if s in qc_samples_for_filter]
                    if qc4_idx:
                        ax_qc.scatter(
                            pca_qc_result[qc4_idx, 0],
                            pca_qc_result[qc4_idx, 1],
                            c='blue',
                            alpha=1.0,
                            s=100,
                            marker='^',
                            edgecolors='black',
                            linewidth=0.5,
                            label='QC4'
                        )
                
                if blauw_samples:
                    blauw_idx = [qc_samples_for_filter.index(s) for s in blauw_samples if s in qc_samples_for_filter]
                    if blauw_idx:
                        ax_qc.scatter(
                            pca_qc_result[blauw_idx, 0],
                            pca_qc_result[blauw_idx, 1],
                            c='gold',
                            alpha=1.0,
                            s=100,
                            marker='D',
                            edgecolors='black',
                            linewidth=0.5,
                            label='blauw QC'
                        )
                
                # Calculate explained variance
                explained_variance_qc = pca_qc.explained_variance_ratio_
                
                ax_qc.set_title(f"{label} (QC-only PCA, {len(filtered_features)} features in ≥90% QC)\nExplained Variance: PC1={explained_variance_qc[0]:.2%}, PC2={explained_variance_qc[1]:.2%}")
                ax_qc.set_xlabel(f"PC1 ({explained_variance_qc[0]:.1%})")
                ax_qc.set_ylabel(f"PC2 ({explained_variance_qc[1]:.1%})")
                ax_qc.grid(True, alpha=0.3)
                
                # Create legend
                from matplotlib.lines import Line2D
                qc_legend_elements = []
                if qc4_samples:
                    qc_legend_elements.append(Line2D([0], [0], marker='^', color='blue', markersize=8,
                                                       markeredgecolor='black', markeredgewidth=0.5,
                                                       label='QC4', linestyle='None'))
                if blauw_samples:
                    qc_legend_elements.append(Line2D([0], [0], marker='D', color='gold', markersize=8,
                                                       markeredgecolor='black', markeredgewidth=0.5,
                                                       label='blauw QC', linestyle='None'))
                
                if qc_legend_elements:
                    ax_qc.legend(handles=qc_legend_elements, title="QC Type", loc='best')
                
                fig_qc.savefig(output_dir / f"{suffix}_qc_filtered_pca.png", dpi=300, bbox_inches='tight')
                plt.close(fig_qc)
                
                print(f"  ✓ Saved QC-only PCA (filtered features) to {output_dir / f'{suffix}_qc_filtered_pca.png'}")

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
                "QC4": [col for col in df.columns if "QC4" in col],
                "blauw": [col for col in df.columns if "blauw" in col],
            }

            for group_name, group_samples in qc_groups.items():
                if not group_samples:
                    print(f"No samples found for group: {group_name}")
                    continue

                group_data = df[group_samples]

                # --- Improved RSD calculation with filtering ---
                # Use the pre-computed mask from BEFORE gap-filling
                # This is CRITICAL: prevents gap-filled features from inflating RSD
                if group_name in qc_complete_features:
                    qc_complete_mask = qc_complete_features[group_name]
                    # Align mask with current df (in case df has been filtered)
                    qc_complete_mask = qc_complete_mask.reindex(df.index, fill_value=False)
                    group_data_complete = group_data[qc_complete_mask]
                else:
                    # Fallback: compute from current data (shouldn't happen)
                    qc_complete_mask = group_data.notna().all(axis=1)
                    group_data_complete = group_data[qc_complete_mask]
                
                if len(group_data_complete) == 0:
                    print(f"No features present in ALL {group_name} samples. Skipping RSD calculation.")
                    continue
                
                group_means = group_data_complete.mean(axis=1)
                group_stds = group_data_complete.std(axis=1)
                rsd = (group_stds / group_means) * 100
                
                # For reference, also calculate on all features (including incomplete ones)
                group_means_all = group_data.mean(axis=1)
                group_stds_all = group_data.std(axis=1)
                rsd_all = (group_stds_all / group_means_all) * 100
                
                # Filter 2: Only features with mean intensity above median
                # This removes low-intensity features that have artificially high RSD
                intensity_threshold = group_means.median()
                high_intensity_mask = group_means >= intensity_threshold
                rsd_filtered = rsd[high_intensity_mask]
                
                # Print RSD summaries
                print(f"\n--- {group_name} ({label}) ---")
                print(f"Features present in ALL {group_name} QC samples: {len(group_data_complete)} / {len(group_data)}")
                print(f"\nFEATURES IN ALL QC SAMPLES ({len(group_data_complete)} features):")
                print(f"  Median RSD: {rsd.median():.2f}%")
                print(f"  Mean RSD: {rsd.mean():.2f}%")
                print(f"  Features with RSD > 20%: {(rsd > 20).sum()} ({(rsd > 20).mean() * 100:.1f}%)")
                print(f"  Features with RSD > 15%: {(rsd > 15).sum()} ({(rsd > 15).mean() * 100:.1f}%)")
                
                print(f"\nHIGH INTENSITY + COMPLETE FEATURES ({high_intensity_mask.sum()} features):")
                print(f"  Median RSD: {rsd_filtered.median():.2f}%")
                print(f"  Mean RSD: {rsd_filtered.mean():.2f}%")
                print(f"  Features with RSD > 20%: {(rsd_filtered > 20).sum()} ({(rsd_filtered > 20).mean() * 100:.1f}%)")
                print(f"  Features with RSD > 15%: {(rsd_filtered > 15).sum()} ({(rsd_filtered > 15).mean() * 100:.1f}%)")
                
                # Also print ALL features RSD for comparison (shows the problem with gap-filled features)
                print(f"\nALL FEATURES (including incomplete, {len(rsd_all)} features):")
                print(f"  Median RSD: {rsd_all.median():.2f}%")
                print(f"  Mean RSD: {rsd_all.mean():.2f}%")
                print(f"  Features with RSD > 20%: {(rsd_all > 20).sum()} ({(rsd_all > 20).mean() * 100:.1f}%)")
                print(f"  Features with RSD > 15%: {(rsd_all > 15).sum()} ({(rsd_all > 15).mean() * 100:.1f}%)")

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