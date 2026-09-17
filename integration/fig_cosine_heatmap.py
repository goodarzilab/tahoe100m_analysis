"""
Cosine distance heatmap with SNF cluster overlay (lower-left triangle).
Generates both versions: with and without white-line gap separators.

Outputs (this folder):
  cosine_heatmap_with_gaps.pdf/png/svg
  cosine_heatmap_no_gaps.pdf/png/svg
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import scanpy as sc
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from snf_pipeline import (
    run_snf, check_reference_clusters, load_drug_annotations, leiden_cluster,
    PLOT_STYLE, CHOSEN_RES, MU, SEED,
)

plt.rcParams.update(PLOT_STYLE)

# Figures and their data go to results/figures, which is gitignored.
OUT_DIR = os.environ.get("TAHOE_FIGURE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# The SNF + Leiden recipe lives in snf_pipeline.py; see that module for the
# parameters and docs/METHODS_snf_clustering.md for the write-up.
_result = run_snf()
check_reference_clusters(_result)

phenotypes = _result.phenotypes
scaled = _result.scaled
MODALITIES = _result.modalities
CELL_LINES = _result.cell_lines
drug_index = _result.drug_index
N_DRUGS = len(drug_index)
K = _result.K
fused = _result.fused
knn_mat = _result.knn
labels = _result.labels
labels_1idx = _result.labels_1idx
cluster_ids = sorted(np.unique(labels_1idx).tolist())
N_CLUSTERS = len(cluster_ids)
PALETTE = dict(zip(cluster_ids, sns.color_palette("tab10", N_CLUSTERS)))
tahoe_drugs = load_drug_annotations()


# ── Cosine distance ─────────────────────────────────────────────────────────
features = np.hstack([scaled[m].values for m in MODALITIES])
cos_dist = cosine_distances(features)
np.fill_diagonal(cos_dist, 0)

off = cos_dist[~np.eye(N_DRUGS, dtype=bool)]
cos_dist_scaled = (cos_dist - off.min()) / (off.max() - off.min())
cos_dist_scaled = np.clip(cos_dist_scaled, 0, 1)
np.fill_diagonal(cos_dist_scaled, 0)
cos_dist_df = pd.DataFrame(cos_dist_scaled, index=drug_index, columns=drug_index)

# Order drugs: by cluster, hierarchical within
drug_order = []
for k in cluster_ids:
    cd = drug_index[labels_1idx == k]
    if len(cd) > 2:
        sub = cos_dist_df.loc[cd, cd].values
        np.fill_diagonal(sub, 0); sub = np.clip(sub, 0, None)
        Z = linkage(squareform(sub, checks=False), method='average')
        drug_order.extend(cd[leaves_list(Z)].tolist())
    else:
        drug_order.extend(cd.tolist())

ordered_dist = cos_dist_df.loc[drug_order, drug_order]
cluster_for_order = np.array([labels_1idx[list(drug_index).index(d)] for d in drug_order])
cluster_sizes = [(labels_1idx == k).sum() for k in cluster_ids]
n_drugs = len(drug_order)

upper_mask = np.triu(np.ones_like(ordered_dist.values, dtype=bool), k=1)
mat_masked = np.ma.masked_where(upper_mask, ordered_dist.values)


def plot_cosine_heatmap(show_gaps, save_basename):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('mako_r').copy()
    cmap.set_bad(color='white')
    im = ax.imshow(mat_masked, cmap=cmap, vmin=0, vmax=1,
                   aspect='equal', interpolation='nearest')

    if show_gaps:
        boundaries = np.cumsum(cluster_sizes)
        for b in boundaries[:-1]:
            ax.plot([-0.5, b - 0.5], [b - 0.5, b - 0.5],
                    color='white', linewidth=1.8)
            ax.plot([b - 0.5, b - 0.5], [b - 0.5, n_drugs - 0.5],
                    color='white', linewidth=1.8)

    bar_w = n_drugs * 0.018
    for i, k_val in enumerate(cluster_for_order):
        ax.add_patch(mpatches.Rectangle((-bar_w - 2, i - 0.5), bar_w, 1,
                                         color=PALETTE[k_val], clip_on=False))
        ax.add_patch(mpatches.Rectangle((i - 0.5, n_drugs + 1), 1, bar_w,
                                         color=PALETTE[k_val], clip_on=False))

    ax.set_xlim(-0.5, n_drugs - 0.5)
    ax.set_ylim(n_drugs - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Scaled cosine distance', fontsize=10)

    handles = [mpatches.Patch(color=PALETTE[k],
                               label=f'C{k} (n={(labels_1idx == k).sum()})')
               for k in cluster_ids]
    ax.legend(handles=handles, title='SNF cluster',
              bbox_to_anchor=(1.25, 1.0), loc='upper left',
              frameon=False, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, save_basename)
    for ext in ('pdf', 'png', 'svg'):
        plt.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_basename}.{{pdf,png,svg}}')


if __name__ == '__main__':
    plot_cosine_heatmap(show_gaps=True,  save_basename='cosine_heatmap_with_gaps')
    plot_cosine_heatmap(show_gaps=False, save_basename='cosine_heatmap_no_gaps')
