"""
Cohen's d heatmaps (cluster vs rest) per modality × cell line, with SQUARE cells.

For each cluster k:
  - Compute Cohen's d (cluster vs rest) for every (modality, cell_line) pair on RAW values.
  - Compute permutation p-values (1000 shuffled cluster labels).
  - Plot as heatmap with cells = squares (aspect='equal').

Outputs (this folder):
  cohens_d_all_clusters.pdf/png/svg          — single multi-panel figure
  cohens_d_cluster_{k}.pdf/png/svg           — per-cluster individual heatmaps
  cohens_d_values.csv                         — long-form table
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
from tqdm import tqdm

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

raw_drug_level = _result.raw

N_LINES = len(CELL_LINES)
N_PERM = 1000



# ── Cohen's d ───────────────────────────────────────────────────────────────
MOD_DISPLAY = {
    'sensitivity':        'Sensitivity',
    'G1_arrest_logodds':  'G1 arrest',
    'G2M_arrest_logodds': 'G2/M arrest',
    'gini':               'Gini',
    'augur':              'Augur',
    'heterogeneity':      'Heterogeneity',
}
PHENO_MODS = [m for m in MODALITIES if m in MOD_DISPLAY]
N_MODS = len(PHENO_MODS)


def cohens_d_vec(in_vals, out_vals):
    ni, no = len(in_vals), len(out_vals)
    if ni < 2 or no < 2:
        return 0.0
    si = np.std(in_vals, ddof=1); so = np.std(out_vals, ddof=1)
    sp = np.sqrt(((ni - 1) * si**2 + (no - 1) * so**2) / (ni + no - 2))
    return float((np.mean(in_vals) - np.mean(out_vals)) / sp) if sp > 0 else 0.0


def compute_d_matrix(in_mask):
    d_mat = np.zeros((N_MODS, N_LINES))
    for mi, mod in enumerate(PHENO_MODS):
        data = raw_drug_level[mod].values
        for li in range(N_LINES):
            col = data[:, li]
            vi = col[in_mask][~np.isnan(col[in_mask])]
            vo = col[~in_mask][~np.isnan(col[~in_mask])]
            d_mat[mi, li] = cohens_d_vec(vi, vo)
    return d_mat


# Observed
observed_d = {k: compute_d_matrix(labels_1idx == k) for k in cluster_ids}

# Permutation null
print(f'Running {N_PERM} permutations...')
perm_counts = {k: np.zeros((N_MODS, N_LINES)) for k in cluster_ids}
for p in tqdm(range(N_PERM), desc='Permutations'):
    rng = np.random.RandomState(p)
    perm_labels = rng.permutation(labels_1idx)
    for k in cluster_ids:
        d_perm = compute_d_matrix(perm_labels == k)
        perm_counts[k] += (np.abs(d_perm) >= np.abs(observed_d[k])).astype(float)

cohens_results = {k: (observed_d[k], (perm_counts[k] + 1) / (N_PERM + 1))
                  for k in cluster_ids}


# ── Cell-line ordering: by mean |d| across clusters/modalities ──────────────
all_d = np.stack([cohens_results[k][0] for k in cluster_ids], axis=0)
line_order = np.argsort(np.abs(all_d).mean(axis=(0, 1)))[::-1]
sorted_lines = [CELL_LINES[i] for i in line_order]


def save_all(fig, basename):
    out = os.path.join(OUT_DIR, basename)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {basename}.{{pdf,png,svg}}')


# ── Plot helper: SQUARE cells via aspect='equal' ────────────────────────────
VABS = 2.0
CELL_SIZE = 0.22  # inches per cell

def plot_one_cluster(k, ax, show_ylabel=True, show_xlabel=True, title_color=None):
    d_mat, p_mat = cohens_results[k]
    d_plot = d_mat[:, line_order]
    p_plot = p_mat[:, line_order]

    im = ax.imshow(d_plot, cmap='RdBu_r', vmin=-VABS, vmax=VABS,
                   aspect='equal', interpolation='nearest')

    # Significance dots
    for mi in range(N_MODS):
        for li in range(N_LINES):
            if p_plot[mi, li] < 0.05:
                ax.text(li, mi, '*', ha='center', va='center',
                        fontsize=7, color='black', fontweight='bold')

    if show_xlabel:
        ax.set_xticks(range(N_LINES))
        ax.set_xticklabels(sorted_lines, rotation=90, fontsize=6.5)
    else:
        ax.set_xticks([])
    if show_ylabel:
        ax.set_yticks(range(N_MODS))
        ax.set_yticklabels([MOD_DISPLAY[m] for m in PHENO_MODS], fontsize=9)
    else:
        ax.set_yticks([])

    n_k = int((labels_1idx == k).sum())
    ax.set_title(f'Cluster {k} (n={n_k})',
                 color=title_color or PALETTE[k],
                 fontweight='bold', fontsize=11)
    return im


# ── Multi-panel figure: clusters stacked vertically (square cells) ──────────
# One row per cluster, all sharing same y-axis labels
fig_h = N_CLUSTERS * (CELL_SIZE * N_MODS + 0.4) + 1.5
fig_w = CELL_SIZE * N_LINES + 2.5
fig, axes = plt.subplots(N_CLUSTERS, 1,
                          figsize=(fig_w, fig_h),
                          sharex=True)
if N_CLUSTERS == 1:
    axes = [axes]

for idx, (ax, k) in enumerate(zip(axes, cluster_ids)):
    is_last = (idx == N_CLUSTERS - 1)
    im = plot_one_cluster(k, ax, show_ylabel=True, show_xlabel=is_last)

# Shared colorbar to the right
cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02, location='right')
cbar.set_label("Cohen's d (cluster vs. rest)", fontsize=9)

fig.suptitle("Cohen's d: cluster vs. rest per modality × cell line\n"
             "(* permutation p < 0.05, 1000 perms)",
             fontsize=11, fontweight='bold', y=0.995)
save_all(fig, 'cohens_d_all_clusters')


# ── Per-cluster individual heatmaps (square cells) ──────────────────────────
for k in cluster_ids:
    fig, ax = plt.subplots(figsize=(CELL_SIZE * N_LINES + 3,
                                     CELL_SIZE * N_MODS + 2.5))
    im = plot_one_cluster(k, ax, show_ylabel=True, show_xlabel=True)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Cohen's d", fontsize=9)
    save_all(fig, f'cohens_d_cluster_{k}')


# ── Save long-form CSV ──────────────────────────────────────────────────────
rows = []
for k in cluster_ids:
    d_mat, p_mat = cohens_results[k]
    for mi, mod in enumerate(PHENO_MODS):
        for li, line in enumerate(CELL_LINES):
            rows.append({
                'cluster': k, 'modality': mod, 'cell_line': line,
                'cohens_d': d_mat[mi, li], 'p_value': p_mat[mi, li],
            })
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'cohens_d_values.csv'),
                          index=False)
print('Saved: cohens_d_values.csv')

for k in cluster_ids:
    _, p_mat = cohens_results[k]
    n_sig = (p_mat < 0.05).sum()
    print(f'Cluster {k}: {n_sig}/{N_MODS * N_LINES} cells significant at p<0.05')
