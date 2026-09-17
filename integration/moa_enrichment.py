"""
MOA fold-enrichment heatmap for the SNF clusters (Fig 4C).

Color  = fold enrichment, observed/expected = A * n_total / (n_k * n_moa_total)
         (1 = no enrichment -> white; >1 = over-represented -> blue). Bounded,
         power-independent effect size.
Outline = FDR significance tier from a one-sided Fisher's exact test
         (q<0.10 thin grey, q<0.05 black, q<0.01 thick black).

This decouples effect (color), sample size, and inference (outline): a strongly
enriched but small MOA shows dark with no outline.

Versions:
  CANONICAL (include_unclear=False) -> moa_enrichment_heatmap_snf.{pdf,png}
     annotated MOAs only; background = 178 annotated drugs (Fig 4C panel).
  WITH-UNCLEAR (include_unclear=True) -> moa_enrichment_heatmap_snf_with_unclear.{pdf,png}
     adds an `unclear` row (Unknown merged in); background = all 379 drugs.
     Descriptive only — unclear is annotation coverage, not a mechanism.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Frozen SNF cluster labels with their MOA annotation; see integration/snf_pipeline.py.
CLUSTERS_CSV = os.path.join(REPO, 'reference', 'snf_clusters.csv')
ROOT = os.environ.get('TAHOE_FIGURE_DIR') or os.path.join(REPO, 'results', 'figures')
os.makedirs(ROOT, exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

a = pd.read_csv(CLUSTERS_CSV)[['drug', 'cluster', 'moa_fine']].copy()
a['moa_fine'] = a['moa_fine'].replace('Unknown', 'unclear')
clusters = sorted(a['cluster'].unique())
CMAP = 'Blues'


def compute(include_unclear: bool):
    universe = a if include_unclear else a[a['moa_fine'] != 'unclear']
    cats = sorted(universe['moa_fine'].unique())
    n_total = len(universe)
    n_k = {k: int((universe['cluster'] == k).sum()) for k in clusters}

    rows = []
    for k in clusters:
        in_k = universe['cluster'] == k
        for cat in cats:
            in_cat = universe['moa_fine'] == cat
            A = int((in_k & in_cat).sum()); B = int((in_k & ~in_cat).sum())
            C = int((~in_k & in_cat).sum()); D = int((~in_k & ~in_cat).sum())
            n_moa = A + C
            fold = (A * n_total) / (n_k[k] * n_moa) if A > 0 else 0.0
            pct = 100.0 * A / n_moa if n_moa > 0 else 0.0   # % of this MOA's drugs in cluster k
            p = fisher_exact([[A, B], [C, D]], alternative='greater')[1] if A > 0 else 1.0
            rows.append(dict(cluster=k, moa=cat, n_in_cluster=A, n_moa_total=n_moa,
                             pct_of_moa=pct, fold_enrichment=fold, p_value=p, tested=A > 0))
    res = pd.DataFrame(rows)
    # FDR only across genuinely tested cells (A>0)
    res['q_value'] = 1.0
    m = res['tested']
    res.loc[m, 'q_value'] = multipletests(res.loc[m, 'p_value'], method='fdr_bh')[1]
    res.to_csv(f'{ROOT}/moa_enrichment_snf_results{"_with_unclear" if include_unclear else ""}.csv',
               index=False)

    pct = res.pivot_table(index='moa', columns='cluster', values='pct_of_moa', fill_value=0)
    qpv = res.pivot_table(index='moa', columns='cluster', values='q_value', fill_value=1.0)
    pct = pct.reindex(index=sorted(cats), columns=clusters, fill_value=0)
    qpv = qpv.reindex(index=sorted(cats), columns=clusters, fill_value=1.0)
    return res, pct, qpv


def plot(pct, qpv, suffix):
    order = list(pct.index)
    fig, ax = plt.subplots(figsize=(4.4, 0.34 * len(order) + 1.8))
    im = ax.imshow(pct.values, cmap=CMAP, vmin=0, vmax=100, aspect='equal')
    ax.set_xticks(np.arange(-0.5, len(clusters), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=1.0)
    ax.tick_params(which='minor', length=0)

    # FDR outline: q < 0.10
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            if qpv.iloc[i, j] < 0.10:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor='black', lw=2.0))

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f'C{c}' for c in clusters], fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == 'unclear':
            lbl.set_style('italic'); lbl.set_color('#b00000')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Mechanism of Action (MOA)', fontsize=11)
    ax.set_title('Distribution of each MOA across clusters\n(outlined: q < 0.10, BH-FDR)', fontsize=11)
    cb = plt.colorbar(im, ax=ax, shrink=0.45, pad=0.03)
    cb.set_label("% of MOA's drugs in cluster", fontsize=10)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{ROOT}/moa_enrichment_heatmap_snf{suffix}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'saved moa_enrichment_heatmap_snf{suffix}.{{pdf,png}}')


for inc, suf in [(False, ''), (True, '_with_unclear')]:
    _, pct, qpv = compute(inc)
    plot(pct, qpv, suf)
