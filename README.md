# tahoe100m_analysis

Analysis code for the Tahoe-100M drug perturbation study: per-condition
phenotypic metrics, SNF clustering of drugs on those metrics, VISION pathway
scoring, and a pseudobulk signature reproducibility analysis.

Each script states its inputs in an editable block at the top. The single-cell
data itself is not distributed here.

## Metrics

One script per metric, computed per (cell line, drug, dose) against the
matched DMSO control. `metrics/count_cells.py` produces the cell counts the
count-based metrics read.

| Layer | Measures | Script |
|---|---|---|
| `sensitivity` | Cell line's share of a pooled well vs DMSO, log2 | `metrics/survival_log2fc.py` |
| `heterogeneity` | Within-population dispersion vs DMSO (ratio) | `metrics/heterogeneity.py` + `heterogeneity_ratio.R` |
| `gini` | Concentration of the response into few cell states | `metrics/gini.py` |
| `augur` | Classifier separability of drug vs control | `metrics/augur.py` |
| `G_arrest` | (G1+G2M)/(G1+G2M+S) vs DMSO, log2 | `metrics/g_arrest.py` |
| `*_logodds` | Fisher log-odds of phase enrichment (natural log) | `metrics/g_arrest_fisher.py` |

The `aggregate_*.py` scripts combine per-plate outputs into the final
cell line × drug-dose matrices. `packaging/build_tahoe_metrics.py` assembles
those into the released AnnData.

## Released data

50 cell lines × 1137 drug-doses, one layer per metric, on HuggingFace at
[`arcinstitute/tahoe_phenotypic_metrics`](https://huggingface.co/datasets/arcinstitute/tahoe_phenotypic_metrics).

```python
from packaging.load_metrics import load_metrics, describe
adata = load_metrics()
describe(adata, "heterogeneity")   # units, direction, notes for any layer
```

Every layer's definition and units are in `adata.uns["layers"]`.

## SNF clustering

`integration/snf_pipeline.py` fuses the metric modalities and clusters the
drugs into four groups; the assignments are in `reference/snf_clusters.csv`.
Figures: `fig_cosine_heatmap.py`, `fig_cohens_d.py`, `moa_enrichment.py`.
Method details in [docs/METHODS_snf_clustering.md](docs/METHODS_snf_clustering.md).

## VISION

`vision/vision_analysis.R` runs VISION pathway scoring on pseudobulk h5ads;
`vision/vision_line_pca.py` reduces the scores to a per-cell-line drug
embedding.

## Pseudobulk correlation

`pseudobulk_correlation.py` measures how reproducible a drug's expression
signature is as a function of cells sampled per condition, by subsampling
against a fixed DMSO pseudobulk and correlating signatures across replicate
plates. `plot_pseudobulk_correlation.py` draws the saturation curve.

## Environment

`environment.yml` pins the package versions used. R scripts need `arrow`,
`dplyr`, `stringr`, and for VISION the `VISION` and `DESeq2` packages.
