# Phenotypic metrics

One section per metric: what it measures, how it was computed, and which
script computes it.

Every metric is computed per **(cell line, drug, dose)** against the DMSO_TF
control cells of the same cell line on the same plate, then aggregated across
the 14 plates into a cell lines x drug-doses matrix. Those matrices are the
layers of the metrics AnnData (see `packaging/`).

---

## Gini heterogeneity — `gini`

**What it measures.** Whether a drug's response is concentrated in a few
transcriptional states or spread evenly across them. Cells are clustered, and
the Gini coefficient of a condition's cluster-occupancy vector is high when its
cells pile into a few clusters and low when they distribute evenly.

**How it was computed** (`metrics/gini.py`, one array task per plate). For each
(cell line, drug-dose): subsample to 1000 cells per condition, pool with the
matching DMSO_TF control cells, then normalise / log1p / PCA / neighbours /
Leiden at resolutions 1.0, 2.0 and 3.0. Build the cluster x condition
contingency table, take each condition's Gini, and z-score it against 100
label permutations. A chi-square test on the table and a KS test between the
drug and control cluster distributions are reported alongside.

```
python metrics/gini.py <plate>.h5ad --output_dir <dir> \
    --perturbation_col drugname_drugconc \
    --control_condition "[('DMSO_TF', 0.0, 'uM')]" \
    --cell_line_col cell_name \
    --resolutions 1.0,2.0,3.0 --n_subsample 1000 \
    --n_permutations 100 --n_tasks 8 --random_seed 42
```

**Aggregation** (`metrics/aggregate_gini.py`). Conditions with fewer than 10
clusters, or fewer than 10 cells per cluster on average, are dropped — the Gini
of an occupancy vector is not meaningful when the clustering is that coarse.
This removes roughly 12% of rows per plate. The surviving values are averaged
over the three resolutions, then over plates, and pivoted to a matrix.

The `gini` layer is a **mean across resolutions 1.0/2.0/3.0**,
not a single resolution.

**Units and direction.** Permutation z-score; higher means the drug's cells are
more concentrated in fewer clusters than chance. Observed range is -4.95 to 40.15, with about 5% missing.

---

## Augur — `augur`

**What it measures.** How strongly a drug perturbs a given cell line. Augur
trains a classifier to separate perturbed cells from controls; the
cross-validated AUC is the perturbation score. A cell line whose transcriptome
barely moves under a drug is hard to classify and scores near 0.5.

**How it was computed** (`metrics/augur.py`, one array task per plate). For each
(drug, dose): pool the drug's cells with the plate's DMSO_TF cells, normalise
excluding highly expressed genes, log1p, take the top 2000 Seurat highly
variable genes, then

```python
pertpy.tl.Augur("random_forest_classifier").predict(
    subsample_size=20, select_variance_features=False, cell_type_col="cell_name")
```

Scoring by `cell_name` means each cell line is scored separately inside the
shared drug-vs-control comparison.

**Aggregation** (`metrics/aggregate_augur.py`). Each `summary_metrics__*.csv`
has one column per cell line, so a column mean gives that cell line's score for
that drug-dose. Rows are assembled per plate and averaged across plates.

**Units and direction.** Classifier AUC; higher means a
stronger transcriptional response. Observed range is
0.43 to 0.999, with about 2% missing.

---

## Within-population heterogeneity — `heterogeneity`

**What it measures.** How dispersed a drug-treated population is in expression
space, relative to its DMSO control. A drug that pushes cells into several
distinct states raises the spread; one that collapses them onto a single state
lowers it.

**How it was computed** (`metrics/heterogeneity.py`, one plate per invocation).
For each (cell line, drug): subset to that drug's cells plus the DMSO_TF
control cells of the same cell line and plate, then

```
normalize_total -> log1p -> PCA
X = X_pca * uns["pca"]["variance"]      # weight each PC by its variance
distance = mean ||x - centroid|| within each group
```

giving `drug_within_distance` and `dmso_within_distance` per condition.

**The ratio** (`metrics/heterogeneity_ratio.R`) is
`drug_within_distance / dmso_within_distance`.

> A **linear ratio.** Values are non-negative with a median of 0.975 and range
> 0 to 3.17; above 1 means the drug-treated population is more dispersed than
> its control.

**Aggregation** (`metrics/aggregate_heterogeneity.py`) pivots the ratios to
cell lines x drug-doses.

**Units and direction.** Dimensionless ratio; higher means more within-population
heterogeneity under drug than under control.

### Notes

**Replicates.** About 20% of conditions were assayed on more than one plate; `aggregate_heterogeneity.py` averages them (`--dedup mean`, the default); `--dedup first` keeps the first occurrence.

---

## Cell counts — input to survival and G arrest

**What it is.** A tally of surviving cells per plate x cell line x drug x dose
x cell cycle phase, restricted to `pass_filter == "full"` cells. Both
count-based metrics build on it: survival compares totals against DMSO, G
arrest compares the phase distribution against DMSO.

**How it was computed** (`metrics/count_cells.py`). Each plate is opened in
backed mode and only `.obs` is touched, so the expression matrix never enters
memory. The `drugname_drugconc` string is parsed into drug, concentration and
unit, `phase` is bucketed to G1 / G2M / S / Other, and the result is grouped
and counted.

---

## G arrest — `G_arrest`

**What it measures.** Whether a drug pushes surviving cells out of cycle. Of the
cells that remain, what fraction sits in a gap phase rather than replicating,
relative to DMSO?

**How it was computed** (`metrics/g_arrest.py`). From the phase-resolved counts,
for each (plate, cell line, drug, dose):

```
G_arrest_proportion = (G1 + G2M) / (G1 + G2M + S)
G_arrest_log2FC     = log2( (p_drug + 1e-9) / (p_DMSO + 1e-9) )
```

where `p_DMSO` is the mean DMSO_TF proportion for the same cell line on the
same plate. Values are then averaged across plates and pivoted to drug-doses x
cell lines. Conditions with a G arrest proportion of exactly zero are dropped
before the ratio is taken.

> **The layer is a log2 fold change** (the output file is named
> `G_arrest_proportions.csv`). Values run -1.50 to 0.74 and
> are centred near zero; positive means more of the surviving population is in
> G1/G2M than in DMSO.

---

## Survival — `sensitivity`

**What it measures.** Whether a cell line is killed by a drug. Tahoe pools many
cell lines into one well, so the readout is each line's **share** of the well:
a line that dies loses share, one that tolerates the drug gains it. Absolute

**How it was computed** (`metrics/survival_log2fc.py`). Per plate: pivot to
drug-dose x cell line counts, normalise each drug row to sum to 1, average the
shares across that plate's DMSO_TF wells for the untreated baseline, then

```
log2( (share_drug + 0.01) / (share_DMSO + 0.01) )
```

The pseudocount keeps the ratio finite for lines a drug wipes out. Values are
averaged across plates and pivoted to drug-doses x cell lines.

**Units and direction.** log2 fold change in compositional share. Negative means
the line lost share relative to DMSO, i.e. was killed. Range is -1.73
to 2.08.

---

## Survival log-odds — figures only

**How it was computed** (`metrics/survival_fisher.py`). The same question tested
formally. For each (plate, cell line, drug, dose):

```
                    this cell line    all other lines
drug well                 a                  c
pooled DMSO wells         b                  d
```

Fisher's exact test one-sided (`alternative="less"`, testing for depletion),
effect size a log2 odds ratio with Haldane-Anscombe 0.5 correction, p-values
BH-FDR corrected across all conditions.

This feeds figures and the tissue/MOA specificity tables rather than a layer of
the metrics AnnData.

**Note on the two survival measures.** `sensitivity` (log2 fold change) and this
log-odds ratio answer the same question with different framings and are both
reported. The log2FC version is what became the AnnData layer.

---

## G arrest log-odds — `G1_arrest_logodds`, `G2M_arrest_logodds`, `G_arrest_logodds`

**What it measures.** The same phase redistribution as `G_arrest`, but tested
against the plate's pooled DMSO control rather than expressed as a ratio of
proportions.

**How it was computed** (`metrics/g_arrest_fisher.py`). For each (plate, cell
line, drug, dose) and each phase group:

```
                    in group-of-interest   not in group
drug well                    a                   c
pooled DMSO                  b                   d
```

Fisher's exact test two-sided for the p-value; effect size a log odds ratio
with Haldane-Anscombe 0.5 correction; BH-FDR across all conditions within an
analysis. Four groupings are run — G1, G2M, S, and G1+G2M — of which three
become layers.

> **The `*_logodds` layers use the natural log.** `--log_base 2` gives log2.

---

## Pseudobulk correlation — replicate concordance vs cells sampled

Not a layer.

**How it was computed** (`pseudobulk_correlation.py`, then
`plot_pseudobulk_correlation.py`). For each (cell line, drug-dose) shared
between plates 6 and 14: subsample N drug cells per plate while holding the
DMSO control fixed (pseudobulked over all control cells on that plate), form
the signature `LFC = mean(drug) - mean(DMSO)` over highly variable genes, and
correlate the plate-6 and plate-14 signatures across genes. Sweeping
N = 25 ... 1600 gives a saturation curve.


