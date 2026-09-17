# Methods — Multi-modal drug clustering by Similarity Network Fusion

## Phenotypic feature construction

Drugs were characterized by six phenotypic modalities derived from the
single-cell data and aggregated to one value per drug per cell line: drug
sensitivity (survival log2 odds-ratio), G1 arrest (log2 odds-ratio), G2/M
arrest (log2 odds-ratio), Augur perturbation score, Gini cluster-occupancy
heterogeneity, and transcriptional heterogeneity (within-population distance).
These per-modality drug × cell-line matrices were read from the aggregated
metrics object (`tahoe_metrics_aggregated.h5ad`) and restricted to the highest
common dose (5 µM), giving 379 drugs profiled across 47 cancer cell lines.
Within each modality, the drug × cell-line matrix was standardized (z-scored)
across drugs separately for each cell line, and residual missing values were
set to zero. This yielded six modality-specific feature matrices (379 drugs ×
47 cell lines each; 282 features per drug in total).

## Similarity Network Fusion

The six feature matrices were integrated into a single drug-similarity network
using Similarity Network Fusion (SNF; Wang et al., Nat. Methods 2014), as
implemented in snfpy (v0.2.2). For each modality, a drug × drug affinity matrix
was computed from pairwise Euclidean distances with a scaled-exponential
(Gaussian) kernel, using K = max(10, √N_drugs) = 19 nearest neighbours and
kernel-width hyperparameter µ = 0.5. The six affinity networks were then fused
by SNF cross-diffusion (K = 19) into a single consensus affinity network over
all 379 drugs.

## Leiden clustering

The fused affinity network was converted to a weighted k-nearest-neighbour
graph by retaining, for each drug, edges to its K = 19 strongest neighbours and
symmetrizing the result; edge weights were the fused affinities. Communities
were detected with the Leiden algorithm (leidenalg v0.11.0, igraph v1.0.0) using
the RBConfiguration vertex partition and a fixed random seed (42). The
resolution parameter was selected by scanning values from 0.05 to 1.50 in steps
of 0.05 and recording, at each resolution, the number of clusters and the
silhouette score (computed on a precomputed distance matrix defined as
1 − the min–max-normalized KNN affinity). A resolution of 0.4 was chosen as it
lay in a stable region of the cluster-count curve and yielded a strong
silhouette score, partitioning the 379 drugs into four clusters (118, 108, 86,
and 67 drugs).

## UMAP visualization

For visualization, the same fused-affinity KNN graph was supplied to Scanpy
(v1.11.5) as a precomputed neighbourhood graph (n_neighbors = K = 19), and a
two-dimensional embedding was computed with `scanpy.tl.umap` (umap-learn
v0.5.9) using `min_dist = 0.4`, `spread = 1.2`, and a fixed random seed (42).
Drugs were plotted in this embedding coloured by Leiden cluster and by annotated
mechanism of action.

## Clustering robustness

Cluster stability was assessed in two ways. First, a modality leave-one-out
analysis re-ran the full SNF and Leiden pipeline (resolution 0.4) on each subset
of five of the six modalities and measured agreement with the full-modality
clustering by the adjusted Rand index, testing whether any single modality
drove the partition. Second, a drug-subsampling bootstrap drew 100 random
subsets of 80% of drugs (K recomputed as √n for each subset), re-ran SNF and
Leiden clustering on each, and recorded the frequency with which each pair of
drugs co-clustered when both were sampled, summarizing the stability of
drug–drug relationships.

## Cluster characterization

For each cluster, modality, and cell line, an effect size was computed as the
pooled-variance Cohen's d of the unscaled metric values comparing drugs in that
cluster against all other drugs. Significance was assessed by a label-permutation
test: cluster labels were shuffled 1,000 times and Cohen's d was recomputed for
each (cluster, modality, cell line) under each permutation, giving a two-sided
empirical p-value of (#{|d_perm| ≥ |d_obs|} + 1) / (N_perm + 1).

Separately, mechanism-of-action enrichment within each cluster was tested with a
one-sided Fisher's exact test (over-representation, `alternative='greater'`) on a
2×2 table of cluster membership versus MOA membership for each cluster–MOA pair,
with Benjamini–Hochberg FDR correction across all tests.

*Software versions: snfpy 0.2.2, scanpy 1.11.5, leidenalg 0.11.0, igraph 1.0.0,
umap-learn 0.5.9, scikit-learn 1.7.2, anndata 0.12.7, numpy 2.2.4, scipy 1.16.3.*
