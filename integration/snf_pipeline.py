"""Similarity network fusion over the phenotypic modalities, and Leiden clustering.

This is the recipe behind the drug clustering in the paper. It was duplicated
verbatim at the top of each figure script; it lives here once and the figure
scripts import it.

The pipeline:

  1. drop three cell lines with unreliable coverage, and drop the layers not
     used for clustering (`G_arrest` and `G_arrest_logodds` are redundant with
     the log-odds layers; `edist` was computed externally)
  2. keep only the top dose (`__5.0`) so every drug contributes one profile
  3. z-score each modality across drugs, within cell line
  4. build a per-modality affinity graph and fuse them with SNF
     (K = max(10, sqrt(n_drugs)) = 19, mu = 0.5)
  5. Leiden on the fused affinity KNN graph
     (RBConfiguration, resolution 0.4, seed 42) -> 4 clusters

The clustering is 4 clusters of 118 / 108 / 86 / 67 drugs, stored in
`reference/snf_clusters.csv`. **`run_snf` returns those stored labels by
default** rather than whatever this run computes: Leiden is only reproducible
for a fixed igraph + leidenalg pair, and K depends on how many drugs survive
filtering. Pass `labels="recompute"` to cluster the data in front of you, and
`check_reference_clusters` to compare the result against the stored labels.

Software versions used: snfpy 0.2.2, scanpy 1.11.5, leidenalg 0.11.0,
umap-learn 0.5.9. See docs/METHODS_snf_clustering.md.
"""

import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import igraph as ig
import leidenalg
from sklearn.preprocessing import StandardScaler
from snf import compute, snf as snf_fuse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_names import normalize_drug, match_key  # noqa: E402

# --- paths: edit for your system ---
H5AD = "/large_storage/ctc/public/tahoe/tahoe_metrics_aggregated.h5ad"
DRUG_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reference", "tahoe-drugs.csv")
CLUSTERS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "reference", "snf_clusters.csv")

# Cell lines with unreliable coverage across plates.
DROP_LINES = ["NCI-H661", "NCI-H2122", "NCI-H596"]
# G_arrest / G_arrest_logodds duplicate the phase-specific log-odds layers;
# edist has no in-repo provenance.
DROP_LAYERS = ["G_arrest", "G_arrest_logodds", "edist"]
DOSE_SUFFIX = "__5.0"
CHOSEN_RES = 0.4
MU = 0.5
SEED = 42

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,   # keep text editable in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


class SNFResult:
    """Everything the figure scripts need from one run of the pipeline."""

    def __init__(self, phenotypes, scaled, raw, drug_index, cell_lines, fused, knn,
                 labels, K):
        self.phenotypes = phenotypes
        self.scaled = scaled            # modality -> drugs x cell lines, z-scored
        self.raw = raw                  # modality -> drugs x cell lines, unscaled
        self.drug_index = drug_index    # drug names, without the dose suffix
        self.cell_lines = cell_lines
        self.fused = fused              # fused affinity matrix
        self.knn = knn                  # KNN-sparsified fused affinity
        self.labels = labels            # 0-indexed cluster membership
        self.K = K

    @property
    def modalities(self):
        return list(self.scaled.keys())

    @property
    def labels_1idx(self):
        """Cluster labels as they appear in the figures (1-indexed)."""
        return self.labels + 1

    def cluster_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"drug": self.drug_index, "cluster": self.labels_1idx})


def load_phenotypes(h5ad: str = H5AD) -> ad.AnnData:
    """Read the metrics AnnData and drop the excluded cell lines and layers."""
    phenotypes = ad.read_h5ad(h5ad)
    phenotypes = phenotypes[~phenotypes.obs.index.isin(DROP_LINES)].copy()
    for layer in DROP_LAYERS:
        if layer in phenotypes.layers:
            del phenotypes.layers[layer]
    return phenotypes


def scale_modalities(phenotypes: ad.AnnData):
    """Top-dose slice per modality, z-scored and raw.

    Returns (scaled, raw, drug_index, cell_lines). The raw frames keep the
    original units, which the violin and Cohen's d figures plot.
    """
    cell_lines = phenotypes.obs_names.tolist()
    top_dose = phenotypes[:, phenotypes.var_names.str.endswith(DOSE_SUFFIX)].copy()
    # Normalise drug names so they join against the reference tables.
    drug_index = pd.Index(
        [normalize_drug(d) for d in
         top_dose.var_names.str.replace(DOSE_SUFFIX, "", regex=False)])

    scaled, raw = {}, {}
    for layer in top_dose.layers:
        df = pd.DataFrame(top_dose.layers[layer].T, index=drug_index, columns=cell_lines)
        raw[layer] = df.copy()
        # z-score across drugs within each cell line; NaNs become the column mean.
        scaled[layer] = pd.DataFrame(StandardScaler().fit_transform(df.values),
                                     index=df.index, columns=df.columns).fillna(0)
    return scaled, raw, drug_index, cell_lines


def leiden_cluster(affinity: np.ndarray, k_neighbors: int, resolution: float,
                   seed: int = SEED):
    """Leiden on the KNN-sparsified affinity graph. Returns (labels, knn matrix)."""
    n = affinity.shape[0]
    knn = np.zeros_like(affinity)
    for i in range(n):
        top = np.argsort(affinity[i])[-k_neighbors:]
        knn[i, top] = affinity[i, top]
    knn = np.maximum(knn, knn.T)   # symmetrise

    src, tgt = np.nonzero(np.triu(knn, k=1))
    graph = ig.Graph(n=n, edges=list(zip(src.tolist(), tgt.tolist())),
                     edge_attrs={"weight": knn[src, tgt].tolist()}, directed=False)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    return np.array(partition.membership), knn


def load_reference_labels(drug_index, clusters_csv: str = CLUSTERS_CSV):
    """The stored cluster labels, aligned to `drug_index` (0-indexed)."""
    stored = pd.read_csv(clusters_csv)
    # Whitespace-insensitive lookup; see drug_names.match_key.
    lookup = dict(zip(stored["drug"].map(match_key), stored["cluster"]))
    missing = [d for d in drug_index if match_key(d) not in lookup]
    if missing:
        raise SystemExit(
            f"{len(missing)} drugs absent from {clusters_csv}, e.g. {missing[:3]}. "
            "Drug names may not be canonical; see drug_names.py.")
    return np.array([lookup[match_key(d)] for d in drug_index]) - 1


def run_snf(h5ad: str = H5AD, resolution: float = CHOSEN_RES,
            mu: float = MU, seed: int = SEED,
            labels: str = "reference") -> SNFResult:
    """Run the full pipeline and return everything downstream needs.

    labels="reference" (default) returns the stored cluster assignments;
    labels="recompute" returns the Leiden labels from this run. The affinity
    matrix, KNN graph and scaled modalities are computed either way.
    """
    phenotypes = load_phenotypes(h5ad)
    scaled, raw, drug_index, cell_lines = scale_modalities(phenotypes)

    n_drugs = len(drug_index)
    K = max(10, int(np.sqrt(n_drugs)))
    affinities = compute.make_affinity([df.values for df in scaled.values()],
                                       metric="euclidean", K=K, mu=mu)
    fused = snf_fuse(affinities, K=K)
    computed, knn = leiden_cluster(fused, K, resolution, seed)

    if labels == "reference":
        membership = load_reference_labels(drug_index)
    elif labels == "recompute":
        membership = computed
    else:
        raise ValueError("labels must be 'reference' or 'recompute'")

    return SNFResult(phenotypes, scaled, raw, drug_index, cell_lines, fused, knn,
                     membership, K)


def check_reference_clusters(result: SNFResult, clusters_csv: str = CLUSTERS_CSV):
    """Fail loudly if the clustering differs from the stored labels.

    Leiden's seed only pins the result for a fixed igraph + leidenalg pair, and
    K depends on how many drugs survive filtering.
    """
    if not os.path.exists(clusters_csv):
        print(f"WARNING: {clusters_csv} not found; skipping cluster check")
        return
    stored = pd.read_csv(clusters_csv)
    expected = dict(zip(stored["drug"].map(match_key), stored["cluster"]))
    current = {match_key(d): c for d, c in
               zip(result.drug_index, result.labels_1idx)}

    missing = [d for d in expected if d not in current]
    if missing:
        raise AssertionError(f"{len(missing)} reference drugs missing from this run")

    changed = [d for d in expected if current[d] != expected[d]]
    if changed:
        raise AssertionError(
            f"{len(changed)}/{len(expected)} drugs differ from the stored "
            "clusters -- check igraph/leidenalg versions against "
            "docs/METHODS_snf_clustering.md")
    print(f"cluster check passed: {len(expected)} drugs match reference/snf_clusters.csv")


def load_drug_annotations(drug_csv: str = DRUG_CSV) -> pd.DataFrame:
    """Drug -> MOA table used to annotate the clusters."""
    return pd.read_csv(drug_csv, index_col=0)


if __name__ == "__main__":
    res = run_snf(labels="recompute")
    sizes = pd.Series(res.labels_1idx).value_counts().sort_index()
    print(f"{len(res.drug_index)} drugs, {len(res.cell_lines)} cell lines, "
          f"{len(res.modalities)} modalities: {res.modalities}")
    print(f"K = {res.K}, resolution = {CHOSEN_RES}, seed = {SEED}")
    print(f"{len(sizes)} clusters, sizes: {sizes.tolist()}")
