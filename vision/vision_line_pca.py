"""Reduce VISION pathway scores to a per-cell-line drug embedding.

Takes the pseudobulked VISION signature scores and produces the drug embedding
the downstream MOA radar consumes.

For each drug-dose, VISION gives a (cell line x pathway) matrix. Stacking those
row-major turns each drug into one long vector over (cell line, pathway) pairs.
The PCA is then run **separately per cell line** -- for each cell line, take
that line's block of pathway columns across all drugs and reduce it to 5 PCs.
Concatenating the per-line blocks gives a drug x (n_lines * 5) embedding with
columns named `<cell line>_PC1` ... `<cell line>_PC5`.

Doing it per line rather than globally keeps each line's pathway response on its
own axes, so a drug's profile stays resolved by cell line instead of being
averaged into a single global response.

Drugs missing any value across the shared (line, pathway) grid are dropped
rather than imputed, which is why the output carries fewer drugs than the
metrics AnnData.

Outputs:
    vision_scores_line_pca.csv           drugs x (cell line PCs)
    vision_scores_line_pca_loadings.csv  pathway loadings per cell line

Usage:
    python vision_line_pca.py --input vision_scores_pseudobulk.h5ad \\
                              --output_dir results/vision
"""

import os
import argparse

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- paths: edit for your system ---
# Pseudobulked VISION scores: obs carries drug_conc and the cell line name,
# var is the pathway/signature axis. Built from the per-plate TSVs written by
# vision_analysis.R.
INPUT = "vision_scores_pseudobulk.h5ad"
OUTPUT_DIR = "results/vision"

DRUG_COL = "drug_conc"
LINE_COL = "Cell_Name_Vevo"
N_COMPONENTS = 5


def drug_matrices(adata, drug_col: str, line_col: str) -> dict:
    """Split into one (cell line x pathway) frame per drug-dose."""
    out = {}
    for drug in adata.obs[drug_col].unique():
        sub = adata[adata.obs[drug_col] == drug]
        X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        out[drug] = pd.DataFrame(X, index=sub.obs[line_col], columns=adata.var_names)
    return out


def flatten(matrices: dict):
    """Stack each drug's (line x pathway) matrix into one row on a shared grid."""
    shared_pathways = sorted(set.intersection(*(set(m.columns) for m in matrices.values())))
    shared_lines = list(next(iter(matrices.values())).index)

    rows, names, skipped = [], [], 0
    for drug, df in matrices.items():
        flat = df.reindex(index=shared_lines, columns=shared_pathways).values.flatten(order="C")
        if np.any(pd.isna(flat)):
            skipped += 1        # dropped rather than imputed
            continue
        rows.append(flat)
        names.append(drug)

    columns = [f"{line}_{path}" for line in shared_lines for path in shared_pathways]
    print(f"  {len(shared_lines)} cell lines x {len(shared_pathways)} pathways")
    print(f"  {len(names)} drugs kept, {skipped} dropped for missing values")
    return pd.DataFrame(rows, index=names, columns=columns), shared_lines


def per_line_pca(flat: pd.DataFrame, lines, n_components: int):
    """PCA within each cell line's block of pathway columns."""
    blocks, labels, loadings = [], [], {}
    for line in lines:
        features = [c for c in flat.columns if c.startswith(f"{line}_")]
        scaled = StandardScaler().fit_transform(flat[features])

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)[-1]
        print(f"    {line:20s} {len(features):5d} pathways, "
              f"{cumulative:.1%} variance in {n_components} PCs")

        blocks.append(scores)
        labels.extend([f"{line}_PC{i + 1}" for i in range(scores.shape[1])])
        loadings[line] = pd.DataFrame(
            pca.components_.T, index=features,
            columns=[f"PC{i + 1}" for i in range(n_components)])

    embedding = pd.DataFrame(np.hstack(blocks), index=flat.index, columns=labels)
    return embedding, pd.concat(loadings, names=["Cell_Line", "Feature"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--drug_col", default=DRUG_COL)
    ap.add_argument("--line_col", default=LINE_COL)
    ap.add_argument("--n_components", type=int, default=N_COMPONENTS)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.n_obs} pseudobulk samples x {adata.n_vars} signatures")

    matrices = drug_matrices(adata, args.drug_col, args.line_col)
    print(f"\n{len(matrices)} drug-doses")
    flat, lines = flatten(matrices)

    print("\nPer-cell-line PCA:")
    embedding, loadings = per_line_pca(flat, lines, args.n_components)

    emb_path = os.path.join(args.output_dir, "vision_scores_line_pca.csv")
    load_path = os.path.join(args.output_dir, "vision_scores_line_pca_loadings.csv")
    embedding.to_csv(emb_path)
    loadings.to_csv(load_path)

    print(f"\nembedding {embedding.shape[0]} drugs x {embedding.shape[1]} columns")
    print(f"wrote {emb_path}\n      {load_path}")


if __name__ == "__main__":
    main()
