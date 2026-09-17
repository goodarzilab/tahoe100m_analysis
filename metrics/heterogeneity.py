"""Within-population transcriptional heterogeneity, per (cell line, drug, dose).

How spread out is a drug-treated population in expression space, relative to
its DMSO control? For each drug, subset to that drug's cells plus the DMSO_TF
control cells of the same cell line and plate, then

    normalize_total -> log1p -> PCA
    X = X_pca * uns["pca"]["variance"]        # weight PCs by their variance
    distance = mean ||x - centroid|| within each group

reported separately for the drug and the control as `drug_within_distance` and
`dmso_within_distance`. The `heterogeneity` layer is the ratio of the two -- see heterogeneity_ratio.R and aggregate_heterogeneity.py.


Usage (one plate per invocation, parallelised over cell lines):
    python heterogeneity.py 6 --output_dir <dir> --n_workers 8
"""

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # inputs may hold a lock
# Pin BLAS to 1 thread per process BEFORE numpy is imported: we parallelize over
# cell lines with a process pool, so each worker must stay single-threaded or the
# threads oversubscribe the allocated CPUs and thrash. (set before numpy import)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import re
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scanpy as sc

DMSO = "[('DMSO_TF', 0.0, 'uM')]"
DATA = "/processed_datasets/scRecount/tahoe/original_h5ad/plate{n}_filtered.h5ad.gz"


def _safe(name):
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))


def _within_distance(adata_sub, drug):
    """drug/DMSO within-population dispersion in variance-weighted PCA space."""
    sc.pp.normalize_total(adata_sub)
    sc.pp.log1p(adata_sub)
    sc.pp.pca(adata_sub)
    X = adata_sub.obsm["X_pca"] * adata_sub.uns["pca"]["variance"]
    md = (adata_sub.obs["drugname_drugconc"] == drug).values
    mc = (adata_sub.obs["drugname_drugconc"] == DMSO).values
    dist_drug = np.linalg.norm(X[md] - X[md].mean(axis=0), axis=1)
    dist_dmso = np.linalg.norm(X[mc] - X[mc].mean(axis=0), axis=1)
    return float(np.mean(dist_drug)), float(np.mean(dist_dmso))


def process_cell_line(task):
    plate_n, cell_line, out_dir = task
    out_path = os.path.join(
        out_dir, f"{_safe(cell_line)}_plate{plate_n}_heterogeneity.parquet")
    if os.path.exists(out_path):
        return (cell_line, "skip (exists)")
    try:
        a = sc.read_h5ad(DATA.format(n=plate_n), backed="r")
        sub = a[a.obs["cell_name"] == cell_line].to_memory()
        a.file.close()
        sub = sub[sub.obs["pass_filter"] == "full"]
        drugs = [d for d in sub.obs["drugname_drugconc"].unique() if d != DMSO]
        rows = []
        for drug in drugs:
            try:
                ad2 = sub[sub.obs["drugname_drugconc"].isin([drug, DMSO])].copy()
                dd, dc = _within_distance(ad2, drug)
                rows.append({
                    "drugname_drugconc": drug,
                    "drug_within_distance": dd,
                    "dmso_within_distance": dc,
                    "plate": f"plate{plate_n}",
                    "cell_line": cell_line,
                })
            except Exception as e:  # noqa: BLE001 - one bad drug must not kill the line
                print(f"  drug error {cell_line}/{drug}: {e}", flush=True)
        pd.DataFrame(rows).to_parquet(out_path)
        return (cell_line, f"ok (n_drugs={len(rows)})")
    except Exception as e:  # noqa: BLE001 - one bad cell line must not kill the plate
        traceback.print_exc()
        return (cell_line, f"FAIL {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plate_num", type=int)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    a = sc.read_h5ad(DATA.format(n=args.plate_num), backed="r")
    cell_lines = sorted(a.obs["cell_name"].astype(str).unique())
    a.file.close()
    print(f"plate{args.plate_num}: {len(cell_lines)} cell lines -> {args.output_dir} "
          f"({args.n_workers} workers)", flush=True)

    tasks = [(args.plate_num, cl, args.output_dir) for cl in cell_lines]
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        for cl, status in ex.map(process_cell_line, tasks):
            print(f"  {cl}: {status}", flush=True)
    print(f"Completed plate{args.plate_num}", flush=True)


if __name__ == "__main__":
    main()
