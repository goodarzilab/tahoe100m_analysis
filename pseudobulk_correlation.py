"""Replicate reproducibility of the drug expression signature as a function of
cells sampled per condition (to guide future experimental design).

For each shared (cell line, drug-dose) condition on plates 6 and 14, we subsample
N perturbed (drug) cells per plate while holding the DMSO control FIXED (pseudobulk
over all control cells per plate), build the signature LFC = mean(drug) - mean(DMSO)
over HVGs, and correlate the plate-6 and plate-14 signatures (Pearson across genes).
Sweeping N gives a saturation curve isolating the dependence on the number of
*perturbed* cells (the experimentally controllable knob), with control-arm noise
removed by fixing DMSO.

Why the signature (LFC vs DMSO) and not the raw pseudobulk: raw pseudobulk is
dominated by the cell-line baseline and correlates ~0.98 at any depth, so it is
uninformative; the drug-specific signal lives in the deviation from control.

Parallelized over cell lines; each worker reads its cell line from BOTH plates.
Outputs one parquet per cell line into <out_dir>; aggregate/plot with
plot_pseudobulk_correlation.py.

Usage:
    python pseudobulk_correlation.py --output_dir <dir> --n_workers 8
"""

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import re
import zlib
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

# --- paths: edit for your system ---
DATA = "/processed_datasets/scRecount/tahoe/original_h5ad/plate{n}_filtered.h5ad.gz"

DMSO = "[('DMSO_TF', 0.0, 'uM')]"
PLATES = (6, 14)
GRID = [25, 50, 100, 200, 400, 800, 1600]   # N = perturbed (drug) cells per plate
N_REPEATS = 10
N_HVG = 2000
MIN_DMSO = 50   # minimum DMSO control cells per plate for a stable reference


def _safe(name):
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))


def _load_cell_line(plate_n, cell_line):
    a = sc.read_h5ad(DATA.format(n=plate_n), backed="r")
    sub = a[a.obs["cell_name"] == cell_line].to_memory()
    a.file.close()
    sub = sub[sub.obs["pass_filter"] == "full"].copy()
    sc.pp.normalize_total(sub)
    sc.pp.log1p(sub)
    return sub


def _dense_hvg(sub, genes):
    """Return cells x len(genes) dense array (HVG-subset, log-normalized)."""
    m = sub[:, genes].X
    return np.asarray(m.todense()) if sp.issparse(m) else np.asarray(m)


def process_cell_line(task):
    cell_line, out_dir = task
    out_path = os.path.join(out_dir, f"{_safe(cell_line)}.parquet")
    if os.path.exists(out_path):
        return (cell_line, "skip (exists)")
    try:
        s6 = _load_cell_line(6, cell_line)
        s14 = _load_cell_line(14, cell_line)

        # Align both plates to the common gene set.
        genes_common = s6.var_names.intersection(s14.var_names)
        s6 = s6[:, genes_common].copy()
        s14 = s14[:, genes_common].copy()

        # HVGs from plate 6 (gene *selection* only; applied to both plates).
        sc.pp.highly_variable_genes(s6, n_top_genes=N_HVG, flavor="seurat")
        hvg_mask = s6.var["highly_variable"].to_numpy()
        hvg_names = s6.var_names[hvg_mask].tolist()

        Xhvg6 = _dense_hvg(s6, hvg_names)
        Xhvg14 = _dense_hvg(s14, hvg_names)
        d6 = s6.obs["drugname_drugconc"].astype(str).to_numpy()
        d14 = s14.obs["drugname_drugconc"].astype(str).to_numpy()
        ctrl6 = np.where(d6 == DMSO)[0]
        ctrl14 = np.where(d14 == DMSO)[0]
        if len(ctrl6) < MIN_DMSO or len(ctrl14) < MIN_DMSO:
            return (cell_line, f"skip (DMSO < {MIN_DMSO})")

        # Stable control reference: pseudobulk over ALL DMSO cells (per plate),
        # held fixed while we vary only the number of perturbed cells. This
        # isolates the drug-cell-count dependence from control-arm noise.
        dmso6 = Xhvg6[ctrl6].mean(0)
        dmso14 = Xhvg14[ctrl14].mean(0)
        rng = np.random.default_rng(zlib.crc32(cell_line.encode()) & 0xFFFFFFFF)

        drugs = (set(np.unique(d6)) & set(np.unique(d14))) - {DMSO}
        rows = []
        for drug in sorted(drugs):
            i6 = np.where(d6 == drug)[0]
            i14 = np.where(d14 == drug)[0]
            cap = min(len(i6), len(i14))
            for N in GRID:                       # N = perturbed cells per plate
                if cap < N:
                    continue
                sig = []
                for _ in range(N_REPEATS):
                    lfc6 = Xhvg6[rng.choice(i6, N, replace=False)].mean(0) - dmso6
                    lfc14 = Xhvg14[rng.choice(i14, N, replace=False)].mean(0) - dmso14
                    if lfc6.std() > 0 and lfc14.std() > 0:
                        sig.append(np.corrcoef(lfc6, lfc14)[0, 1])
                if sig:
                    rows.append({
                        "cell_line": cell_line, "drug": drug, "N": N,
                        "corr_mean": float(np.mean(sig)),
                        "corr_sd": float(np.std(sig)),
                        "n_drug_p6": int(len(i6)), "n_drug_p14": int(len(i14)),
                        "n_dmso_p6": int(len(ctrl6)), "n_dmso_p14": int(len(ctrl14)),
                    })
        pd.DataFrame(rows).to_parquet(out_path)
        return (cell_line, f"ok (rows={len(rows)})")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return (cell_line, f"FAIL {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    a = sc.read_h5ad(DATA.format(n=PLATES[0]), backed="r")
    a14 = sc.read_h5ad(DATA.format(n=PLATES[1]), backed="r")
    shared = sorted(set(a.obs["cell_name"].astype(str).unique())
                    & set(a14.obs["cell_name"].astype(str).unique()))
    a.file.close(); a14.file.close()
    print(f"{len(shared)} shared cell lines; N grid={GRID}; {N_REPEATS} repeats "
          f"-> {args.output_dir} ({args.n_workers} workers)", flush=True)

    tasks = [(cl, args.output_dir) for cl in shared]
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        for cl, status in ex.map(process_cell_line, tasks):
            print(f"  {cl}: {status}", flush=True)
    print("Completed cell-number reproducibility sweep", flush=True)


if __name__ == "__main__":
    main()
