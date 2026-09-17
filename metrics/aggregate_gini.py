"""Combine the per-plate Gini results into the cell line x drug-dose matrix.

Reads the CSVs written by gini.py (one per plate) and, for each plate:

  1. drops conditions whose clustering was too coarse to be informative --
     fewer than --min_clusters clusters, or fewer than --min_cells_per_cluster
     cells per cluster on average
  2. averages gini_drug_zscore and gini_ctrl_zscore over the surviving
     clustering resolutions for each (cell line, drug-dose)

then averages across plates and pivots to cell lines x drug-doses.

Outputs:
    gini_zscore_long.csv    one row per (cell line, drug-dose)
    gini_zscore_matrix.csv  drug-doses x cell lines; transposed, this is the
                            `gini` layer of the metrics AnnData

The values are a mean over resolutions 1.0/2.0/3.0, not a single resolution.

Usage:
    python aggregate_gini.py --gini_dir results/gini --output_dir results/gini/aggregated
"""

import os
import sys
import glob
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_names import parse_condition  # noqa: E402

# --- paths: edit for your system ---
# Per-plate outputs of gini.py. For the paper these lived at
# /large_storage/ctc/public/tahoe/gini/
GINI_DIR = "results/gini"
OUTPUT_DIR = "results/gini/aggregated"

# Conditions below these thresholds are dropped: the Gini of a cluster
# occupancy vector is meaningless when there are barely any clusters, or when
# clusters hold a handful of cells each.
MIN_CELLS_PER_CLUSTER = 10
MIN_CLUSTERS = 10


def parse_drug_dose(perturbation: pd.Series) -> pd.DataFrame:
    """Split condition strings into canonical drugname and dose columns.

    Parsed with ast.literal_eval via drug_names.parse_condition, so drug names
    containing commas survive intact.
    """
    parsed = {c: parse_condition(c) for c in perturbation.unique()}
    return pd.DataFrame({
        "drugname": perturbation.map(lambda c: parsed[c][0]),
        "dose": perturbation.map(lambda c: parsed[c][1]),
    }, index=perturbation.index)


def load_plate(path: str, min_cells_per_cluster: int, min_clusters: int) -> pd.DataFrame:
    """Filter one plate's results and average over clustering resolutions."""
    df = pd.read_csv(path)
    df["avg_cells_per_cluster"] = df["total_cells"] / df["n_clusters"]

    n_before = len(df)
    df = df[(df["avg_cells_per_cluster"] >= min_cells_per_cluster)
            & (df["n_clusters"] >= min_clusters)]
    plate = os.path.basename(path).split("_")[0]
    print(f"  {plate}: {n_before} rows -> {len(df)} after filtering")

    return (df.groupby(["cell_line", "perturbation"])
              [["gini_drug_zscore", "gini_ctrl_zscore"]]
              .mean()
              .reset_index()
              .assign(plate=plate))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gini_dir", default=GINI_DIR,
                    help="directory of per-plate CSVs written by gini.py")
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--min_cells_per_cluster", type=int, default=MIN_CELLS_PER_CLUSTER)
    ap.add_argument("--min_clusters", type=int, default=MIN_CLUSTERS)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.gini_dir, "*heterogeneity_analysis.csv")))
    if not paths:
        raise SystemExit(f"no per-plate gini CSVs found in {args.gini_dir}")
    print(f"Found {len(paths)} plate result files")

    per_plate = [load_plate(p, args.min_cells_per_cluster, args.min_clusters)
                 for p in paths]
    gini = pd.concat(per_plate, ignore_index=True)
    gini = pd.concat([gini, parse_drug_dose(gini["perturbation"])], axis=1)

    # Average the per-plate values for conditions measured on more than one plate.
    long = (gini.groupby(["cell_line", "perturbation", "drugname", "dose"])
                ["gini_drug_zscore"]
                .mean()
                .reset_index()
                .rename(columns={"perturbation": "drugname_drugconc"}))

    matrix = long.pivot(index=["drugname_drugconc", "drugname", "dose"],
                        columns="cell_line", values="gini_drug_zscore")

    os.makedirs(args.output_dir, exist_ok=True)
    long_path = os.path.join(args.output_dir, "gini_zscore_long.csv")
    matrix_path = os.path.join(args.output_dir, "gini_zscore_matrix.csv")
    long.to_csv(long_path, index=False)
    matrix.to_csv(matrix_path)

    print(f"\n{len(long)} (cell line, drug-dose) values across "
          f"{long['cell_line'].nunique()} cell lines and "
          f"{long['drugname_drugconc'].nunique()} drug-doses")
    print(f"matrix {matrix.shape[0]} drug-doses x {matrix.shape[1]} cell lines, "
          f"{matrix.isna().sum().sum()} missing")
    print(f"wrote {long_path}\n      {matrix_path}")


if __name__ == "__main__":
    main()
