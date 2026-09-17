"""Pivot the heterogeneity ratios into the cell line x drug-dose matrix.

Reads the parquet written by heterogeneity_ratio.R and reshapes
`normalized_heterogeneity` into the matrix behind the `heterogeneity` layer of
the metrics AnnData.

About 20% of (cell line, drug, dose) conditions were assayed on more than one
plate, so the pivot needs a rule for collapsing replicates:

    --dedup mean    average the replicates  [default]
    --dedup first   keep the first occurrence in the parquet

Condition keys are parsed with ast.literal_eval (via drug_names), so drug names
containing commas survive intact.

Usage:
    python aggregate_heterogeneity.py \\
        --parquet /path/to/tahoe_heterogeneity.parquet \\
        --output heterogeneity_matrix.csv
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_names import condition_to_key  # noqa: E402

# --- paths: edit for your system ---
# Output of heterogeneity_ratio.R. For the paper this was
# /large_storage/ctc/public/tahoe/analysis/tahoe_heterogeneity.parquet
PARQUET = "results/heterogeneity/tahoe_heterogeneity.parquet"
OUTPUT = "results/heterogeneity/heterogeneity_matrix.csv"


def canonical_keys(conditions: pd.Series) -> pd.Series:
    """Map each distinct condition string to its canonical "<drug>__<dose>" key."""
    mapping = {c: condition_to_key(c) for c in conditions.unique()}
    return conditions.map(mapping)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--dedup", choices=["first", "mean"], default="mean",
                    help="how to collapse conditions measured on several plates")
    args = ap.parse_args()

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df["drugname_drugconc"] = canonical_keys(df["drugname_drugconc"])

    grouped = df.groupby(["cell_line", "drugname_drugconc"])["normalized_heterogeneity"]
    sizes = grouped.size()
    n_dup = int((sizes > 1).sum())
    print(f"{len(df)} rows -> {len(sizes)} (cell line, drug-dose) conditions")
    print(f"{n_dup} ({n_dup / len(sizes) * 100:.1f}%) measured on more than one plate; "
          f"collapsing with --dedup {args.dedup}")

    matrix = (grouped.first() if args.dedup == "first" else grouped.mean()).unstack()

    matrix.to_csv(args.output)
    print(f"matrix {matrix.shape[0]} cell lines x {matrix.shape[1]} drug-doses, "
          f"{int(matrix.isna().sum().sum())} missing")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
