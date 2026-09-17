"""G arrest: does a drug push cells out of cycle, relative to DMSO?

From the cell counts produced by count_cells.py, for each
(plate, cell line, drug, dose):

    G_arrest_proportion = (G1 + G2M) / (G1 + G2M + S)

is the fraction of surviving cells sitting in a gap phase rather than
replicating. Each condition is compared against the mean DMSO_TF proportion for
the same cell line on the same plate:

    G_arrest_log2FC = log2( (p_drug + eps) / (p_DMSO + eps) )

then averaged across plates and pivoted to drug-doses x cell lines.

The values are log2 fold changes of a proportion, centred near zero; positive means more of the surviving population is in G1/G2M than in
DMSO, i.e. arrest.

Conditions whose G arrest proportion is exactly zero are dropped before the
ratio is taken.

Usage:
    python g_arrest.py --counts plate_line_drug_concentration_cellcycle_count.csv \\
                       --output G_arrest_proportions.csv
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_names import drug_dose_key  # noqa: E402

# --- paths: edit for your system ---
COUNTS = "plate_line_drug_concentration_cellcycle_count.csv"
OUTPUT = "G_arrest_proportions.csv"

# Guards log2 against a zero denominator; small enough not to move any real value.
EPSILON = 1e-9
GAP_PHASES = ["G1", "G2M"]
CYCLE_PHASES = ["G1", "G2M", "S"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=COUNTS,
                    help="phase-resolved counts CSV from count_cells.py")
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.counts)
    # Older count files carried the input filename suffix in the plate label.
    df["plate"] = df["plate"].str.replace("_filtered.gz", "", regex=True)

    counts = df.pivot_table(
        index=["plate", "line", "drug_name", "concentration"],
        columns="cell_cycle_group", values="count", fill_value=0,
    ).reset_index()
    counts.columns.name = None

    phases = [p for p in CYCLE_PHASES + ["Other"] if p in counts.columns]
    gap = [p for p in GAP_PHASES if p in counts.columns]
    counts["Total_count"] = counts[phases].sum(axis=1)
    counts["G_arrest_count"] = counts[gap].sum(axis=1) if gap else 0
    counts["G_arrest_proportion"] = np.where(
        counts["Total_count"] > 0,
        counts["G_arrest_count"] / counts["Total_count"].replace(0, np.nan),
        0.0,
    )

    # Per-plate, per-line DMSO reference.
    dmso = counts[(counts["drug_name"] == "DMSO_TF") & (counts["concentration"] == 0)]
    reference = (dmso.groupby(["plate", "line"])["G_arrest_proportion"]
                     .mean()
                     .reset_index()
                     .rename(columns={"G_arrest_proportion": "Avg_DMSO_G_arrest_proportion"}))
    counts = counts.merge(reference, on=["plate", "line"], how="left")

    counts = counts[counts["G_arrest_proportion"] != 0]

    counts["G_arrest_log2FC_vs_DMSO"] = np.log2(
        (counts["G_arrest_proportion"] + EPSILON)
        / (counts["Avg_DMSO_G_arrest_proportion"] + EPSILON))

    per_condition = (counts.groupby(["line", "drug_name", "concentration"])
                           ["G_arrest_log2FC_vs_DMSO"]
                           .mean()
                           .reset_index())
    per_condition["drug_concentration_index"] = [
        drug_dose_key(d, c) for d, c in
        zip(per_condition["drug_name"], per_condition["concentration"])]

    matrix = per_condition.pivot_table(index="drug_concentration_index",
                                       columns="line",
                                       values="G_arrest_log2FC_vs_DMSO")
    matrix.to_csv(args.output, index=True)

    print(f"{matrix.shape[0]} drug-doses x {matrix.shape[1]} cell lines, "
          f"{int(matrix.isna().sum().sum())} missing")
    print(f"range {matrix.min().min():.3f} to {matrix.max().max():.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
