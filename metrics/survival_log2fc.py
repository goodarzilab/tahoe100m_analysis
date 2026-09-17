"""Survival as a log2 fold change in each cell line's share of a drug well.

Tahoe pools many cell lines into one well, so a cell line's *share* of the
well is the readout: a line that is killed by the drug loses share, one that
tolerates it gains share. Absolute counts cannot be used directly because wells
differ in how many cells were recovered overall.

Per plate:

  1. pivot to drug-dose x cell line counts
  2. normalise each drug row to sum to 1, giving each cell line's share of that
     well
  3. average the shares across that plate's DMSO_TF wells to get the untreated
     baseline share per cell line
  4. log2( (share_drug + 0.01) / (share_DMSO + 0.01) )

then average across plates and pivot to drug-doses x cell lines. The pseudocount
of 0.01 keeps the ratio finite for lines wiped out by a drug.

Outputs:
    log2fc_survivals_averaged_across_plate.csv        all doses; this is the
                                                      `sensitivity` layer
    log2fc_survivals_averaged_across_plate_<dose>.csv one per dose

Negative means the line lost share relative to DMSO.

Usage:
    python survival_log2fc.py --counts plate_line_drug_concentration_count.csv \\
                              --output_dir .
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_names import drug_dose_key  # noqa: E402

# --- paths: edit for your system ---
COUNTS = "plate_line_drug_concentration_count.csv"
OUTPUT_DIR = "."

CONTROL_PREFIX = drug_dose_key("DMSO_TF", 0.0)
PSEUDOCOUNT = 0.01
DOSES = ["0.05", "0.5", "5.0"]


def load_counts(path: str) -> pd.DataFrame:
    """Read the count table, collapsing cell cycle phases if they are present."""
    df = pd.read_csv(path)
    if "cell_cycle_group" in df.columns:
        # count_cells.py emits phase-resolved counts; survival wants the totals.
        keys = ["plate", "line", "drug_name", "concentration", "unit"]
        df = df.groupby(keys, observed=True)["count"].sum().reset_index()

    df = df[df["plate"] != "test"]
    df = df[df["count"] != 0]
    return pd.DataFrame({
        "plate": df["plate"],
        "cell_line": df["line"],
        "drug": df["drug_name"],
        "dose": df["concentration"].astype(float),
        "n_cells": df["count"].astype(float),
    })


def plate_log2fc(plate_df: pd.DataFrame) -> pd.DataFrame | None:
    """Share-normalised log2 fold change vs the plate's DMSO wells."""
    plate_df = plate_df.copy()
    plate_df["drug_dose"] = [drug_dose_key(d, x) for d, x
                             in zip(plate_df["drug"], plate_df["dose"])]
    counts = plate_df.pivot(index="drug_dose", columns="cell_line", values="n_cells")

    # Each drug row becomes the composition of cell lines in that well.
    shares = counts.divide(counts.sum(axis=1), axis=0)

    dmso_rows = [i for i in shares.index if CONTROL_PREFIX in i]
    if not dmso_rows:
        return None
    baseline = shares.loc[dmso_rows].mean()

    drugs = shares.drop(dmso_rows)
    return np.log2((drugs + PSEUDOCOUNT) / (baseline + PSEUDOCOUNT))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=COUNTS,
                    help="count table; phase-resolved counts are collapsed automatically")
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    counts = load_counts(args.counts)

    per_plate = {}
    for plate, group in counts.groupby("plate"):
        result = plate_log2fc(group)
        if result is None:
            print(f"  {plate}: no DMSO wells, skipped")
            continue
        per_plate[plate] = result
    if not per_plate:
        raise SystemExit("no plates produced results")
    print(f"{len(per_plate)} plates")

    stacked = pd.concat(per_plate.values(), keys=per_plate.keys(),
                        names=["plate", "drug"])
    long = (stacked.reset_index()
                   .melt(id_vars=["plate", "drug"], var_name="line", value_name="log2fc"))
    matrix = (long.groupby(["drug", "line"])["log2fc"].mean()
                  .reset_index()
                  .pivot(index="drug", columns="line", values="log2fc"))

    out = os.path.join(args.output_dir, "log2fc_survivals_averaged_across_plate.csv")
    matrix.to_csv(out)
    print(f"{matrix.shape[0]} drug-doses x {matrix.shape[1]} cell lines, "
          f"{int(matrix.isna().sum().sum())} missing -> {out}")

    for dose in DOSES:
        subset = matrix[matrix.index.str.endswith(f"__{dose}")]
        path = os.path.join(args.output_dir,
                            f"log2fc_survivals_averaged_across_plate_{dose}.csv")
        subset.to_csv(path)
        print(f"  dose {dose}: {subset.shape[0]} drugs -> {path}")


if __name__ == "__main__":
    main()
