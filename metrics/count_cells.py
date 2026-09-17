"""Tally surviving cells per plate x cell line x drug x dose x cell cycle phase.

This is the input to both downstream count-based metrics: survival (how many
cells of a line remain after treatment, relative to DMSO) and G arrest (how the
surviving cells redistribute across cell cycle phases).

Only `pass_filter == "full"` cells are counted. Reads `.obs` only -- the plate
files are opened in backed mode so the expression matrix never enters memory.

Output: one CSV with columns
    plate, line, drug_name, concentration, unit, cell_cycle_group, count

Summing `count` over `cell_cycle_group` gives the plain per-condition totals
used by the survival scripts.

Every cell in the Tahoe plates carries a phase of G1, G2M or S; the `Other`
bucket exists for robustness but is empty in practice.

Usage:
    python count_cells.py --plate_dir <dir> --output plate_line_drug_concentration_cellcycle_count.csv
"""

import os
import re
import gc
import glob
import argparse

import pandas as pd
import scanpy as sc
from tqdm import tqdm

# --- paths: edit for your system ---
PLATE_DIR = "/processed_datasets/scRecount/tahoe/original_h5ad/"
OUTPUT = "plate_line_drug_concentration_cellcycle_count.csv"

ALLOWED_PHASES = ["G1", "G2M", "S"]
# "[('DrugA', 1.0, 'uM')]" -> DrugA, 1.0, uM
CONDITION_RE = r"\[\('(.+?)',\s*([\d.]+),\s*'(\w+)'\)\]"


def plate_label(filename: str) -> str:
    """plate7_filtered.h5ad.gz -> plate7."""
    m = re.match(r"(plate\d+)", os.path.basename(filename))
    return m.group(1) if m else os.path.basename(filename).split(".")[0]


def count_plate(path: str) -> pd.DataFrame | None:
    """Tally one plate's full-pass cells. Returns None if there are none."""
    adata = sc.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        obs = obs[obs["pass_filter"] == "full"].copy()
        if obs.empty:
            print(f"  no full-pass cells in {os.path.basename(path)}, skipping")
            return None

        obs[["drug_name", "concentration", "unit"]] = (
            obs["drugname_drugconc"].str.extract(CONDITION_RE))
        obs["concentration"] = pd.to_numeric(obs["concentration"], errors="coerce")
        obs["cell_cycle_group"] = obs["phase"].apply(
            lambda x: x if x in ALLOWED_PHASES else "Other")

        tally = (obs.groupby(["cell_name", "drug_name", "concentration", "unit",
                              "cell_cycle_group"], observed=True)
                    .size()
                    .reset_index(name="count"))
        tally["plate"] = plate_label(path)
        return tally[["plate", "cell_name", "drug_name", "concentration",
                      "unit", "cell_cycle_group", "count"]]
    finally:
        # Backed mode holds the file open; release it before the next plate.
        if adata.isbacked:
            adata.file.close()
        del adata
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plate_dir", default=PLATE_DIR)
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.plate_dir, "*.h5ad.gz")))
    if not paths:
        raise SystemExit(f"no plate files found in {args.plate_dir}")
    print(f"Found {len(paths)} plate files")

    tallies = []
    for path in tqdm(paths, desc="Processing plates"):
        try:
            t = count_plate(path)
            if t is not None:
                tallies.append(t)
        except Exception as e:  # one bad plate must not lose the rest
            print(f"Error processing {path}: {e}")

    if not tallies:
        raise SystemExit("no counts produced")

    df = pd.concat(tallies, ignore_index=True).rename(columns={"cell_name": "line"})
    df.to_csv(args.output, index=False)
    print(f"\n{len(df)} rows across {df['plate'].nunique()} plates, "
          f"{df['line'].nunique()} cell lines, {df['drug_name'].nunique()} drugs")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
