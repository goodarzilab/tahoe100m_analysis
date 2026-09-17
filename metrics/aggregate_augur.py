"""Combine the per-plate Augur results into the cell line x drug-dose matrix.

Reads the `summary_metrics__<drug>_<dose>.csv` files written by augur.py. Each
holds Augur's cross-validation metrics with one column per cell line, so a
column mean gives that cell line's perturbation score for that drug-dose. Those
rows are assembled into a per-plate drug-dose x cell line matrix, then averaged
across plates.

Outputs:
    <plate>__augur_matrix.csv   one per plate, drug-doses x cell lines
    augur_matrix.csv            averaged across plates; transposed, this is the
                                `augur` layer of the metrics AnnData

Usage:
    python aggregate_augur.py --augur_dir results/augur --output_dir results/augur
"""

import os
import glob
import argparse

import pandas as pd

# --- paths: edit for your system ---
# Directory holding the per-plate subdirectories written by augur.py.
AUGUR_DIR = "results/augur"
OUTPUT_DIR = "results/augur"


def load_plate(plate_dir: str) -> pd.DataFrame:
    """Build one plate's drug-dose x cell line matrix of Augur scores."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(plate_dir, "summary_metrics__*.csv"))):
        df = pd.read_csv(path)
        # Filename is summary_metrics__<drug>_<dose>.csv; the part after the
        # double underscore is the drug-dose key.
        key = os.path.basename(path).split("__", 1)[1].removesuffix(".csv")
        # Cell lines are the numeric columns; the rest are metric labels.
        numeric = df.select_dtypes(include="number")
        rows[key] = numeric.mean(axis=0)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--augur_dir", default=AUGUR_DIR,
                    help="directory containing plate<N>/ subdirectories")
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    plate_dirs = sorted(d for d in glob.glob(os.path.join(args.augur_dir, "plate*"))
                        if os.path.isdir(d))
    if not plate_dirs:
        raise SystemExit(f"no plate*/ directories found in {args.augur_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    matrices = {}
    for d in plate_dirs:
        plate = os.path.basename(d)
        m = load_plate(d)
        if m.empty:
            print(f"  {plate}: no summary_metrics files, skipped")
            continue
        matrices[plate] = m
        out = os.path.join(args.output_dir, f"{plate}__augur_matrix.csv")
        m.to_csv(out)
        print(f"  {plate}: {m.shape[0]} drug-doses x {m.shape[1]} cell lines -> {out}")

    if not matrices:
        raise SystemExit("no Augur results found")

    # Mean across plates, aligning on the union of drug-doses and cell lines.
    combined = pd.concat(matrices.values())
    mean = combined.groupby(level=0).mean().sort_index()

    out = os.path.join(args.output_dir, "augur_matrix.csv")
    mean.to_csv(out)
    print(f"\ncombined over {len(matrices)} plates: "
          f"{mean.shape[0]} drug-doses x {mean.shape[1]} cell lines, "
          f"{mean.isna().sum().sum()} missing")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
