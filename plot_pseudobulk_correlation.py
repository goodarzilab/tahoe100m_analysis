#!/usr/bin/env python3
"""Aggregate the per-cell-line cell-number sweep and plot the saturation curve:
signature replicate concordance (plate6 vs plate14) vs cells sampled per condition.

Outputs (in the data dir):
    pseudobulk_correlation_curve.{pdf,png}
    pseudobulk_correlation_summary.csv
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# --- paths: edit for your system ---
# Directory of per-cell-line parquets written by pseudobulk_correlation.py.
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pseudobulk_correlation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    files = [f for f in files if not os.path.basename(f).startswith("cellnumber")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{len(files)} cell lines, {df['drug'].nunique()} drugs, {len(df)} rows")

    # Drug-signature (log2FC, HVG) concordance, distribution across all
    # (cell line, drug) conditions at each depth N.
    grid = sorted(df["N"].unique())
    data = [df.loc[df["N"] == N, "corr_mean"].dropna().to_numpy() for N in grid]

    summ = (df.groupby("N")["corr_mean"]
            .agg(signature_concordance_mean="mean", median="median",
                 q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                 std="std", n_conditions="count").reset_index())
    out_csv = os.path.join(args.data_dir, "pseudobulk_correlation_summary.csv")
    summ.to_csv(out_csv, index=False)
    print(summ.round(3).to_string(index=False))

    GRAY, BLUE = "#b3b3b3", "#2c6fb0"
    pos = np.arange(len(grid))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.yaxis.grid(True, color="#ededed", linewidth=0.7); ax.set_axisbelow(True)

    # gray violins
    parts = ax.violinplot(data, positions=pos, widths=0.82, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(GRAY); pc.set_edgecolor("#8a8a8a")
        pc.set_alpha(0.55); pc.set_linewidth(0.8)
    # blue IQR box (q25-q75) with white median line
    q1 = [np.percentile(d, 25) for d in data]
    q3 = [np.percentile(d, 75) for d in data]
    med = [float(np.median(d)) for d in data]
    boxw = 0.26
    for xi, lo, hi, m in zip(pos, q1, q3, med):
        ax.add_patch(plt.Rectangle((xi - boxw / 2, lo), boxw, hi - lo,
                                   facecolor=BLUE, alpha=0.92, edgecolor="none", zorder=4))
        ax.hlines(m, xi - boxw / 2, xi + boxw / 2, color="white", linewidth=1.6, zorder=6)

    ax.set_xticks(pos); ax.set_xticklabels([int(n) for n in grid])
    ax.set_xlabel("perturbed cells sampled per condition (per plate; DMSO fixed)")
    ax.set_ylabel("perturbation-signature concordance\nlog2FC, HVGs (Pearson r, plate6 vs plate14)")
    ax.set_title("Replicate reproducibility of the perturbation signature vs perturbed-cell number",
                 fontsize=11, color="#333333", pad=10)
    ax.set_ylim(-0.35, 1.0)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8, linestyle=":")
    for xi, d in zip(pos, data):
        ax.annotate(f"n={len(d)}", (float(xi), -0.31), ha="center", fontsize=6.5, color="#999999")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.data_dir, f"pseudobulk_correlation_curve.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"\nWrote violin curve + summary to {args.data_dir}")


if __name__ == "__main__":
    main()
