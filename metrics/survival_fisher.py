"""Survival as a tested log-odds ratio, via Fisher's exact test.

A complement to survival_log2fc.py. Same question -- did this cell line lose
representation under this drug? -- but each condition gets a formal test against
the plate's pooled DMSO wells rather than a fold change.

For each (plate, cell line, drug, dose), the 2x2 table is

                        this cell line    all other lines
    drug well                 a                  c
    pooled DMSO wells         b                  d

so `a` is the line's count in the drug well, `c` the rest of that well, and
`b`/`d` the same split pooled over every DMSO_TF well on that plate.

Fisher's exact test is run one-sided (`alternative="less"`), testing
specifically for **depletion** of the line under drug. The effect size is a
log2 odds ratio with a Haldane-Anscombe 0.5 correction:

    log2( ((a + 0.5)(d + 0.5)) / ((b + 0.5)(c + 0.5)) )

P-values are FDR-corrected (Benjamini-Hochberg) across all conditions.

Outputs:
    fisher_results_log_odds_ratio.csv    one row per condition
    effect_size_matrix_odds_ratio.csv    drug-doses x cell lines, log2 odds
    significance_matrix_neg_log10p.csv   drug-doses x cell lines, -log10 FDR

These feed figures rather than a layer of the metrics AnnData; the `sensitivity`
layer comes from survival_log2fc.py.

Usage:
    python survival_fisher.py --counts plate_line_drug_concentration_count.csv \\
                              --output_dir .
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_names import drug_dose_key  # noqa: E402
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# --- paths: edit for your system ---
COUNTS = "plate_line_drug_concentration_count.csv"
OUTPUT_DIR = "."

CONTROL = "DMSO_TF"


def load_counts(path: str) -> pd.DataFrame:
    """Read the count table, collapsing cell cycle phases if they are present."""
    df = pd.read_csv(path)
    if "cell_cycle_group" in df.columns:
        keys = ["plate", "line", "drug_name", "concentration", "unit"]
        df = df.groupby(keys, observed=True)["count"].sum().reset_index()
    return df[df["count"] > 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=COUNTS)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    counts = load_counts(args.counts)
    control_raw = counts[counts["drug_name"] == CONTROL]
    drug = counts[counts["drug_name"] != CONTROL].copy()

    # Pooled control arm: every DMSO well on the plate, taken together.
    per_dmso_well = (control_raw.groupby(["plate", "drug_name", "concentration"])
                     ["count"].sum().reset_index())
    pooled_total = (per_dmso_well.groupby("plate")["count"].sum()
                    .rename("pooled_d_plus_b").reset_index())
    pooled_line = (control_raw.groupby(["plate", "line"])["count"].sum()
                   .rename("pooled_b").reset_index())
    control = pooled_line.merge(pooled_total, on="plate").set_index(["plate", "line"])

    # Each drug well's total, so `c` is everything in the well that is not this line.
    per_drug_well = (drug.groupby(["plate", "drug_name", "concentration"])["count"].sum()
                     .rename("total_well_count_a_plus_c").reset_index())
    drug = drug.merge(per_drug_well, on=["plate", "drug_name", "concentration"])

    rows = []
    for _, row in tqdm(drug.iterrows(), total=len(drug), desc="Fisher tests"):
        key = (row["plate"], row["line"])
        if key not in control.index:
            continue
        a = row["count"]
        c = row["total_well_count_a_plus_c"] - a
        b = control.loc[key, "pooled_b"]
        d = control.loc[key, "pooled_d_plus_b"] - b
        if b <= 0:
            continue

        # One-sided: we are asking whether the line is depleted under drug.
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="less")
        log_odds_ratio = np.log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))

        rows.append({
            "plate": row["plate"], "line": row["line"],
            "drug_name": row["drug_name"], "concentration": row["concentration"],
            "log_odds_ratio": log_odds_ratio, "p_value": p_value,
            "count_in_drug_well (a)": a, "pooled_count_in_control (b)": b,
        })

    results = pd.DataFrame(rows).dropna(subset=["p_value"])
    reject, p_adj, _, _ = multipletests(results["p_value"], alpha=0.05, method="fdr_bh")
    results["p_adj_fdr"] = p_adj
    results["significant_fdr_0.05"] = reject
    results = results.sort_values(by=["p_adj_fdr", "log_odds_ratio"], ascending=[True, True])

    results.to_csv(os.path.join(args.output_dir, "fisher_results_log_odds_ratio.csv"),
                   index=False)

    results["drug__concentration"] = [
        drug_dose_key(d, c) for d, c in
        zip(results["drug_name"], results["concentration"])]
    min_nonzero = results.loc[results["p_adj_fdr"] > 0, "p_adj_fdr"].min()
    results["neg_log10_p"] = -np.log10(results["p_adj_fdr"].replace(0, min_nonzero))

    # aggfunc="mean" averages a condition measured on more than one plate.
    significance = results.pivot_table(index="drug__concentration", columns="line",
                                       values="neg_log10_p", aggfunc="mean")
    effect_size = results.pivot_table(index="drug__concentration", columns="line",
                                      values="log_odds_ratio", aggfunc="mean")
    significance.to_csv(os.path.join(args.output_dir, "significance_matrix_neg_log10p.csv"))
    effect_size.to_csv(os.path.join(args.output_dir, "effect_size_matrix_odds_ratio.csv"))

    print(f"\n{len(results)} tested conditions")
    print(f"matrices: {effect_size.shape[0]} drug-doses x {effect_size.shape[1]} cell lines")
    print(f"significant at FDR 0.05: {int(results['significant_fdr_0.05'].sum())}")


if __name__ == "__main__":
    main()
