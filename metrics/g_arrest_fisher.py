"""Cell cycle arrest as a log-odds ratio, tested with Fisher's exact test.

A complement to g_arrest.py: instead of a log2 ratio of proportions, each
condition gets a formal test against the plate's pooled DMSO control.

For each (plate, cell line, drug, dose) and each phase group of interest, build
the 2x2 table

                    in group-of-interest   not in group
    drug well                a                   c
    pooled DMSO              b                   d

where the DMSO arm pools every DMSO_TF well for that cell line on that plate.
Fisher's exact test (two-sided) gives the p-value, and the effect size is a
log odds ratio with a Haldane-Anscombe 0.5 correction:

    log( ((a + 0.5)(d + 0.5)) / ((b + 0.5)(c + 0.5)) )

The `*_logodds` layers use the **natural** log (`--log_base e`, the default);
`--log_base 2` gives log2 odds instead.

P-values are FDR-corrected (Benjamini-Hochberg) across all conditions within an
analysis.

Four analyses are run:
    G1_arrest          G1
    G2M_arrest         G2M
    S_arrest           S
    G_arrest_(G1+G2M)  G1 and G2M together

Outputs per analysis:
    cell_cycle_arrest_results_<name>.csv   one row per condition
    effect_size_matrix_<name>.csv          drug-doses x cell lines, log odds
    significance_matrix_<name>.csv         drug-doses x cell lines, -log10 FDR

The effect-size matrices for G1_arrest, G2M_arrest and G_arrest_(G1+G2M) become
the `G1_arrest_logodds`, `G2M_arrest_logodds` and `G_arrest_logodds` layers.

Usage:
    python g_arrest_fisher.py --counts plate_line_drug_concentration_cellcycle_count.csv \\
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
COUNTS = "plate_line_drug_concentration_cellcycle_count.csv"
OUTPUT_DIR = "."

CONTROL = "DMSO_TF"
ANALYSES = {
    "G1_arrest": ["G1"],
    "G2M_arrest": ["G2M"],
    "S_arrest": ["S"],
    "G_arrest_(G1+G2M)": ["G1", "G2M"],
}


def arrest_analysis(df: pd.DataFrame, groups_of_interest: list[str],
                    analysis_name: str, log_base: str = "e") -> pd.DataFrame:
    """Fisher's exact test for enrichment in `groups_of_interest` vs pooled DMSO."""
    print(f"\n--- {analysis_name}: {'+'.join(groups_of_interest)} ---")
    control_raw = df[df["drug_name"] == CONTROL]
    drug_raw = df[df["drug_name"] != CONTROL]

    # Pooled control arm: every DMSO well for that cell line on that plate.
    control_total = (control_raw.groupby(["plate", "line"])["count"].sum()
                     .rename("pooled_d_plus_b"))
    control_goi = (control_raw[control_raw["cell_cycle_group"].isin(groups_of_interest)]
                   .groupby(["plate", "line"])["count"].sum()
                   .rename("pooled_b"))
    control = pd.concat([control_total, control_goi], axis=1)
    control["pooled_b"] = control["pooled_b"].fillna(0).astype(int)
    control["pooled_d"] = control["pooled_d_plus_b"] - control["pooled_b"]

    # Drug arm.
    keys = ["plate", "line", "drug_name", "concentration"]
    drug_total = (drug_raw.groupby(keys)["count"].sum()
                  .rename("total_well_count_a_plus_c").reset_index())
    drug_goi = (drug_raw[drug_raw["cell_cycle_group"].isin(groups_of_interest)]
                .groupby(keys)["count"].sum().rename("a").reset_index())
    analysis = drug_total.merge(drug_goi, on=keys, how="left")
    analysis["a"] = analysis["a"].fillna(0).astype(int)
    analysis["c"] = analysis["total_well_count_a_plus_c"] - analysis["a"]

    final = analysis.merge(control, left_on=["plate", "line"], right_index=True)

    rows = []
    for _, row in tqdm(final.iterrows(), total=len(final), desc="Fisher tests"):
        if row["pooled_d_plus_b"] == 0:
            continue
        a, b = row["a"], row["pooled_b"]
        c, d = row["c"], row["pooled_d"]

        _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        # Haldane-Anscombe correction keeps the odds ratio finite when a cell is 0.
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        log_odds_ratio = np.log(odds_ratio) if log_base == "e" else np.log2(odds_ratio)

        rows.append({
            "plate": row["plate"], "line": row["line"],
            "drug_name": row["drug_name"], "concentration": row["concentration"],
            "log_odds_ratio": log_odds_ratio, "p_value": p_value,
            "count_in_drug_well_GOI (a)": a,
            "pooled_count_in_control_GOI (b)": b,
            "total_in_drug_well (a+c)": row["total_well_count_a_plus_c"],
            "total_pooled_in_control (b+d)": row["pooled_d_plus_b"],
        })

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results = results.dropna(subset=["p_value"])
    reject, p_adj, _, _ = multipletests(results["p_value"], alpha=0.05, method="fdr_bh")
    results["p_adj_fdr"] = p_adj
    results["significant_fdr_0.05"] = reject
    return results.sort_values(by=["p_adj_fdr", "log_odds_ratio"],
                               ascending=[True, False])


def save_matrices(results: pd.DataFrame, analysis_name: str, output_dir: str):
    """Pivot the per-condition results into effect-size and significance matrices."""
    if results.empty:
        print(f"no results for {analysis_name}")
        return

    results = results.copy()
    results["drug__concentration"] = [
        drug_dose_key(d, c) for d, c in
        zip(results["drug_name"], results["concentration"])]

    # -log10(0) is infinite; floor exact zeros at the smallest non-zero FDR seen.
    min_nonzero = results.loc[results["p_adj_fdr"] > 0, "p_adj_fdr"].min()
    if pd.isna(min_nonzero):
        min_nonzero = 1e-300
    results["neg_log10_p"] = -np.log10(results["p_adj_fdr"].replace(0, min_nonzero))

    significance = results.pivot_table(index="drug__concentration", columns="line",
                                       values="neg_log10_p", aggfunc="mean")
    effect_size = results.pivot_table(index="drug__concentration", columns="line",
                                      values="log_odds_ratio", aggfunc="mean")

    sig_path = os.path.join(output_dir, f"significance_matrix_{analysis_name}.csv")
    eff_path = os.path.join(output_dir, f"effect_size_matrix_{analysis_name}.csv")
    significance.to_csv(sig_path)
    effect_size.to_csv(eff_path)
    print(f"  {effect_size.shape[0]} drug-doses x {effect_size.shape[1]} cell lines")
    print(f"  wrote {eff_path}\n         {sig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=COUNTS,
                    help="phase-resolved counts CSV from count_cells.py")
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--analyses", nargs="*", choices=list(ANALYSES),
                    default=list(ANALYSES), help="which phase groupings to run")
    ap.add_argument("--log_base", choices=["e", "2"], default="e",
                    help="base for the log odds ratio")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    counts = pd.read_csv(args.counts)
    counts = counts[counts["count"] > 0]

    for name in args.analyses:
        results = arrest_analysis(counts, ANALYSES[name], name, args.log_base)
        results.to_csv(os.path.join(args.output_dir,
                                    f"cell_cycle_arrest_results_{name}.csv"), index=False)
        save_matrices(results, name, args.output_dir)


if __name__ == "__main__":
    main()
