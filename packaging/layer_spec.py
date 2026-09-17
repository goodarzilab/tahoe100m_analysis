"""What each layer of the metrics AnnData actually contains.

One entry per layer. This is the single source of truth: build_tahoe_metrics.py
writes it into `uns` and the README table is generated from it, so the
documentation stays in step with the data.

Units:

  * `heterogeneity` is a linear drug/DMSO ratio
  * `G_arrest` is a log2 ratio of a proportion
  * the `*_logodds` layers use the natural log
"""

from dataclasses import dataclass, asdict, field


@dataclass
class LayerSpec:
    description: str
    units: str
    direction: str            # what a higher value means
    producer: str             # repo-relative script that computes it
    inputs: str               # what that script reads
    statistical_test: str = "none"
    multiple_testing: str = "none"
    nan_meaning: str = "condition not assayed, or dropped by a QC filter"
    notes: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        # h5ad uns cannot hold None; empty lists are fine but keep it simple.
        d["notes"] = list(self.notes)
        return d


LAYERS = {
    "sensitivity": LayerSpec(
        description=(
            "Survival. Each cell line's share of a pooled drug well, log2 fold "
            "changed against that plate's DMSO_TF baseline share, averaged "
            "across plates."),
        units="log2 fold change of compositional share (pseudocount 0.01)",
        direction="higher = the line held or gained share, i.e. tolerated the drug",
        producer="metrics/survival_log2fc.py",
        inputs="per-plate cell counts from metrics/count_cells.py",
    ),
    "heterogeneity": LayerSpec(
        description=(
            "Within-population transcriptional dispersion under drug relative "
            "to DMSO. Mean distance of cells to their group centroid in "
            "variance-weighted PCA space, drug divided by control."),
        units="dimensionless ratio (drug / DMSO); median 0.975, range 0 to 3.17",
        direction="higher = the treated population is more dispersed than its control",
        producer="metrics/heterogeneity.py, metrics/heterogeneity_ratio.R",
        inputs="per-plate h5ads; ratio computed in R",
        notes=[
            "A linear ratio.",
            "About 20% of conditions were assayed on more than one plate; the "
            "value is the mean across plates.",
        ],
    ),
    "gini": LayerSpec(
        description=(
            "Gini coefficient of a condition's Leiden cluster-occupancy "
            "vector, z-scored against 100 label permutations, averaged over "
            "clustering resolutions 1.0/2.0/3.0 and across plates."),
        units="permutation z-score; range -4.95 to 40.15",
        direction="higher = the drug's cells concentrate into fewer transcriptional states",
        producer="metrics/gini.py, metrics/aggregate_gini.py",
        inputs="per-plate h5ads",
        statistical_test="permutation z-score (100 label permutations)",
        nan_meaning=(
            "condition not assayed, or dropped for having fewer than 10 "
            "clusters or fewer than 10 cells per cluster"),
        notes=[
            "A mean across three clustering resolutions, not a single resolution.",
        ],
    ),
    "augur": LayerSpec(
        description=(
            "Augur perturbation score: cross-validated AUC of a random forest "
            "separating a cell line's drug-treated cells from its DMSO cells."),
        units="classifier AUC; observed range 0.43 to 0.999",
        direction="higher = a stronger transcriptional response to the drug",
        producer="metrics/augur.py, metrics/aggregate_augur.py",
        inputs="per-plate h5ads via pertpy",
        statistical_test="random forest cross-validation (pertpy Augur)",
    ),
    "G_arrest": LayerSpec(
        description=(
            "Fraction of surviving cells in a gap phase, (G1+G2M)/(G1+G2M+S), "
            "log2 fold changed against the same cell line's mean DMSO_TF "
            "proportion on the same plate, averaged across plates."),
        units="log2 ratio of a proportion; range -1.50 to 0.74",
        direction="higher = more of the surviving population is out of cycle",
        producer="metrics/g_arrest.py",
        inputs="per-plate cell counts from metrics/count_cells.py",
        notes=[
            "A log2 ratio of a proportion (the output file is named "
            "G_arrest_proportions.csv).",
        ],
    ),
}

_ARREST_LOGODDS = {
    "G1_arrest_logodds": ("G1", "G1"),
    "G2M_arrest_logodds": ("G2M", "G2/M"),
    "G_arrest_logodds": ("G1+G2M", "G1 or G2/M"),
}
for _name, (_group, _label) in _ARREST_LOGODDS.items():
    LAYERS[_name] = LayerSpec(
        description=(
            f"Enrichment of {_label} phase cells under drug versus the plate's "
            f"pooled DMSO_TF control, as a log odds ratio with a "
            f"Haldane-Anscombe 0.5 correction. Phase group: {_group}."),
        units="natural log odds ratio",
        direction=f"higher = more {_label} cells than in the pooled control",
        producer="metrics/g_arrest_fisher.py",
        inputs="per-plate cell counts from metrics/count_cells.py",
        statistical_test="Fisher's exact test, two-sided",
        multiple_testing="Benjamini-Hochberg FDR across all conditions in the analysis",
        notes=[
            "Natural log.",
        ],
    )

def as_uns_dict():
    """The layer registry in a form h5ad `uns` accepts (str/num/list/dict only)."""
    return {name: spec.to_dict() for name, spec in LAYERS.items()}


def markdown_table():
    """Layer table for the README / HuggingFace dataset card."""
    rows = ["| Layer | What it is | Units | Higher means |",
            "|---|---|---|---|"]
    for name, s in LAYERS.items():
        desc = s.description.split(". ")[0].rstrip(".")
        rows.append(f"| `{name}` | {desc} | {s.units} | {s.direction} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(markdown_table())
    print()
    for name, spec in LAYERS.items():
        print(f"{name}: {len(spec.notes)} note(s)")
