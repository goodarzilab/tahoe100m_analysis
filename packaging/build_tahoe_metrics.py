"""Assemble the released metrics AnnData: 50 cell lines x 1137 drug-doses.

Builds each metric layer from the matrices written by the scripts in
`metrics/`, attaches the cell line and drug annotation, and writes a `uns`
block describing what every layer contains.

`obs` (DepMap cell line metadata), `obsm` (mutation matrices) and `var` (drug
annotation) are taken from the annotation object given as `--source`, as is any
layer whose input matrix is not supplied.

Usage:
    python packaging/build_tahoe_metrics.py \\
        --source /large_storage/ctc/public/tahoe/tahoe_metrics_aggregated.h5ad \\
        --sensitivity results/survival/log2fc_survivals_averaged_across_plate.csv \\
        --heterogeneity results/heterogeneity/heterogeneity_matrix.csv \\
        --gini results/gini/aggregated/gini_zscore_matrix.csv \\
        --g_arrest results/cell_cycle/G_arrest_proportions.csv \\
        --arrest_dir results/cell_cycle \\
        --output tahoe_metrics.h5ad
"""

import os
import sys
import hashlib
import argparse
import datetime
import subprocess

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer_spec import LAYERS, as_uns_dict, markdown_table  # noqa: E402
from drug_names import drug_dose_key, match_key  # noqa: E402

# --- paths: edit for your system ---
SOURCE = "/large_storage/ctc/public/tahoe/tahoe_metrics_aggregated.h5ad"
OUTPUT = "tahoe_metrics.h5ad"

# X is a copy of this layer in the released object.
X_LAYER = "augur"

HF_REPO_ID = "arcinstitute/tahoe_phenotypic_metrics"

DATASET = {
    "title": "Tahoe-100M phenotypic metrics",
    "description": ("Per (cell line, drug, dose) phenotypic metrics derived from "
                    "the Tahoe-100M single-cell drug perturbation dataset."),
    "code": "https://github.com/goodarzilab/tahoe100m_analysis",
    "license": "CC-BY-4.0",
    "huggingface": HF_REPO_ID,
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_matrix(path: str) -> pd.DataFrame:
    """Read a metric matrix CSV, normalising the several index conventions in use.

    aggregate_gini.py writes a three-level index (drugname_drugconc, drugname,
    dose) with cell lines as the remaining columns; the others write a single
    key column. Where the drug name and dose are separate columns, build the
    canonical "<drug>__<dose>" key from them and drop the metadata columns.
    """
    df = pd.read_csv(path, index_col=0)
    if {"drugname", "dose"}.issubset(df.columns):
        df.index = df["drugname"].astype(str) + "__" + df["dose"].astype(str)
        df = df.drop(columns=[c for c in ("drugname", "dose", "drugname_drugconc")
                              if c in df.columns])
    return df


def to_var_keyed(df: pd.DataFrame, var_names, obs_names, label: str):
    """Align an input matrix onto the (obs x var) axes of the object.

    Accepts either orientation and either "Drug_0.05" or "Drug__0.05" row keys.
    Returns an obs x var float array with NaN where the input has no value.
    """
    obs_set = {match_key(o) for o in obs_names}
    # Orient so rows are drug-doses and columns are cell lines.
    if len({match_key(c) for c in df.columns} & obs_set) < len(obs_set) / 2:
        df = df.T

    def canonical(key):
        key = str(key)
        if "__" not in key and "_" in key:
            # "Drug_0.05" -> "Drug__0.05"; split on the LAST underscore only.
            drug, dose = key.rsplit("_", 1)
            key = f"{drug}__{dose}"
        return match_key(key)

    df = df.copy()
    df.index = [canonical(i) for i in df.index]
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [match_key(c) for c in df.columns]

    out = pd.DataFrame(np.nan, index=[match_key(o) for o in obs_names],
                       columns=[match_key(v) for v in var_names])
    rows = out.columns.intersection(df.index)
    cols = out.index.intersection(df.columns)
    out.loc[cols, rows] = df.loc[rows, cols].T.values

    missing = len(out.columns) - len(rows)
    if missing:
        print(f"    {label}: {missing}/{len(out.columns)} drug-doses absent from input")
    return out.values.astype(float)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    import importlib
    out = {}
    for pkg in ["numpy", "pandas", "anndata", "scanpy", "scipy", "sklearn",
                "statsmodels", "snf", "leidenalg", "igraph", "pertpy"]:
        try:
            out[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE,
                    help="existing metrics h5ad; supplies obs/obsm/var and any layer not rebuilt")
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--sensitivity")
    ap.add_argument("--heterogeneity")
    ap.add_argument("--gini")
    ap.add_argument("--augur")
    ap.add_argument("--g_arrest")
    ap.add_argument("--arrest_dir",
                    help="directory of effect_size_matrix_*.csv from g_arrest_fisher.py")
    ap.add_argument("--logodds_base", choices=["e", "2"], default="e",
                    help="'e' keeps the natural-log scale; '2' converts "
                         "the *_logodds layers to log2 (multiplies by 1/ln(2))")
    args = ap.parse_args()

    print(f"Reading annotation and fallback layers from {args.source}")
    source = ad.read_h5ad(args.source)
    obs_names, var_names = source.obs_names, source.var_names
    print(f"  {source.n_obs} cell lines x {source.n_vars} drug-doses")

    inputs = {
        "sensitivity": args.sensitivity,
        "heterogeneity": args.heterogeneity,
        "gini": args.gini,
        "augur": args.augur,
        "G_arrest": args.g_arrest,
    }
    if args.arrest_dir:
        for layer, fname in [
            ("G1_arrest_logodds", "effect_size_matrix_G1_arrest.csv"),
            ("G2M_arrest_logodds", "effect_size_matrix_G2M_arrest.csv"),
            ("G_arrest_logodds", "effect_size_matrix_G_arrest_(G1+G2M).csv"),
        ]:
            path = os.path.join(args.arrest_dir, fname)
            if os.path.exists(path):
                inputs[layer] = path

    layers, provenance, checksums = {}, {}, {}
    # match_key -> the spelling as it appears in the metric inputs
    input_spellings = {}
    print("\nLayers:")
    for name in LAYERS:
        spec = as_uns_dict()[name]
        path = inputs.get(name)
        if path:
            matrix = load_matrix(path)
            for key in matrix.index:
                text = str(key)
                if "__" in text:
                    d, sep, dose = text.rpartition("__")
                    input_spellings.setdefault(match_key(text), drug_dose_key(d, dose))
            values = to_var_keyed(matrix, var_names, obs_names, name)
            if name.endswith("_logodds") and args.logodds_base == "2":
                values = values * np.log2(np.e)
            checksums[os.path.basename(path)] = sha256(path)
        else:
            if name not in source.layers:
                print(f"  {name:20s} SKIPPED (no input and not in source)")
                continue
            values = np.asarray(source.layers[name], dtype=float)

        if name.endswith("_logodds"):
            spec["log_base"] = "natural" if args.logodds_base == "e" else "log2"
            if args.logodds_base == "2":
                spec["units"] = "log2 odds ratio"

        layers[name] = values
        provenance[name] = spec
        tag = "source" if not path else "rebuilt"
        finite = int(np.isfinite(values).sum())
        print(f"  {name:20s} {tag:8s} {finite:6d} values, "
              f"range {np.nanmin(values):8.3f} to {np.nanmax(values):7.3f}")


    # Drug names come from the metric inputs, which are parsed straight from the
    # plate metadata; fall back to whitespace normalisation for any drug no
    # input covers.
    var = source.var.copy()
    renamed = {}
    canonical = []
    for name in source.var_names:
        drug, dose = str(name).rsplit("__", 1)
        key = input_spellings.get(match_key(name)) or drug_dose_key(drug, dose)
        canonical.append(key)
        if key != str(name):
            renamed[str(name)] = key
    if len(set(canonical)) != len(canonical):
        raise SystemExit("canonical var names are not unique; refusing to write")
    var.index = pd.Index(canonical, name=source.var.index.name)
    if "drug" in var.columns:
        var["drug"] = [drug_dose_key(d, "").removesuffix("__") for d in var["drug"]]
    print(f"\nvar axis: {len(renamed)} names normalised")
    for old_name, new_name in list(renamed.items())[:4]:
        print(f"  {old_name!r} -> {new_name!r}")

    adata = ad.AnnData(
        X=layers[X_LAYER].copy(),
        obs=source.obs.copy(),
        var=var,
        layers=layers,
    )
    for key in source.obsm:
        adata.obsm[key] = source.obsm[key].copy()

    adata.uns["layers"] = provenance
    adata.uns["X"] = f"copy of layers['{X_LAYER}']"
    adata.uns["dataset"] = dict(DATASET)
    adata.uns["provenance"] = {
        "built": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "source_sha256": sha256(args.source),
        "input_sha256": checksums,
        "versions": package_versions(),
    }

    adata.write_h5ad(args.output, compression="gzip")
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nWrote {args.output} ({size_mb:.1f} MB): "
          f"{adata.n_obs} x {adata.n_vars}, {len(adata.layers)} layers")

    verify_roundtrip(args.output, adata)

    card = os.path.join(os.path.dirname(args.output) or ".", "README.md")
    with open(card, "w") as fh:
        fh.write(dataset_card(adata))
    print(f"Wrote {card} (HuggingFace dataset card)")


def dataset_card(adata: ad.AnnData) -> str:
    """The HuggingFace dataset card, generated from the same layer registry."""
    return f"""---
license: cc-by-4.0
tags:
  - single-cell
  - drug-perturbation
  - cancer
---

# Tahoe-100M phenotypic metrics

Per (cell line, drug, dose) phenotypic metrics derived from the Tahoe-100M
single-cell drug perturbation dataset.
**{adata.n_obs} cell lines x {adata.n_vars} drug-doses**, one layer per phenotype.

```python
from huggingface_hub import hf_hub_download
import anndata as ad

path = hf_hub_download("{HF_REPO_ID}", "tahoe_metrics.h5ad",
                       repo_type="dataset")
adata = ad.read_h5ad(path)
adata.uns["layers"]["heterogeneity"]     # what the numbers mean
```

## Layers

{markdown_table()}

`X` is a copy of `layers["{X_LAYER}"]`.

## Units

- `heterogeneity` is a linear drug/DMSO ratio (median 0.975, range 0 to 3.17)
- `G_arrest` is a log2 ratio of a proportion
- the `*_logodds` layers use the natural log

Every layer's definition, units, direction and producer script are in
`adata.uns["layers"]`; build details are in `adata.uns["provenance"]`.

## Provenance

Built by `packaging/build_tahoe_metrics.py` in
[{DATASET['code']}]({DATASET['code']}), which documents how each metric was
computed. `obs` is DepMap cell line metadata, `obsm` holds mutation matrices,
`var` is the drug annotation (MOA, targets, SMILES, PubChem).
"""


def verify_roundtrip(path: str, built: ad.AnnData):
    """h5ad `uns` accepts only simple types; confirm everything survived."""
    back = ad.read_h5ad(path)
    problems = []

    for name, values in built.layers.items():
        got = np.asarray(back.layers[name], dtype=float)
        if not (np.allclose(values, got, equal_nan=True)):
            problems.append(f"layer {name} changed on write")

    for name in built.uns["layers"]:
        original = built.uns["layers"][name]
        restored = back.uns["layers"][name]
        for field, value in original.items():
            got = restored.get(field)
            if isinstance(value, list):
                got = list(got) if got is not None else []
                if [str(v) for v in value] != [str(v) for v in got]:
                    problems.append(f"uns.layers.{name}.{field} changed")
            elif str(value) != str(got):
                problems.append(f"uns.layers.{name}.{field} changed "
                                f"({value!r} -> {got!r})")

    if problems:
        print("\nROUND-TRIP PROBLEMS:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("Round-trip check passed: layers and uns survive write/read unchanged")


if __name__ == "__main__":
    main()
