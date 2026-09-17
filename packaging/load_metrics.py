"""Load the released Tahoe phenotypic metrics AnnData.

    from packaging.load_metrics import load_metrics, describe

    adata = load_metrics()
    describe(adata, "heterogeneity")

The object is 50 cell lines x 1137 drug-doses with one layer per phenotype.
Every layer's definition, units and notes are in `adata.uns["layers"]`.
"""

import argparse

import anndata as ad

REPO_ID = "arcinstitute/tahoe_phenotypic_metrics"
FILENAME = "tahoe_metrics.h5ad"


def load_metrics(repo_id: str = REPO_ID, filename: str = FILENAME,
                 cache_dir: str | None = None) -> ad.AnnData:
    """Download (and cache) the metrics AnnData from HuggingFace and read it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename,
                           repo_type="dataset", cache_dir=cache_dir)
    return ad.read_h5ad(path)


def describe(adata: ad.AnnData, layer: str | None = None) -> None:
    """Print what a layer contains, or list every layer."""
    spec = adata.uns.get("layers", {})
    if layer is None:
        for name in adata.layers:
            info = spec.get(name, {})
            print(f"{name:22s} {info.get('units', 'units not recorded')}")
        return

    if layer not in spec:
        raise KeyError(f"{layer!r} is not described; have {sorted(spec)}")
    info = spec[layer]
    print(f"{layer}\n")
    print(f"  {info['description']}\n")
    print(f"  units     : {info['units']}")
    print(f"  higher    : {info['direction']}")
    print(f"  producer  : {info['producer']}")
    print(f"  NaN means : {info['nan_meaning']}")
    for note in info.get("notes", []):
        print(f"  note      : {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", help="describe one layer instead of listing all")
    ap.add_argument("--repo_id", default=REPO_ID)
    args = ap.parse_args()

    adata = load_metrics(repo_id=args.repo_id)
    print(f"{adata.n_obs} cell lines x {adata.n_vars} drug-doses, "
          f"{len(adata.layers)} layers\n")
    describe(adata, args.layer)


if __name__ == "__main__":
    main()
