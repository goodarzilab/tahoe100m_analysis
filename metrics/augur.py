"""Augur: how strongly does each drug perturb each cell line?

Augur asks how separable a perturbed population is from its control by training
a classifier to tell them apart; the cross-validated AUC becomes the
perturbation score. Cell lines whose transcriptome barely moves under a drug
are hard to classify and score near 0.5.

For one plate, for each (drug, dose): pool the drug's cells with the plate's
DMSO_TF cells, normalise (excluding highly expressed genes), log1p, take the
top 2000 Seurat highly variable genes, and run

    pertpy.tl.Augur("random_forest_classifier")
      .predict(subsample_size=20, select_variance_features=False,
               cell_type_col="cell_name")

so that each cell line is scored separately within the shared drug-vs-control
comparison.

Writes per (drug, dose), into <output_dir>/plate<N>/:
    summary_metrics__<drug>_<dose>.csv      per-cell-line AUC and CV metrics
    feature_importances__<drug>_<dose>.csv  random-forest gene importances
    drugname_dose.tsv.gz                    the drug-dose inventory for the plate

Combine plates with aggregate_augur.py to get the `augur` layer.

Usage (one array task per plate):
    python augur.py 6 --output_dir <dir> --n_processes 12
"""

import pertpy as pt
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
from typing import Dict, Tuple
import ast

# --- paths: edit for your system ---
PLATES = "/processed_datasets/scRecount/tahoe/original_h5ad/plate{n}_filtered.h5ad.gz"
OUTPUT_DIR = "results/augur"

def parse_drug_info_efficient(drugname_series: pd.Series, drop_uniform_unit: bool = True) -> Dict[str, pd.Series]:
    """
    Efficiently parse drug information from a series of drug strings.
    
    Args:
        drugname_series: Series containing drug strings in format "[('drugname', dose, 'unit')]"
        drop_uniform_unit: If True, don't return unit series if all units are the same
    
    Returns:
        Dictionary of Series containing 'drugname', 'dose', and optionally 'unit'
    """
    # Get unique values to parse only once
    unique_drugs = drugname_series.unique()
    
    # Create mapping dictionaries for each component
    drugname_map = {}
    dose_map = {}
    unit_map = {}
    
    # Parse unique values once
    for drug_str in unique_drugs:
        parsed = ast.literal_eval(drug_str)[0]  # [0] since format is always a single-item list
        drugname_map[drug_str] = parsed[0]
        dose_map[drug_str] = parsed[1]
        unit_map[drug_str] = parsed[2]
    
    # Check if units are uniform
    uniform_unit = len(set(unit_map.values())) == 1 if drop_uniform_unit else False
    
    # Create result series using map (more efficient than apply)
    result = {
        'drugname': drugname_series.map(drugname_map),
        'dose': drugname_series.map(dose_map)
    }
    
    if not (drop_uniform_unit and uniform_unit):
        result['unit'] = drugname_series.map(unit_map)
    
    return result

def run_augur_analysis(plate_num: int, output_dir: str, n_processes: int):
    """
    Run Augur analysis on a specific plate's data.
    
    Args:
        plate_num: The plate number to analyze
        output_dir: Base directory to save outputs
        n_processes: Number of processes to use for Augur
    """
    # Create plate-specific output directory
    plate_dir = os.path.join(output_dir, f"plate{plate_num}")
    os.makedirs(plate_dir, exist_ok=True)
    
    # Load the data
    data_path = PLATES.format(n=plate_num)
    print(f"Processing plate {plate_num} from {data_path}")
    
    # Load data in backed mode
    adata = sc.read_h5ad(data_path, backed='r')
    
    # Parse drug information
    parsed = parse_drug_info_efficient(adata.obs['drugname_drugconc'])
    for col, series in parsed.items():
        adata.obs[col] = series
    
    # Get unique drug treatments (excluding DMSO_TF)
    drugs = adata.obs['drugname'].unique()
    drugs = [drug for drug in drugs if drug != 'DMSO_TF']
    print(f"Found {len(drugs)} drugs to analyze (excluding DMSO_TF)")
    
    # Create a summary DataFrame for drug-dose combinations
    drug_dose_summary = []
    
    # Initialize Augur
    aug = pt.tl.Augur("random_forest_classifier")
    
    # Process each drug
    for drug in drugs:
        print(f"\nProcessing drug: {drug}")
        
        # Get unique doses for this drug
        doses = adata.obs[adata.obs['drugname'] == drug]['dose'].unique()
        
        for dose in doses:
            print(f"Processing dose: {dose}")
            
            # Clean drug name for filenames and create base name with dose
            safe_drug_name = re.sub(r'[^A-Za-z0-9\-]+', '_', drug)
            base_name = f"{safe_drug_name}_{dose}"
            
            try:
                # Get drug and control cells
                drug_idx = (adata.obs['drugname'] == drug) & (adata.obs['dose'] == dose)
                control_idx = adata.obs['drugname'] == 'DMSO_TF'

                if (np.sum(drug_idx) == 0) | (np.sum(control_idx) == 0):
                    print(f"No cells found for {drug} at {dose}uM")
                    continue

                idx = drug_idx | control_idx
                
                # Create subset and load into memory
                subset = adata[idx, :].to_memory()
                
                # Record drug-dose combination and cell counts
                drug_dose_summary.append({
                    'drug': drug,
                    'dose': dose,
                    'treatment_cells': sum(drug_idx),
                    'control_cells': sum(control_idx)
                })
                
                # Normalize and preprocess
                sc.pp.normalize_total(subset, exclude_highly_expressed=True)
                sc.pp.log1p(subset)
                sc.pp.highly_variable_genes(subset, flavor="seurat", n_top_genes=2000)
                
                # Create condition labels
                subset.obs['condition'] = subset.obs['drugname']
                subset.obs["condition"].replace({"DMSO_TF": "ctrl", drug: "stim"}, inplace=True)
                
                # Prepare data for Augur
                loaded_data = aug.load(subset, label_col="condition", cell_type_col="cell_name")
                
                # Run Augur prediction
                print(f"Running Augur for {drug} at {dose}uM...")
                start_time = time.time()
                aug_adata, aug_results = aug.predict(
                    loaded_data,
                    subsample_size=20,
                    n_threads=n_processes,
                    select_variance_features=False,
                    # span=1
                )
                elapsed_time = time.time() - start_time
                print(f"Prediction took {elapsed_time:.2f} seconds")
                
                # Save results
                # Summary metrics
                df_summary = aug_results["summary_metrics"].copy()
                df_summary["drug"] = drug
                df_summary["dose"] = dose
                df_summary.to_csv(os.path.join(plate_dir, f"summary_metrics__{base_name}.csv"), index=False)
                
                # Feature importances
                df_feat = aug_results["feature_importances"].copy()
                df_feat["drug"] = drug
                df_feat["dose"] = dose
                df_feat.to_csv(os.path.join(plate_dir, f"feature_importances__{base_name}.csv"), index=False)
                
                # Plots
                # Lollipop plot
                #fig, ax = plt.subplots(figsize=(10, 6))
                #aug.plot_lollipop(aug_results, ax=ax)
                #plt.title(f"{drug} ({dose}uM) - Plate {plate_num}")
                #fig.savefig(os.path.join(plate_dir, f"lollipop_{base_name}.png"), dpi=300, bbox_inches='tight')
                #plt.close(fig)
                
                # Important features plot
                #fig, ax = plt.subplots(figsize=(12, 8))
                #aug.plot_important_features(aug_results, ax=ax)
                #plt.title(f"{drug} ({dose}uM) - Plate {plate_num}")
                #fig.savefig(os.path.join(plate_dir, f"important_features_{base_name}.png"), dpi=300, bbox_inches='tight')
                #plt.close(fig)
                
                print(f"Completed analysis for {drug} at {dose}uM")
                
            except Exception as e:
                print(f"Error processing {drug} at {dose}uM: {str(e)}")
                continue
    
    # Save drug-dose summary
    pd.DataFrame(drug_dose_summary).to_csv(
        os.path.join(plate_dir, "drugname_dose.tsv.gz"),
        sep='\t',
        index=False,
        compression='gzip'
    )
    
    # Close the file
    adata.file.close()
    print(f"\nCompleted all analyses for plate {plate_num}")

def main():
    parser = argparse.ArgumentParser(description='Run Augur analysis on a specific plate')
    parser.add_argument('plate_num', type=int, help='Plate number to analyze')
    parser.add_argument('--output_dir', type=str, 
                      default=OUTPUT_DIR,
                      help='Directory to save results')
    parser.add_argument('--n_processes', type=int, default=12,
                      help='Number of processes to use for Augur')
    
    args = parser.parse_args()
    run_augur_analysis(args.plate_num, args.output_dir, args.n_processes)

if __name__ == "__main__":
    main() 
