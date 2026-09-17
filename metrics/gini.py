"""Gini heterogeneity: is a drug's response concentrated in a few cell states?

For one plate, for each (cell line, drug-dose) condition:

  1. subset to the condition's cells plus the plate's DMSO_TF control cells for
     that cell line, subsampling to --n_subsample cells per condition
  2. normalise, log1p, PCA, neighbours, Leiden cluster at each --resolutions value
  3. build the cluster x condition contingency table and compute the Gini
     coefficient of each condition's cluster occupancy vector -- high Gini means
     the cells pile into a few clusters, low Gini means they spread evenly
  4. permute the condition labels --n_permutations times to get a null, and
     report the drug and control Gini as z-scores against it
     (gini_drug_zscore, gini_ctrl_zscore)

Also reports a chi-square test on the contingency table and a KS test between
the drug and control cluster distributions.

Parameters used for the paper (one array task per plate):

    python gini.py <plate>.h5ad --output_dir <dir> \
        --perturbation_col drugname_drugconc \
        --control_condition "[('DMSO_TF', 0.0, 'uM')]" \
        --cell_line_col cell_name \
        --resolutions 1.0,2.0,3.0 --n_subsample 1000 \
        --n_permutations 100 --n_tasks 8 --random_seed 42

Writes one CSV per plate; combine them across plates with aggregate_gini.py,
which produces the matrix behind the `gini` layer of the metrics AnnData.

Parallelised over conditions with multiprocessing.Pool (--n_tasks).
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import os
import re
import time
import numpy as np
import argparse
from typing import Dict, Tuple, List, Optional, Union
import ast
from scipy.stats import chi2_contingency, ks_2samp
import gc
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_gini_coefficient(array: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a numpy array.
    Based on: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    
    Args:
        array: Array of values
        
    Returns:
        Gini coefficient
    """
    # All values are treated equally, arrays must be 1d:
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def subsample_indices_by_condition(obs_df: pd.DataFrame, condition_col: str, n_cells: int, random_state: int = 42) -> np.ndarray:
    """
    Get indices for subsampling cells to have equal numbers from each condition.
    More memory efficient than creating new AnnData objects.
    
    Args:
        obs_df: DataFrame with observation metadata
        condition_col: Column name containing condition labels
        n_cells: Target number of cells per condition
        random_state: Random seed for reproducibility
        
    Returns:
        Array of indices for subsampled cells
    """
    np.random.seed(random_state)
    
    conditions = obs_df[condition_col].unique()
    subsampled_indices = []
    
    # Find the minimum number of cells across conditions
    min_cells = min([sum(obs_df[condition_col] == cond) for cond in conditions])
    actual_n_cells = min(n_cells, min_cells)
    
    for condition in conditions:
        condition_indices = np.where(obs_df[condition_col] == condition)[0]
        if len(condition_indices) >= actual_n_cells:
            sampled_indices = np.random.choice(condition_indices, size=actual_n_cells, replace=False)
        else:
            sampled_indices = condition_indices
        subsampled_indices.extend(sampled_indices)
    
    return np.array(subsampled_indices)

def calculate_gini_zscore_metrics(obs_df: pd.DataFrame, cluster_col: str, condition_col: str, 
                                control_name: str, treatment_name: str, 
                                n_permutations: int = 100, random_state: int = 42) -> Dict:
    """
    Calculate Gini coefficient z-scores through permutation testing.
    
    Args:
        obs_df: DataFrame with observation metadata including cluster assignments
        cluster_col: Column name containing cluster assignments
        condition_col: Column name containing condition labels
        control_name: Name of the control condition
        treatment_name: Name of the treatment condition
        n_permutations: Number of permutations for null distribution
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with Gini coefficient metrics and z-scores
    """
    np.random.seed(random_state)
    
    # Create contingency table (counts for each cluster and treatment)
    contingency_table = pd.crosstab(obs_df[cluster_col], obs_df[condition_col])
    
    # Calculate observed Gini coefficients for each condition
    observed_gini = contingency_table.apply(lambda x: calculate_gini_coefficient(x.values), axis=0)
    
    # Generate null distribution through permutation testing
    random_ginis = []
    for i in range(n_permutations):
        # Shuffle treatment labels
        shuffled_treatment = obs_df[condition_col].sample(frac=1, random_state=random_state + i).reset_index(drop=True)
        shuffled_treatment.index = obs_df[condition_col].index
        
        # Create contingency table with shuffled labels
        shuffled_contingency = pd.crosstab(obs_df[cluster_col], shuffled_treatment)
        
        # Ensure column names are strings and match expected conditions
        shuffled_contingency.index = [str(x) for x in shuffled_contingency.index]
        shuffled_contingency.columns = [str(x) for x in shuffled_contingency.columns]
        
        # Calculate Gini for this permutation
        perm_gini = shuffled_contingency.apply(lambda x: calculate_gini_coefficient(x.values), axis=0)
        random_ginis.append(perm_gini)
    
    # Convert to DataFrame for easier statistics
    random_df = pd.DataFrame(random_ginis)
    
    # Calculate z-scores
    null_means = random_df.mean()
    null_stds = random_df.std()
    z_scores = (observed_gini - null_means) / null_stds
    
    # Prepare results dictionary
    results = {
        # Observed Gini coefficients
        'gini_ctrl_observed': observed_gini.get(control_name, np.nan),
        'gini_drug_observed': observed_gini.get(treatment_name, np.nan),
        
        # Null distribution statistics
        'gini_ctrl_null_mean': null_means.get(control_name, np.nan),
        'gini_ctrl_null_std': null_stds.get(control_name, np.nan),
        'gini_drug_null_mean': null_means.get(treatment_name, np.nan),
        'gini_drug_null_std': null_stds.get(treatment_name, np.nan),
        
        # Z-scores
        'gini_ctrl_zscore': z_scores.get(control_name, np.nan),
        'gini_drug_zscore': z_scores.get(treatment_name, np.nan),
        
        # Additional metadata
        'n_permutations': n_permutations,
        'median_cells_per_cluster': obs_df[cluster_col].value_counts().median()
    }
    
    return results

def calculate_chi2_test(obs_df: pd.DataFrame, cluster_col: str, condition_col: str) -> Dict:
    """
    Calculate Chi-square test for independence between clusters and conditions.
    
    Args:
        obs_df: DataFrame with observation metadata
        cluster_col: Column name containing cluster assignments
        condition_col: Column name containing condition labels
        
    Returns:
        Dictionary with Chi-square test results
    """
    
    # Create contingency table
    contingency_table = pd.crosstab(obs_df[cluster_col], obs_df[condition_col])
    
    # Perform Chi-square test
    chi2_stat, chi2_pval, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
    
    return {
        'chi2_statistic': chi2_stat,
        'chi2_pvalue': chi2_pval,
        'chi2_dof': chi2_dof
    }

def calculate_ks_test(obs_df: pd.DataFrame, cluster_col: str, condition_col: str, 
                     control_name: str, treatment_name: str) -> Dict:
    """
    Calculate Kolmogorov-Smirnov test comparing cluster distributions between conditions.
    
    Args:
        obs_df: DataFrame with observation metadata
        cluster_col: Column name containing cluster assignments
        condition_col: Column name containing condition labels
        control_name: Name of the control condition
        treatment_name: Name of the treatment condition
        
    Returns:
        Dictionary with KS test results
    """
    # Create contingency table and calculate proportions
    contingency_table = pd.crosstab(obs_df[cluster_col], obs_df[condition_col])
    
    # Calculate proportions for each condition
    ctrl_proportions = contingency_table[control_name] / contingency_table[control_name].sum()
    drug_proportions = contingency_table[treatment_name] / contingency_table[treatment_name].sum()
    
    # Perform KS test
    ks_stat, ks_pval = ks_2samp(ctrl_proportions, drug_proportions)
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval
    }

def calculate_cluster_distribution_metrics(obs_df: pd.DataFrame, cluster_col: str, condition_col: str, 
                                         control_name: str = 'DMSO_TF', treatment_name: str = None, 
                                         n_permutations: int = 100, random_state: int = 42) -> Dict:
    """
    Calculate comprehensive cluster distribution metrics including Gini z-scores, Chi2, and KS tests.
    
    Args:
        obs_df: DataFrame with observation metadata including cluster assignments
        cluster_col: Column name containing cluster assignments
        condition_col: Column name containing condition labels
        control_name: Name of the control condition (e.g., 'DMSO_TF')
        treatment_name: Name of the treatment condition (e.g., drug name)
        n_permutations: Number of permutations for null distribution
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with all distribution metrics
    """
    # Basic cell counts and metadata
    contingency_table = pd.crosstab(obs_df[cluster_col], obs_df[condition_col])
    
    results = {
        'n_clusters': len(contingency_table),
        'total_cells': len(obs_df),
        'ctrl_cells': sum(obs_df[condition_col] == control_name),
        'drug_cells': sum(obs_df[condition_col] == treatment_name),
    }
    
    # Add Gini z-score metrics
    gini_metrics = calculate_gini_zscore_metrics(
        obs_df, cluster_col, condition_col, control_name, treatment_name, 
        n_permutations, random_state
    )
    results.update(gini_metrics)
    
    # Add Chi-square test
    chi2_metrics = calculate_chi2_test(obs_df, cluster_col, condition_col)
    results.update(chi2_metrics)
    
    # Add KS test
    ks_metrics = calculate_ks_test(obs_df, cluster_col, condition_col, control_name, treatment_name)
    results.update(ks_metrics)
    
    return results

def get_heterogeneity_statistics(adata_path: str, 
                                perturbation_col: str,
                                control_condition: str,
                                perturbation_condition: str,
                                cell_line_col: Optional[str] = None,
                                cell_line_subset: Optional[str] = None,
                                resolutions: List[float] = [0.1, 0.3, 0.5, 0.8, 1.0],
                                n_subsample: int = 1000,
                                n_permutations: int = 100,
                                random_seed: int = 42) -> List[Dict]:
    """
    Calculate heterogeneity statistics for a single perturbation vs control comparison.
    
    Args:
        adata_path: Path to the AnnData file
        perturbation_col: Column name containing perturbation labels
        control_condition: Name of the control condition
        perturbation_condition: Name of the perturbation condition
        cell_line_col: Optional column name for cell line filtering
        cell_line_subset: Optional specific cell line to subset to
        resolutions: List of clustering resolutions to test
        n_subsample: Number of cells to subsample per condition
        n_permutations: Number of permutations for null distribution
        random_seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing heterogeneity metrics for each resolution
    """
    try:
        # Load data in backed mode
        adata = sc.read_h5ad(adata_path, backed='r')
        logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Filter for the specific perturbation and control
        condition_mask = (adata.obs[perturbation_col] == perturbation_condition) | \
                        (adata.obs[perturbation_col] == control_condition)
        
        # If cell_line_subset is provided, also filter for that cell line
        if cell_line_subset and cell_line_col and cell_line_col in adata.obs.columns:
            cell_line_mask = adata.obs[cell_line_col] == cell_line_subset
            condition_mask = condition_mask & cell_line_mask
            logger.info(f"Filtering to cell line: {cell_line_subset}")
        
        if not condition_mask.any():
            logger.warning(f"No cells found for {perturbation_condition} vs {control_condition}" + 
                          (f" in {cell_line_subset}" if cell_line_subset else ""))
            return []
        
        # Get indices
        indices = np.where(condition_mask)[0]
        
        # Create subset - bring to memory
        subset = adata[indices, :].to_memory()
        
        # Close the file handle
        adata.file.close()
        del adata
        
        # Check if we have both conditions
        available_conditions = subset.obs[perturbation_col].unique()
        if control_condition not in available_conditions or perturbation_condition not in available_conditions:
            logger.warning(f"Missing conditions. Available: {available_conditions}")
            return []
        
        # Process the subset
        results = _process_single_subset(
            subset, perturbation_col, control_condition, perturbation_condition,
            resolutions, n_subsample, n_permutations, random_seed,
            cell_line=cell_line_subset
        )
        
        # Clean up memory
        del subset
        gc.collect()
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {perturbation_condition} vs {control_condition}" + 
                    (f" in {cell_line_subset}" if cell_line_subset else "") + f": {str(e)}")
        gc.collect()
        return []

def _process_single_subset(subset, perturbation_col, control_condition, perturbation_condition,
                          resolutions, n_subsample, n_permutations, random_seed,
                          cell_line=None) -> List[Dict]:
    """
    Helper function to process a single subset (e.g., one cell line) for heterogeneity analysis.
    """
    # Apply subsampling if needed
    if n_subsample > 0:
        subsample_idx = subsample_indices_by_condition(subset.obs, perturbation_col, n_subsample, random_seed)
        subset = subset[subsample_idx, :].copy()
    
    # Normalize and preprocess
    sc.pp.normalize_total(subset, exclude_highly_expressed=True)
    sc.pp.log1p(subset)
    sc.pp.highly_variable_genes(subset, flavor="seurat", n_top_genes=2000)
    
    # Use only highly variable genes for clustering
    subset = subset[:, subset.var.highly_variable].copy()
    
    # Scale data
    sc.pp.scale(subset, max_value=10)
    
    # Compute PCA with fixed random state
    sc.tl.pca(subset, svd_solver='arpack', random_state=random_seed)
    
    # Compute neighborhood graph
    sc.pp.neighbors(subset, n_neighbors=10, n_pcs=25, random_state=random_seed)
    
    # Test different clustering resolutions
    results = []
    
    for resolution in resolutions:
        # Perform clustering with fixed random state
        sc.tl.leiden(subset, resolution=resolution, key_added=f'leiden_{resolution}', 
                    random_state=random_seed)
        
        # Calculate distribution metrics for this clustering
        cluster_col = f'leiden_{resolution}'
        metrics = calculate_cluster_distribution_metrics(
            subset.obs, cluster_col, perturbation_col, control_condition, perturbation_condition, 
            n_permutations, random_seed
        )
        
        # Add metadata
        metrics['perturbation'] = perturbation_condition
        metrics['control'] = control_condition
        metrics['resolution'] = resolution
        
        # Add cell line information if available
        if cell_line:
            metrics['cell_line'] = cell_line
        
        results.append(metrics)
    
    return results

def _analyze_perturbation_wrapper(task_data):
    """
    Wrapper function for parallel processing of perturbation analysis.
    This is a module-level function that can be pickled for multiprocessing.
    """
    return get_heterogeneity_statistics(*task_data)

def run_heterogeneity_analysis(adata_path: str, 
                              output_dir: Optional[str] = None, 
                              perturbation_col: str = 'drugname_drugconc',
                              control_condition: str = 'DMSO_TF',
                              cell_line_col: Optional[str] = 'cell_name',
                              cell_line: Optional[str] = None,
                              resolutions: List[float] = [1, 2, 3],
                              n_subsample: int = 1000,
                              n_permutations: int = 100,
                              n_tasks: int = 1,
                              random_seed: int = 42,
                              stratify_by_cell_line: bool = True) -> pd.DataFrame:
    """
    Run comprehensive heterogeneity analysis on an AnnData file.
    
    Args:
        adata_path: Path to the AnnData file
        output_dir: Optional directory to save results (if None, no files are saved)
        perturbation_col: Column name containing perturbation labels
        control_condition: Name of the control condition
        cell_line_col: Optional column name for cell line stratification
        cell_line: Optional specific cell line to analyze (if provided, will iterate over perturbation x cell_line combinations)
        resolutions: List of clustering resolutions to test
        n_subsample: Number of cells to subsample per condition
        n_permutations: Number of permutations for null distribution
        n_tasks: Number of parallel tasks (1 = sequential processing)
        random_seed: Random seed for reproducibility
        stratify_by_cell_line: If True, analyze each perturbation separately for each cell line (ignored if cell_line is specified)
        
    Returns:
        DataFrame containing all heterogeneity analysis results
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing {adata_path}")
    
    # Load data once in backed mode to get perturbation info
    adata = sc.read_h5ad(adata_path, backed='r')
    logger.info(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Get unique perturbations (excluding control)
    perturbations = adata.obs[perturbation_col].unique()
    perturbations = [p for p in perturbations if p != control_condition]
    logger.info(f"Found {len(perturbations)} perturbations to analyze (excluding {control_condition})")
    
    # Get cell lines if needed
    if cell_line:
        # Specific cell line provided - analyze just that one
        cell_lines = [cell_line]
        logger.info(f"Analyzing specific cell line: {cell_line}")
    elif cell_line_col and cell_line_col in adata.obs.columns and stratify_by_cell_line:
        # Get all available cell lines for stratification
        cell_lines = adata.obs[cell_line_col].unique()
        logger.info(f"Found {len(cell_lines)} cell lines for stratification")
    else:
        cell_lines = [None]  # No cell line stratification
    
    # Close the file handle for now
    adata.file.close()
    
    # Prepare tasks for analysis
    tasks = []
    
    if cell_line or (cell_line_col and cell_line_col in adata.obs.columns and stratify_by_cell_line):
        # Iterate over perturbation x cell_line combinations
        for perturbation in perturbations:
            for cl in cell_lines:
                tasks.append((
                    adata_path, perturbation_col, control_condition, perturbation,
                    cell_line_col, cl, resolutions, n_subsample, n_permutations, 
                    random_seed
                ))
        
        logger.info(f"Created {len(tasks)} tasks ({len(perturbations)} perturbations × {len(cell_lines)} cell lines)")
    else:
        # No cell line stratification - iterate over perturbations only
        for perturbation in perturbations:
            tasks.append((
                adata_path, perturbation_col, control_condition, perturbation,
                cell_line_col, None, resolutions, n_subsample, n_permutations, 
                random_seed
            ))
        
        logger.info(f"Created {len(tasks)} tasks (perturbations only)")
    
    # Process perturbations (either in parallel or sequentially)
    all_results = []
    
    if n_tasks > 1:
        logger.info(f"Using {n_tasks} parallel processes for {len(tasks)} tasks")
        with Pool(processes=n_tasks) as pool:
            results = list(tqdm(
                pool.imap(_analyze_perturbation_wrapper, tasks),
                total=len(tasks),
                desc="Processing tasks"
            ))
    else:
        logger.info(f"Processing {len(tasks)} tasks sequentially")
        results = [_analyze_perturbation_wrapper(task) for task in tqdm(tasks, desc="Processing tasks")]
    
    # Aggregate all results into a single list
    for result_list in results:
        if result_list:  # Not empty
            all_results.extend(result_list)
    
    logger.info(f"Completed analysis. Got {len(all_results)} total results.")
    
    # Convert to DataFrame
    if all_results:
        combined_results = pd.DataFrame(all_results)
        logger.info(f"Created DataFrame with {len(combined_results)} rows and {len(combined_results.columns)} columns")
    else:
        logger.warning("No results to return.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Save results if output directory is specified
    if output_dir:
        # Generate output filename based on input file
        input_basename = os.path.splitext(os.path.basename(adata_path))[0]
        
        # Add cell line suffix if analyzing specific cell line
        if cell_line:
            input_basename += f"__{cell_line}"
        
        # Save combined results
        output_file = os.path.join(output_dir, f"{input_basename}__heterogeneity_analysis.csv")
        combined_results.to_csv(output_file, index=False)
        logger.info(f"Saved {len(all_results)} results to {output_file}")
    
    return combined_results

def main():
    parser = argparse.ArgumentParser(description='Run heterogeneity analysis on an AnnData file')
    parser.add_argument('adata_path', type=str, help='Path to AnnData file to analyze')
    parser.add_argument('--output_dir', type=str, 
                      default=None,
                      help='Directory to save results (optional - if not provided, no files are saved)')
    parser.add_argument('--perturbation_col', type=str, default='drugname',
                      help='Column name containing perturbation labels')
    parser.add_argument('--control_condition', type=str, default='DMSO_TF',
                      help='Name of the control condition')
    parser.add_argument('--cell_line_col', type=str, default='cell_name',
                      help='Column name for cell line stratification (optional)')
    parser.add_argument('--cell_line', type=str, default=None,
                      help='Specific cell line to analyze (if provided, analyzes perturbation x cell_line combinations)')
    parser.add_argument('--resolutions', type=str, default='0.1,0.3,0.5,0.8,1.0',
                      help='Comma-separated list of clustering resolutions to test')
    parser.add_argument('--n_subsample', type=int, default=1000,
                      help='Number of cells to subsample per condition')
    parser.add_argument('--n_permutations', type=int, default=100,
                      help='Number of permutations for null distribution')
    parser.add_argument('--n_tasks', type=int, default=10,
                      help='Number of parallel tasks (1 = sequential processing)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--no_stratify', action='store_true',
                      help='Do not stratify analysis by cell line')
    
    args = parser.parse_args()
    
    # Parse resolutions
    resolutions = [float(r.strip()) for r in args.resolutions.split(',')]
    logger.info(f"Testing resolutions: {resolutions}")
    logger.info(f"Subsampling to {args.n_subsample} cells per condition")
    logger.info(f"Using {args.n_permutations} permutations for null distribution")
    logger.info(f"Using {args.n_tasks} parallel tasks")
    logger.info(f"Using random seed: {args.random_seed}")
    
    # Handle optional columns
    cell_line_col = args.cell_line_col if args.cell_line_col != 'None' else None
    stratify_by_cell_line = not args.no_stratify and cell_line_col is not None
    
    # Set default output directory if not provided
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = 'results/gini'
        logger.info(f"No output directory specified, using default: {output_dir}")
    
    results_df = run_heterogeneity_analysis(
        args.adata_path, output_dir, args.perturbation_col, args.control_condition,
        cell_line_col, args.cell_line, resolutions, args.n_subsample, args.n_permutations, 
        args.n_tasks, args.random_seed, stratify_by_cell_line
    )
    
    logger.info(f"Analysis complete. Returned DataFrame with {len(results_df)} rows.")

if __name__ == "__main__":
    main() 