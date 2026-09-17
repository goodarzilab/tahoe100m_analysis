#!/usr/bin/env Rscript

# vision_analysis.R
#
# Run VISION pathway scoring on a single-plate pseudobulk h5ad.
# Accepts a single pre-filtered GMT file and a configurable control condition.
#
# Usage:
#   Rscript vision_analysis.R \
#     --input  /path/to/pseudobulk.h5ad \
#     --gmt    /path/to/filtered.gmt \
#     --control_col perturbation \
#     --control DMSO \
#     --prefix plate_id
#
# Output TSVs are written to --outdir (default: input file directory),
# named <prefix>_vision_scores.tsv.gz and <prefix>_vision_diff_scores.tsv.gz.


# ── 0. ARGUMENT PARSING ────────────────────────────────────────────────────────

suppressPackageStartupMessages(library(optparse))

option_list <- list(
  make_option(c("--input"),       type = "character", default = NULL,
              help = "Path to a pseudobulk h5ad file OR a directory of h5ad files [required]"),
  make_option(c("--outdir"),  type = "character", default = NULL,
              help = "Directory for output TSVs. Defaults to input file directory"),
  make_option(c("--prefix"),  type = "character", default = NULL,
              help = "Output filename prefix (e.g. plate_id). If set, outputs are named <prefix>_vision_scores.tsv.gz etc."),
  make_option(c("--gmt"),         type = "character", default = NULL,
              help = "Path to pre-filtered GMT file for VISION signatures [required]"),
  make_option(c("--control_col"), type = "character", default = "classification",
              help = "Obs column used to identify the control condition [default: %default]"),
  make_option(c("--control"),     type = "character", default = "non-targeting control",
              help = "Value in --control_col that identifies control samples [default: %default]"),
  make_option(c("--min_cells"),   type = "integer",   default = 10,
              help = "Minimum n_cells value to retain a pseudobulk sample [default: %default]"),
  make_option(c("--workers"),     type = "integer",   default = 20,
              help = "Number of parallel workers [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input)) stop("--input is required")
if (is.null(opt$gmt))   stop("--gmt is required")
if (!file.exists(opt$gmt)) stop(paste("GMT file not found:", opt$gmt))

if (!file.exists(opt$input)) stop(paste("--input not found:", opt$input))
input_files <- opt$input

make_output_paths <- function(h5ad_path, outdir, prefix) {
  base_dir <- if (!is.null(outdir)) outdir else dirname(h5ad_path)
  if (!is.null(prefix)) {
    list(
      scores = file.path(base_dir, paste0(prefix, "_vision_scores.tsv.gz")),
      diff   = file.path(base_dir, paste0(prefix, "_vision_diff_scores.tsv.gz"))
    )
  } else {
    list(
      scores = file.path(base_dir, "pseudobulk_vision_scores.tsv.gz"),
      diff   = file.path(base_dir, "pseudobulk_differential_vision_scores.tsv.gz")
    )
  }
}

cat("Input       :", opt$input,       "\n")
cat("GMT         :", opt$gmt,         "\n")
cat("Control col :", opt$control_col, "\n")
cat("Control     :", opt$control,     "\n")
cat("Min cells   :", opt$min_cells,   "\n")
cat("Workers     :", opt$workers,     "\n\n")


# ── 1. LIBRARIES ───────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(anndata)
  library(fgsea)
  library(tidyverse)
  library(data.table)
  library(Matrix)
  library(VISION)
  library(DESeq2)
  library(BiocParallel)
})


# ── 2. LOAD GENE SETS ──────────────────────────────────────────────────────────

cat("Loading gene sets from", opt$gmt, "...\n")
gs_f.ls <- gmtPathways(opt$gmt)
cat("Gene sets loaded:", length(gs_f.ls), "\n\n")


# ── 3. PER-FILE PROCESSING FUNCTION ───────────────────────────────────────────

run_one <- function(h5ad_path, out_paths, gmt_path, gs_f.ls, control_col, control, min_cells, n_workers) {
  cat(sprintf("\n[%s]\n", basename(h5ad_path)))

  cat("  Loading h5ad...\n")
  e.ad     <- read_h5ad(h5ad_path)
  e_obs.df <- e.ad$obs

  for (col in c("cell_line", "plate", "n_cells", control_col)) {
    if (!col %in% colnames(e_obs.df)) {
      stop(sprintf(
        "Required column '%s' not found in obs. Available columns: %s",
        col, paste(colnames(e_obs.df), collapse = ", ")
      ))
    }
  }

  keep <- e_obs.df$n_cells >= min_cells
  cat(sprintf("  Filtering n_cells >= %d: %d / %d samples retained\n",
              min_cells, sum(keep), nrow(e_obs.df)))
  e_obs.df <- e_obs.df[keep, , drop = FALSE]

  n_control <- sum(e_obs.df[[control_col]] == control, na.rm = TRUE)
  if (n_control == 0) {
    stop(sprintf(
      "No samples found where %s == '%s'. Check --control value.",
      control_col, control
    ))
  }
  cat(sprintf("  Control samples (%s == '%s'): %d\n", control_col, control, n_control))

  g_e_gs.chv <- intersect(e.ad$var_names, unlist(gs_f.ls))
  cat(sprintf("  Genes overlapping gene sets: %d\n", length(g_e_gs.chv)))
  if (length(g_e_gs.chv) == 0) {
    stop(paste0(
      "No genes overlap between the h5ad var index and the GMT gene sets.\n",
      "  h5ad var example: ", paste(head(e.ad$var_names, 5), collapse = ", "), "\n",
      "  GMT gene example: ", paste(head(unlist(gs_f.ls), 5), collapse = ", "), "\n",
      "  Ensure vision_reindex_vars.py ran and the correct GMT is supplied."
    ))
  }

  e.mx <- as.matrix(e.ad$X[keep, g_e_gs.chv])
  rm(e.ad); gc()

  # DESeq2 size-factor normalization. Only counts(normalized = TRUE) is used
  # downstream, and that depends solely on the size factors — so
  # estimateSizeFactors() is sufficient. The full DESeq() additionally runs
  # gene-wise dispersion + Wald fitting across workers, which is unused here
  # (design = ~1) and very memory-hungry at scale.
  cat("  Running DESeq2 size-factor normalization...\n")
  dds <- DESeqDataSetFromMatrix(
    countData = t(e.mx),
    colData   = e_obs.df,
    design    = ~1
  )
  dds               <- estimateSizeFactors(dds)
  normalized_counts <- counts(dds, normalized = TRUE)
  rm(dds, e.mx); gc()

  # VISION
  cat("  Running VISION...\n")
  data.vis <- Vision(
    data       = normalized_counts,
    signatures = gmt_path,
    meta       = e_obs.df[, control_col, drop = FALSE]
  )
  options(mc.cores = n_workers)
  data.vis        <- calcSignatureScores(data.vis, sig_gene_importance = FALSE)
  vis_gs_score.mx <- data.vis@SigScores
  rm(data.vis, normalized_counts); gc()

  # Raw scores + per-batch control-subtracted scores
  cat("  Computing per-batch control-subtracted scores (grouping by cell_line + plate)...\n")
  e_obs.df <- e_obs.df %>% mutate(cell_id = rownames(e_obs.df))

  vis_gs <- as.data.frame(vis_gs_score.mx)
  colnames(vis_gs) <- paste0("gs_", colnames(vis_gs))
  optional_cols <- c("drug", "drugname_drugconc", "sample")
  optional_cols <- optional_cols[optional_cols %in% colnames(e_obs.df)]
  join_cols <- unique(c("cell_id", "cell_line", "plate", optional_cols, control_col))
  vis_gs <- vis_gs %>%
    mutate(cell_id = rownames(vis_gs)) %>%
    inner_join(e_obs.df[, join_cols], by = "cell_id")

  diff_vis_gs <- vis_gs %>%
    group_by(cell_line, plate) %>%
    mutate(across(
      starts_with("gs_"),
      ~ . - median(.[.data[[control_col]] == control], na.rm = TRUE)
    )) %>%
    ungroup() %>%
    filter(.data[[control_col]] != control)

  out_dir <- dirname(out_paths$scores)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  write_tsv(vis_gs,      out_paths$scores)
  write_tsv(diff_vis_gs, out_paths$diff)
  cat(sprintf("  Scores written   -> %s\n", out_paths$scores))
  cat(sprintf("  Diff written     -> %s\n", out_paths$diff))
}


# ── 4. RUN OVER ALL INPUT FILES ────────────────────────────────────────────────

register(MulticoreParam(opt$workers))

for (h5ad_path in input_files) {
  out_paths <- make_output_paths(h5ad_path, opt$outdir, opt$prefix)
  run_one(h5ad_path, out_paths, opt$gmt, gs_f.ls, opt$control_col, opt$control, opt$min_cells, opt$workers)
}

cat("\nAll done.\n")
