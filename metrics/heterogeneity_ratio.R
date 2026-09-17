#!/usr/bin/env Rscript
# Normalise the within-population distances into the heterogeneity ratio.
#
# Reads the per-(cell line, plate) parquets written by heterogeneity.py and
# computes, for each row,
#
#     normalized_heterogeneity = drug_within_distance / dmso_within_distance
#
# A value above 1 means the drug-treated population is more dispersed in
# expression space than its DMSO control; below 1 means less. This is a linear
# ratio; take log2() if a symmetric scale is wanted.
#
# Also splits the drugname_drugconc string into separate drug and dose columns.
#
#
# Usage:
#   Rscript heterogeneity_ratio.R <input_dir> <output.parquet>

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript heterogeneity_ratio.R <input_dir> <output.parquet>")
}
input_dir <- args[1]
output_path <- args[2]

# --- paths: for the paper these were ---
# input_dir   : the per-(cell line, plate) parquet directory written by
#               heterogeneity.py, mirrored at
#               /large_storage/ctc/public/tahoe/heterogeneity/
# output_path : /large_storage/ctc/public/tahoe/analysis/tahoe_heterogeneity.parquet

df <- collect(open_dataset(input_dir))

# Parse "[('Drug name', 0.05, 'uM')]" with an anchored regex rather than by
# splitting on commas, so drug names containing commas survive intact.
parsed <- str_match(df$drugname_drugconc, "^\\[\\('(.*)',\\s*([0-9.]+),\\s*'([^']*)'\\)\\]$")

unparsed <- is.na(parsed[, 2])
if (any(unparsed)) {
  stop(sprintf(
    "%d condition strings did not match the expected \"[('drug', dose, 'unit')]\" form, e.g. %s",
    sum(unparsed), df$drugname_drugconc[which(unparsed)[1]]))
}

df <- df %>%
  mutate(
    # trimws: two drug names carry a trailing space in the source data.
    drug = trimws(parsed[, 2]),
    dose = parsed[, 3],
    normalized_heterogeneity = drug_within_distance / dmso_within_distance
  )

write_parquet(df, output_path)

cat(sprintf(
  "%d rows, %d cell lines, %d drugs -> %s\n",
  nrow(df), n_distinct(df$cell_line), n_distinct(df$drug), output_path
))
