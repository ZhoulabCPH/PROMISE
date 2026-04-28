# =============================================================================
# Proteomics Preprocessing and DreamAI KNN Imputation Pipeline
#
# Description:
#   This script performs the following steps on raw proteomic data:
#     1. Read the raw proteomics expression matrix (features x samples).
#     2. Apply log2 transformation to raw intensities.
#     3. Apply median centering normalization across samples.
#     4. Exclude proteins with > 25% missing values across samples.
#     5. Impute remaining missing values using k-nearest neighbors (k = 10)
#        implemented in the DreamAI R package.
#
# Required input format:
#   - Proteomics table: one protein/gene per row, one sample per column.
#     Must contain a feature identifier column (e.g., "Gene").
#   - Clinical table (optional): one row per sample with a sample ID column.
#
# Required packages:
#   - DreamAI  (install from GitHub: WangLab-MSSM/DreamAI/Code)
#   - readr
#
# Usage:
#   1. Edit the config block below to set file paths and parameters.
#   2. Optionally customize normalize_proteomics_ids() and
#      normalize_clinical_ids() for cohort-specific ID matching.
#   3. Run with: Rscript proteomics_dreamai_imputation.R
# =============================================================================

suppressPackageStartupMessages({
  library(DreamAI)
  library(readr)
})

# Set seed for reproducibility of KNN imputation
set.seed(0)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
config <- list(
  # Path to the raw proteomics expression file (.csv, .tsv, or .txt)
  proteomics_path       = "path/to/proteomics_expression.tsv",

  # Path to the clinical annotation file (set NULL if not used)
  clinical_path         = NULL,

  # Path for the output imputed matrix
  output_path           = "path/to/proteomics_dreamai_imputed.csv",

  # Name of the column containing protein/gene identifiers
  feature_column        = "Gene",

  # Name of the sample ID column in the clinical table
  clinical_id_column    = "sample_id",

  # Number of leading non-data rows to skip (e.g., description rows)
  remove_leading_rows   = 0L,

  # Number of trailing metadata columns to remove
  remove_trailing_columns = 0L,

  # Whether to treat zero values as missing (NA)
  convert_zero_to_na    = FALSE,

  # Whether to remove duplicated feature identifiers (keep first occurrence)
  deduplicate_features  = TRUE,

  # Maximum allowed fraction of missing values per protein (>25% are excluded)
  max_missing_ratio     = 0.25,

  # Whether to apply log2 transformation to raw intensities
  log2_transform        = TRUE,

  # Whether to apply median centering normalization across samples
  median_centering      = TRUE,

  # Whether to subset samples to those present in the clinical table
  match_with_clinical   = FALSE,

  # Whether to write the imputed matrix to disk
  write_output          = TRUE,

  # DreamAI arguments: use only KNN with k = 10
  dreamai_args = list(
    k                 = 10,        # Number of nearest neighbors for KNN
    maxiter_MF        = 10,        # Max iterations for MissForest (unused here)
    ntree             = 100,       # Number of trees for MissForest (unused here)
    maxnodes          = NULL,      # Max terminal nodes per tree (unused here)
    maxiter_ADMIN     = 30,        # Max iterations for ADMIN (unused here)
    tol               = 1e-2,      # Convergence tolerance
    gamma_ADMIN       = NA,        # Regularization for ADMIN (unused here)
    gamma             = 50,        # Regularization parameter
    CV                = FALSE,     # Cross-validation flag
    fillmethod        = "row_mean",# Initial fill method before KNN
    maxiter_RegImpute = 10,        # Max iterations for RegImpute (unused here)
    conv_nrmse        = 1e-6,      # Convergence criterion
    iter_SpectroFM    = 40,        # Iterations for SpectroFM (unused here)
    method            = "KNN",     # Use ONLY KNN imputation
    out               = "KNN"      # Output the KNN result (not Ensemble)
  )
)


# =============================================================================
# Utility Functions
# =============================================================================

#' Check whether a file exists; stop with an informative error if not.
assert_file_exists <- function(path) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    stop(sprintf("File does not exist: %s", path), call. = FALSE)
  }
}


#' Read a tabular file (.csv, .tsv, or .txt) and return a tibble.
read_tabular_file <- function(path) {
  assert_file_exists(path)
  extension <- tolower(tools::file_ext(path))

  if (extension %in% c("tsv", "txt")) {
    return(readr::read_tsv(path, show_col_types = FALSE, progress = FALSE))
  }

  if (extension == "csv") {
    return(readr::read_csv(path, show_col_types = FALSE, progress = FALSE))
  }

  stop(
    sprintf("Unsupported file extension for %s. Use .csv, .tsv, or .txt.", path),
    call. = FALSE
  )
}


#' Sanitize column names: trim whitespace and replace internal spaces with dots.
sanitize_column_names <- function(x) {
  x <- trimws(x)
  x <- gsub("[[:space:]]+", ".", x)
  x
}


#' Normalize proteomics sample IDs (customize for cohort-specific conventions).
normalize_proteomics_ids <- function(x) {
  # Override this function if proteomics column names require
  # cohort-specific normalization before matching to the clinical table.
  x
}


#' Normalize clinical sample IDs (customize for cohort-specific conventions).
normalize_clinical_ids <- function(x) {
  # Override this function if clinical sample IDs require
  # cohort-specific normalization before matching to the proteomics matrix.
  x
}


# =============================================================================
# Core Processing Functions
# =============================================================================

#' Parse the raw proteomics data frame into a numeric expression matrix.
#'
#' Steps:
#'   - Remove leading rows and trailing columns if configured.
#'   - Extract and validate feature identifiers.
#'   - Optionally deduplicate features (keep first occurrence).
#'   - Convert expression columns to numeric.
#'   - Optionally convert zeros to NA.
#'
#' @param proteomics_df  Data frame read from the input file.
#' @param config         Configuration list.
#' @return A numeric matrix with features as rows and samples as columns.
prepare_proteomics_table <- function(proteomics_df, config) {
  proteomics_df <- as.data.frame(proteomics_df, check.names = FALSE)

  # --- Remove leading non-data rows ---
  if (config$remove_leading_rows > 0L) {
    proteomics_df <- proteomics_df[-seq_len(config$remove_leading_rows), , drop = FALSE]
  }

  # --- Remove trailing metadata columns ---
  if (config$remove_trailing_columns > 0L) {
    keep_ncol <- ncol(proteomics_df) - config$remove_trailing_columns
    if (keep_ncol <= 0L) {
      stop("`remove_trailing_columns` is too large for the input proteomics table.",
           call. = FALSE)
    }
    proteomics_df <- proteomics_df[, seq_len(keep_ncol), drop = FALSE]
  }

  # --- Sanitize column names ---
  colnames(proteomics_df) <- sanitize_column_names(colnames(proteomics_df))

  # --- Validate feature column existence ---
  if (!config$feature_column %in% colnames(proteomics_df)) {
    stop(
      sprintf("Feature column '%s' was not found in the proteomics table. "
              , config$feature_column),
      call. = FALSE
    )
  }

  # --- Extract and clean feature identifiers ---
  feature_ids <- trimws(as.character(proteomics_df[[config$feature_column]]))
  valid_rows  <- !is.na(feature_ids) & nzchar(feature_ids)
  proteomics_df <- proteomics_df[valid_rows, , drop = FALSE]
  feature_ids   <- feature_ids[valid_rows]

  # --- Deduplicate features (keep first occurrence) ---
  if (config$deduplicate_features) {
    dup_mask      <- duplicated(feature_ids)
    n_dup         <- sum(dup_mask)
    if (n_dup > 0L) {
      message(sprintf("Removed %d duplicated feature(s); keeping first occurrence.", n_dup))
    }
    proteomics_df <- proteomics_df[!dup_mask, , drop = FALSE]
    feature_ids   <- feature_ids[!dup_mask]
  }

  # --- Build numeric expression matrix ---
  expression_df <- proteomics_df[, setdiff(colnames(proteomics_df), config$feature_column),
                                 drop = FALSE]
  expression_df <- as.data.frame(
    lapply(expression_df, function(col) suppressWarnings(as.numeric(col))),
    check.names = FALSE
  )

  # --- Optionally convert zeros to NA ---
  if (config$convert_zero_to_na) {
    expression_df[expression_df == 0] <- NA
  }

  rownames(expression_df) <- feature_ids
  colnames(expression_df) <- normalize_proteomics_ids(colnames(expression_df))

  expression_matrix <- as.matrix(expression_df)
  if (!is.numeric(expression_matrix)) {
    storage.mode(expression_matrix) <- "numeric"
  }

  if (nrow(expression_matrix) == 0L || ncol(expression_matrix) == 0L) {
    stop("The processed proteomics matrix is empty. Check the input format.",
         call. = FALSE)
  }

  message(sprintf("Proteomics matrix prepared: %d features x %d samples",
                  nrow(expression_matrix), ncol(expression_matrix)))
  expression_matrix
}


#' Apply log2 transformation to a numeric expression matrix.
#'
#' Non-positive values (which are undefined under log2) are set to NA before
#' transformation. This is consistent with standard proteomics workflows where
#' raw intensities are strictly positive.
#'
#' @param expression_matrix  Numeric matrix (features x samples).
#' @return Log2-transformed numeric matrix.
log2_transform <- function(expression_matrix) {
  if (!is.matrix(expression_matrix)) {
    expression_matrix <- as.matrix(expression_matrix)
  }

  # Count non-positive finite values that will become NA
  non_positive_count <- sum(expression_matrix <= 0, na.rm = TRUE)
  if (non_positive_count > 0L) {
    message(sprintf("Note: %d non-positive value(s) set to NA before log2 transformation.",
                    non_positive_count))
    expression_matrix[expression_matrix <= 0] <- NA
  }

  expression_matrix <- log2(expression_matrix)

  message(sprintf("Log2 transformation applied: %d features x %d samples",
                  nrow(expression_matrix), ncol(expression_matrix)))
  expression_matrix
}


#' Apply median centering normalization across samples.
#'
#' For each sample (column), the column median is subtracted so that all samples
#' share the same median of zero. This corrects for systematic differences in
#' total protein loading between samples.
#'
#' @param expression_matrix  Numeric matrix (features x samples), typically
#'                           already log2-transformed.
#' @return Median-centered numeric matrix.
median_centering <- function(expression_matrix) {
  if (!is.matrix(expression_matrix)) {
    expression_matrix <- as.matrix(expression_matrix)
  }

  # Compute per-sample (column) medians, ignoring NAs
  col_medians <- apply(expression_matrix, 2, median, na.rm = TRUE)

  # Check for samples where the median could not be computed
  bad_samples <- is.na(col_medians)
  if (any(bad_samples)) {
    warning(sprintf(
      "Median centering: %d sample(s) have all-NA columns and cannot be centered.",
      sum(bad_samples)
    ))
  }

  # Subtract each sample's median (sweep across columns)
  expression_matrix <- sweep(expression_matrix, 2, col_medians, FUN = "-")

  message(sprintf("Median centering normalization applied: %d features x %d samples",
                  nrow(expression_matrix), ncol(expression_matrix)))
  expression_matrix
}


#' Subset the expression matrix to samples present in the clinical table.
#'
#' @param expression_matrix  Numeric matrix (features x samples).
#' @param clinical_df        Data frame with clinical annotations.
#' @param config             Configuration list.
#' @return Subsetted numeric matrix containing only matched samples.
match_samples_with_clinical <- function(expression_matrix, clinical_df, config) {
  clinical_df <- as.data.frame(clinical_df, check.names = FALSE)

  if (!config$clinical_id_column %in% colnames(clinical_df)) {
    stop(
      sprintf("Clinical ID column '%s' was not found in the clinical table.",
              config$clinical_id_column),
      call. = FALSE
    )
  }

  clinical_ids <- normalize_clinical_ids(
    as.character(clinical_df[[config$clinical_id_column]])
  )
  clinical_ids <- clinical_ids[!is.na(clinical_ids) & nzchar(clinical_ids)]

  common_ids <- intersect(colnames(expression_matrix), clinical_ids)
  if (length(common_ids) == 0L) {
    stop(
      paste(
        "No matched sample IDs between the proteomics matrix and clinical table.",
        "Ensure: 1) proteomics columns are normalized;",
        "2) clinical_id_column is correct;",
        "3) normalize_*_ids() functions are adjusted if needed."
      ),
      call. = FALSE
    )
  }

  matched_matrix <- expression_matrix[, common_ids, drop = FALSE]
  message(sprintf("Matched samples with clinical data: %d features x %d samples",
                  nrow(matched_matrix), ncol(matched_matrix)))
  matched_matrix
}


#' Filter out proteins whose fraction of missing values exceeds a threshold.
#'
#' Proteins with > max_missing_ratio missing values are excluded.
#' E.g., with max_missing_ratio = 0.25, proteins missing in more than 25% of
#' samples are removed.
#'
#' @param expression_matrix  Numeric matrix (features x samples).
#' @param max_missing_ratio  Maximum allowed fraction of NA values (exclusive >).
#' @return Filtered numeric matrix.
filter_features_by_missingness <- function(expression_matrix, max_missing_ratio = 0.25) {
  if (!is.matrix(expression_matrix)) {
    expression_matrix <- as.matrix(expression_matrix)
  }

  # Compute per-protein missing fraction
  missing_ratio <- rowMeans(is.na(expression_matrix))

  # Keep proteins with missing ratio <= threshold (exclude those > threshold)
  filtered_matrix <- expression_matrix[missing_ratio <= max_missing_ratio, , drop = FALSE]

  if (nrow(filtered_matrix) == 0L) {
    stop("All features were removed after missingness filtering. "
         , call. = FALSE)
  }

  message(sprintf(
    "Missingness filter (exclude > %.0f%%): %d -> %d features retained",
    max_missing_ratio * 100,
    nrow(expression_matrix),
    nrow(filtered_matrix)
  ))

  filtered_matrix
}


#' Run DreamAI imputation on the expression matrix.
#'
#' When method = "KNN" and out = "KNN", this performs KNN-based imputation
#' with k nearest neighbors.
#'
#' @param expression_matrix  Numeric matrix (features x samples) with NAs.
#' @param dreamai_args       List of arguments passed to DreamAI::DreamAI().
#' @return Imputed numeric matrix with no missing values.
run_dreamai_imputation <- function(expression_matrix, dreamai_args) {
  # Determine which output key to extract from DreamAI results
  out_key <- dreamai_args$out
  if (is.null(out_key) || !nzchar(out_key)) {
    out_key <- "Ensemble"
  }

  # Call DreamAI
  dreamai_call      <- c(list(expression_matrix), dreamai_args)
  imputation_result <- do.call(DreamAI::DreamAI, dreamai_call)

  # Extract the result corresponding to the selected method
  if (!out_key %in% names(imputation_result)) {
    stop(sprintf(
      "DreamAI output does not contain '%s'. Available: %s",
      out_key,
      paste(names(imputation_result), collapse = ", ")
    ), call. = FALSE)
  }

  imputed_matrix <- imputation_result[[out_key]]
  message(sprintf("DreamAI KNN imputation completed: %d features x %d samples",
                  nrow(imputed_matrix), ncol(imputed_matrix)))

  # Verify no remaining NAs
  remaining_na <- sum(is.na(imputed_matrix))
  if (remaining_na > 0L) {
    warning(sprintf("Imputed matrix still contains %d NA value(s).", remaining_na))
  }

  imputed_matrix
}


#' Write the imputed matrix to a CSV file.
#'
#' @param imputed_matrix  Numeric matrix to write.
#' @param output_path     File path for the output CSV.
write_imputed_matrix <- function(imputed_matrix, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  write.csv(imputed_matrix, output_path, quote = FALSE)
  message(sprintf("Imputed proteomics matrix saved to: %s", output_path))
}


# =============================================================================
# Main Pipeline
# =============================================================================

#' Execute the full proteomics preprocessing and imputation pipeline.
#'
#' Pipeline order:
#'   1. Read raw proteomics data.
#'   2. Parse into a numeric matrix (deduplicate, clean, etc.).
#'   3. (Optional) Match samples with clinical annotations.
#'   4. Log2 transformation of raw intensities.
#'   5. Median centering normalization across samples.
#'   6. Exclude proteins with > 25% missing values.
#'   7. KNN imputation (k = 10) via DreamAI.
#'   8. (Optional) Write the imputed matrix to CSV.
#'
#' @param config  Configuration list (see top of script).
#' @return Invisible list with filtered_matrix and imputed_matrix.
process_proteomics <- function(config) {

  # --- Validate configuration ---
  if (is.null(config$proteomics_path) || !nzchar(config$proteomics_path)) {
    stop("Please set `config$proteomics_path` to a valid file path.", call. = FALSE)
  }
  if (isTRUE(config$write_output) &&
      (is.null(config$output_path) || !nzchar(config$output_path))) {
    stop("Please set `config$output_path` when `write_output = TRUE`.", call. = FALSE)
  }
  if (!is.numeric(config$max_missing_ratio) ||
      config$max_missing_ratio <= 0 ||
      config$max_missing_ratio >= 1) {
    stop("`config$max_missing_ratio` must be in the interval (0, 1).", call. = FALSE)
  }

  # --- Step 1: Read raw proteomics data ---
  message("Step 1: Reading raw proteomics data...")
  proteomics_df    <- read_tabular_file(config$proteomics_path)

  # --- Step 2: Parse into numeric expression matrix ---
  message("Step 2: Preparing expression matrix...")
  expression_matrix <- prepare_proteomics_table(proteomics_df, config)

  # --- Step 3 (optional): Match samples with clinical table ---
  if (isTRUE(config$match_with_clinical)) {
    if (is.null(config$clinical_path)) {
      stop("`match_with_clinical = TRUE` requires a valid `clinical_path`.",
           call. = FALSE)
    }
    message("Step 3: Matching samples with clinical data...")
    clinical_df       <- read_tabular_file(config$clinical_path)
    expression_matrix <- match_samples_with_clinical(expression_matrix, clinical_df, config)
  }

  # --- Step 4: Log2 transformation ---
  if (isTRUE(config$log2_transform)) {
    message("Step 4: Applying log2 transformation...")
    expression_matrix <- log2_transform(expression_matrix)
  }

  # --- Step 5: Median centering normalization ---
  if (isTRUE(config$median_centering)) {
    message("Step 5: Applying median centering normalization...")
    expression_matrix <- median_centering(expression_matrix)
  }

  # --- Step 6: Filter by missingness ---
  message("Step 6: Filtering proteins by missingness...")
  expression_matrix <- filter_features_by_missingness(
    expression_matrix  = expression_matrix,
    max_missing_ratio  = config$max_missing_ratio
  )

  # --- Step 7: KNN imputation via DreamAI ---
  message("Step 7: Running DreamAI KNN imputation (k = 10)...")
  imputed_matrix <- run_dreamai_imputation(
    expression_matrix = expression_matrix,
    dreamai_args      = config$dreamai_args
  )

  # --- Step 8: Write output ---
  if (isTRUE(config$write_output)) {
    message("Step 8: Saving imputed matrix...")
    write_imputed_matrix(imputed_matrix, config$output_path)
  }

  message("Pipeline completed successfully.")

  invisible(list(
    filtered_matrix = expression_matrix,
    imputed_matrix  = imputed_matrix
  ))
}


# =============================================================================
# Execute
# =============================================================================
# result <- process_proteomics(config)
