# =============================================================================
# aggregate_cell_morphology.py
#
# Aggregate nucleus-level morphometric, textural, and intensity features to
# patient-level summary statistics, stratified by cell type.
#
# For each segmented nucleus the upstream pipeline (e.g., HoVer-Net +
# feature extraction) has already computed a comprehensive set of
# handcrafted descriptors (see PMID: 37880211):
#
#   Morphometric features:
#     Area, BoundingBoxArea, Eccentricity, Circularity, Elongation, Extent,
#     MajorAxisLength, MinorAxisLength, Perimeter, Solidity,
#     CurvatureStd, CurvatureMax, CurvatureMin.
#
#   GLCM textural features:
#     Angular Second Moment (ASM), Contrast, Correlation, Entropy,
#     Homogeneity.
#
#   First-order intensity statistics:
#     IntensityMean, IntensityStd, IntensityMax, IntensityMin.
#
# This script reads per-nucleus feature CSVs, groups nuclei by patient and
# cell type, computes the within-group mean for every feature, appends the
# cell-type proportion, and writes a single patient-level feature table.
#
# Input format (per-nucleus CSV):
#   Columns: Patient_Name, type, ASM, Contrast, Correlation, Entropy,
#            Homogeneity, IntensityMean, IntensityStd, IntensityMax,
#            IntensityMin, Area, AreaBbox, CellEccentricities, Circularity,
#            Elongation, Extent, MajorAxisLength, MinorAxisLength, Perimeter,
#            Solidity, CurvMean, CurvStd, CurvMax, CurvMin
#   - One row per segmented nucleus.
#   - 'type' is the integer cell-type label (1–5); type 0 = background
#     artifacts and is excluded.
#
# Output format (patient-level CSV):
#   Columns: Patient_Name, type, Cell_Ratio, <all feature columns>, Label
#   - One row per (patient, cell_type) combination.
#   - 'Cell_Ratio' is the proportion of nuclei of that type within the
#     patient.
#   - Feature columns contain the mean value across all nuclei of that type
#     within the patient.
#   - 'Label' indicates the risk group (e.g., "High" / "Low").
#
# Usage:
#   python aggregate_cell_morphology.py \
#       --low_csv  <path_to_low_risk_cell_features.csv>  \
#       --high_csv <path_to_high_risk_cell_features.csv>  \
#       --output   <path_to_output_patient_features.csv>
# =============================================================================

import os
import argparse

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

# Cell-type labels used in the segmentation model (1–5).
# Type 0 is background / artifact and is always excluded.
CELL_TYPE_LABELS = [1, 2, 3, 4, 5]

# Nucleus-level feature columns to aggregate (mean) per patient per cell type.
# Order: GLCM texture -> intensity statistics -> morphometric descriptors.
FEATURE_COLUMNS = [
    # --- GLCM textural features ---
    'ASM',                  # Angular Second Moment (energy / uniformity)
    'Contrast',             # GLCM contrast
    'Correlation',          # GLCM correlation
    'Entropy',              # GLCM entropy
    'Homogeneity',          # GLCM homogeneity (inverse difference moment)
    # --- First-order intensity statistics ---
    'IntensityMean',        # Mean pixel intensity within the nucleus
    'IntensityStd',         # Standard deviation of pixel intensities
    'IntensityMax',         # Maximum pixel intensity
    'IntensityMin',         # Minimum pixel intensity
    # --- Morphometric descriptors ---
    'Area',                 # Nuclear area (pixels)
    'AreaBbox',             # Bounding-box area
    'CellEccentricities',  # Eccentricity of the fitted ellipse
    'Circularity',          # 4*pi*Area / Perimeter^2
    'Elongation',           # MajorAxisLength / MinorAxisLength
    'Extent',               # Area / BoundingBoxArea
    'MajorAxisLength',      # Length of the major axis of the fitted ellipse
    'MinorAxisLength',      # Length of the minor axis of the fitted ellipse
    'Perimeter',            # Contour perimeter
    'Solidity',             # Area / ConvexHullArea
    # --- Boundary curvature summary ---
    'CurvMean',             # Mean curvature along the nucleus boundary
    'CurvStd',              # Standard deviation of boundary curvature
    'CurvMax',              # Maximum boundary curvature
    'CurvMin',              # Minimum boundary curvature
]

# Column names for the output patient-level feature table.
OUTPUT_COLUMNS = ['Patient_Name', 'type', 'Cell_Ratio'] + FEATURE_COLUMNS


# =============================================================================
# Core Functions
# =============================================================================

def compute_cell_type_proportions(cell_data):
    """Compute the proportion of each cell type within one patient.

    Args:
        cell_data: DataFrame of nucleus-level records for a single patient.
                   Must contain a 'type' column with integer cell-type labels.

    Returns:
        DataFrame of shape (1, len(CELL_TYPE_LABELS)) where each column is
        the fraction of nuclei belonging to that cell type. Missing types
        are filled with 0.
    """
    type_counts = cell_data.groupby('type').size()
    type_proportions = type_counts.reindex(CELL_TYPE_LABELS, fill_value=0) / len(cell_data)
    return pd.DataFrame(type_proportions).T


def aggregate_patient_features(cell_data):
    """Aggregate nucleus-level features to patient-level by cell type.

    For each cell type present in the patient, the function:
      1. Computes the cell-type proportion (fraction of total nuclei).
      2. Takes the mean of each morphometric / textural / intensity feature
         across all nuclei of that type.
      3. Returns one row per cell type.

    Args:
        cell_data: DataFrame of nucleus-level records for a single patient.
                   Must contain columns: 'Patient_Name', 'type', and all
                   entries in FEATURE_COLUMNS.

    Returns:
        List of single-row DataFrames, one per cell type present.
    """
    patient_name = cell_data['Patient_Name'].iloc[0]
    proportions = compute_cell_type_proportions(cell_data)

    patient_rows = []
    for cell_type, type_subset in cell_data.groupby('type'):
        # Mean of all features across nuclei of this cell type
        feature_means = type_subset[FEATURE_COLUMNS].values.astype(float).mean(axis=0)

        # Assemble the output row
        row_values = (
            [patient_name, cell_type, proportions[cell_type].values[0]]
            + feature_means.tolist()
        )
        row_df = pd.DataFrame([row_values], columns=OUTPUT_COLUMNS)
        patient_rows.append(row_df)

    return patient_rows


def process_cohort(csv_path, label):
    """Process a cohort CSV and return patient-level aggregated features.

    Steps:
      1. Read the per-nucleus feature CSV.
      2. Remove duplicate rows and background artifacts (type == 0).
      3. Group by patient and aggregate features per cell type.
      4. Assign a cohort label (e.g., "High" or "Low").

    Args:
        csv_path: Path to the input per-nucleus feature CSV.
        label:    String label assigned to all patients in this cohort.

    Returns:
        DataFrame with patient-level features (one row per patient × cell type).
    """
    # -- Read and clean the input data --
    cell_df = pd.read_csv(csv_path)

    # Drop the unnamed index column if present
    if cell_df.columns[0].startswith('Unnamed'):
        cell_df = cell_df.iloc[:, 1:]

    cell_df = cell_df.drop_duplicates()

    # Exclude background / artifact nuclei (type == 0)
    cell_df = cell_df[cell_df['type'] != 0].copy()

    if cell_df.empty:
        raise ValueError(f"No valid nuclei remain after filtering type == 0 in: {csv_path}")

    # -- Aggregate per patient per cell type --
    patient_groups = cell_df.groupby('Patient_Name')
    all_patient_rows = []

    for _, patient_data in patient_groups:
        rows = aggregate_patient_features(patient_data)
        all_patient_rows.extend(rows)

    cohort_df = pd.concat(all_patient_rows, axis=0, ignore_index=True)
    cohort_df['Label'] = label

    print(f"  [{label}] Processed {cohort_df['Patient_Name'].nunique()} patients, "
          f"{len(cohort_df)} (patient x cell_type) rows from: {csv_path}")

    return cohort_df


# =============================================================================
# Main
# =============================================================================

def main():
    """Parse arguments, process Low / High risk cohorts, and save results."""

    parser = argparse.ArgumentParser(
        description='Aggregate nucleus-level morphometric features to patient level.'
    )
    parser.add_argument(
        '--low_csv', type=str, required=True,
        help='Path to the per-nucleus feature CSV for the low-risk cohort.'
    )
    parser.add_argument(
        '--high_csv', type=str, required=True,
        help='Path to the per-nucleus feature CSV for the high-risk cohort.'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path to save the aggregated patient-level feature CSV.'
    )
    args = parser.parse_args()

    # -- Process each cohort --
    print("Aggregating nucleus-level features to patient level...")
    low_features  = process_cohort(args.low_csv,  label='Low')
    high_features = process_cohort(args.high_csv, label='High')

    # -- Combine both cohorts and save --
    combined = pd.concat([low_features, high_features], axis=0, ignore_index=True)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    combined.to_csv(args.output, index=False)
    print(f"Patient-level features saved to: {args.output}")
    print(f"  Total rows: {len(combined)}  |  "
          f"Total patients: {combined['Patient_Name'].nunique()}  |  "
          f"Feature columns: {len(FEATURE_COLUMNS)}")


if __name__ == '__main__':
    main()

