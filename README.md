# PROMISE

PROMISE (Proteomics-guided Multimodal Integration for Survival Estimation) is a proteomics-guided multimodal deep learning workflow for prognosis and risk stratification in surgically resected small cell lung cancer (SCLC). The framework learns from routine H&E-stained whole-slide images (WSIs) and paired quantitative proteomic profiles during model development, and is designed to support WSI-only risk prediction at inference time.

PROMISE consists of three conceptual components:

1. **Slide-level pathomics representation learning** from tiled H&E WSIs using pathology foundation-model features and gated-attention multiple-instance learning.
2. **Pathomics-proteomics cross-modal representation learning**, where WSI-derived representations are aligned with proteomic profiles through a WSI-to-protein prediction module.
3. **Proteomics-guided multimodal fusion**, where predicted proteomic representations guide patch-level histology aggregation for Cox-regression-based survival prediction.

This repository is organized as a practical research workflow rather than a single installable Python package. Most scripts are intended to be run from their own workflow directories after editing the corresponding configuration files and local data paths.

## Repository Layout

```text
PROMISE/
|-- WSI digitization and pre-processing/
|   |-- tiling_WSI_multi_thread.py
|   `-- get_foundation_model_features.py
|-- Proteomic data processing/
|   `-- proteomics_processing.R
|-- PROMISE architecture/
|   |-- Pathomics-proteomics cross-modal representation learning/
|   |   |-- Train_WSI_TO_Protein.py
|   |   |-- Val_WSI_TO_Protein.py
|   |   |-- config/
|   |   |-- models/
|   |   `-- utils/
|   `-- Proteomics-guided multimodal fusion/
|       |-- Train_MultiModal_Survival.py
|       |-- Eval_MultiModal_Survival.py
|       |-- config/
|       |-- models/
|       `-- utils/
|-- Downstream analysis/
|   |-- WSI level attention mapping/
|   |   |-- visualize_wsi_heatmap.py
|   |   |-- analyze_attention_distribution.py
|   |   |-- wsi_weight_heatmap_visualization.py
|   |   |-- key_patch_grid_visualization.py
|   |   `-- mismatch_region_patch_visualization.py
|   |-- Protein level attribution analysis/
|   |   `-- visualize_wsi_protein_interaction.py
|   |-- Cell level morphologic and textural profiling/
|   |   |-- Textural features/
|   |   `-- hover_net-master/
|   `-- Functional enrichment analysis/
`-- README.md
```

## Important Note

The repository reflects a research-code workflow used for model development, evaluation, and interpretability analysis. Large datasets, trained checkpoints, whole-slide image archives, proteomic matrices, and private clinical tables are not included.

Several directories such as `Datasets/`, `Log/`, `Result/`, `Inputdir/`, `Out_PutDir/`, and `MyModels/` are local working folders used during experimentation. They should be treated as runtime placeholders rather than mandatory public inputs.

Many scripts use relative paths or YAML configuration files. Before running a module, update the corresponding paths to match your local project structure. Because some directory names contain spaces, wrap paths in quotes when using shell commands.

## Environment

The workflow was developed around common scientific Python and R environments. A GPU-enabled PyTorch environment is recommended for model training and foundation-model feature extraction.

Typical Python dependencies include:

- `python >= 3.8`
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `seaborn`
- `tables`
- `pyyaml`
- `openslide-python`
- `imageio`
- `shapely`
- `pyarrow`
- `huggingface_hub`
- `timm`
- `transformers`

Additional model-specific dependencies may be required depending on the selected pathology foundation model:

- `CONCH`
- `CONCH_v1_5`
- `UNI`
- `UNI_v2`
- `Virchow2`

The proteomic preprocessing workflow is written in R. The manuscript workflow used R for proteomic preprocessing, functional enrichment, and statistical analysis. Typical R-side dependencies may include packages for missing-value imputation, enrichment analysis, survival analysis, and visualization.

## Workflow

### 01 WSI digitization and pre-processing

This stage converts raw diagnostic WSIs into quality-controlled image patches and then extracts patch-level features using a selected pathology foundation model.

#### 01.1 Tile whole-slide images

Raw WSIs should be organized in one input directory. Supported slide formats include common pathology image formats such as `.svs`, `.mrxs`, `.ndpi`, `.scn`, `.tif`, `.tiff`, `.jpg`, and `.jpeg`.

```text
dataset/
`-- WSIs/
    |-- patient_001_slide.svs
    |-- patient_002_slide.ndpi
    `-- patient_003_slide.tif
```

`tiling_WSI_multi_thread.py` reads WSIs, downsamples them to the target magnification, extracts non-overlapping or overlapping windows, applies Otsu-based tissue filtering, and exports 224 x 224-pixel patches by default.

```bash
python "WSI digitization and pre-processing/tiling_WSI_multi_thread.py" \
  --slide ./dataset/WSIs \
  --out ./dataset/patches \
  --px 224 \
  --target_mag 10 \
  --min_tissue_area 0.60 \
  --num_threads 6
```

Optional ROI annotation files can be used by enabling `--skipws`, which skips slides without sibling `.csv` or `.json` annotation files.

#### 01.2 Extract patch-level foundation-model features

After tiling, patch images should be organized as one folder per slide or as a folder tree that can be searched recursively.

```text
dataset/
`-- patches/
    |-- patient_001_slide/
    |   |-- patient_001_slide_(0,0).jpg
    |   |-- patient_001_slide_(224,0).jpg
    |   `-- ...
    `-- patient_002_slide/
```

`get_foundation_model_features.py` supports multiple pathology foundation models and writes one feature table per slide.

```bash
python "WSI digitization and pre-processing/get_foundation_model_features.py" \
  --model UNI \
  --patches_dir ./dataset/patches \
  --output_dir ./dataset/slide_features \
  --batch_size 64 \
  --device auto
```

Supported model names are:

```text
CONCH, CONCH_v1_5, UNI, UNI_v2, Virchow2
```

If a model requires Hugging Face authentication, provide a token through `--hf_token` or through environment variables such as `HF_TOKEN`.

The feature extraction script writes Feather files under:

```text
<output_dir>/<model>/<slide>.feather
```

The model-training configuration files in this repository expect cohort-level feature files such as `.pkl` files. If needed, aggregate or convert per-slide feature tables into the cohort-level format expected by the training datasets.

### 02 Proteomic data processing

This stage prepares the proteomic target matrix used for cross-modal learning. In the study workflow, matched FFPE samples were profiled using label-free LC-MS/MS quantitative proteomics. Raw proteomic measurements were median-centered, log2-transformed, filtered by missingness, and imputed.

The preprocessing script is provided as an R workflow template:

```bash
Rscript "Proteomic data processing/proteomics_processing.R"
```

A typical processed proteomic matrix should contain one patient or specimen per row and one protein feature per column. The modeling configuration in this repository assumes a 6,646-protein input by default, but the architecture dimensions can be modified in the YAML configuration files.

Example format:

```text
PatientID,Protein_0001,Protein_0002,Protein_0003,...
patient_001,8.31,5.22,2.14,...
patient_002,8.10,5.01,2.39,...
patient_003,7.94,5.17,2.21,...
```

### 03 Pathomics-proteomics cross-modal representation learning

This stage trains the WSI-to-protein module. It learns to map WSI-derived pathomics representations into a latent proteomic space, enabling proteomic representations to be imputed from histology features.

Before training, edit:

```text
PROMISE architecture/
`-- Pathomics-proteomics cross-modal representation learning/
    `-- config/
        `-- config.yaml
```

Key entries to update include:

- WSI feature paths, such as `CHCAMS_Discovery_Image`, `CHCAMS_External_Image`, `HMUCH_External_Image`, and `TMUGH_External_Image`.
- Clinical CSV paths, such as `CHCAMS_Discovery_Clincial`, `CHCAMS_Train_Clincial`, and `CHCAMS_Test_Clincial`.
- Proteomic matrix path, such as `CHCAMS_Protein`.
- Output paths for predicted protein representations, such as `Train_PreProtein` and `Test_PreProtein`.
- Checkpoint paths, including `initial_checkpoint`, `Locked_checkpoint`, and `Model_Out`.

Run training from inside the module directory so that the relative `./config/config.yaml` path resolves correctly:

```bash
cd "PROMISE architecture/Pathomics-proteomics cross-modal representation learning"

python Train_WSI_TO_Protein.py
```

The training script reads all default parameters from `config/config.yaml`. Any YAML key can also be overridden from the command line. For example:

```bash
python Train_WSI_TO_Protein.py \
  --batch_size 8 \
  --start_lr 5e-4 \
  --Epoch 10000
```

After training, run validation or inference to generate predicted protein representations for the discovery and external cohorts:

```bash
python Val_WSI_TO_Protein.py
```

The validation script standardizes predicted protein features using statistics fitted on the training cohort, then applies the same scaling to validation and external cohorts.

### 04 Proteomics-guided multimodal survival modeling

This stage trains the final PROMISE survival model. The model fuses:

1. Gated-attention-pooled WSI features.
2. Protein-guided cross-attention WSI features.
3. Predicted proteomic representations.

The fused representation is used to estimate patient-level hazard scores through a Cox-regression survival objective.

Before training, edit:

```text
PROMISE architecture/
`-- Proteomics-guided multimodal fusion/
    `-- config/
        `-- config.yaml
```

Important configuration fields include:

- WSI feature paths.
- Predicted protein CSV paths from the WSI-to-protein stage.
- Clinical tables with survival labels.
- Training hyperparameters.
- Checkpoint paths.

Run training from inside the multimodal-fusion directory:

```bash
cd "PROMISE architecture/Proteomics-guided multimodal fusion"

python Train_MultiModal_Survival.py
```

The training script evaluates the model during training using survival metrics such as C-index and log-rank-based risk separation.

Run evaluation and patch-weight extraction with:

```bash
python Eval_MultiModal_Survival.py
```

The evaluation script can generate:

- Patient-level hazard predictions.
- Discovery train/test evaluation results.
- External-cohort evaluation results.
- Patch-level attention weights for downstream WSI interpretability.

Patch-weight outputs are written to local log paths such as:

```text
Visual/Log/Patchweight_Train.csv
Visual/Log/Patchweight_Test.csv
```

Depending on the selected model variant, patch-weight files may include both MIL attention weights and protein-WSI interaction weights.

### 05 WSI-level attention mapping

The WSI-level attention mapping scripts reconstruct WSI-like patch mosaics from HDF5 patch archives and project patch-level attention weights back to their spatial coordinates.

#### 05.1 Generate WSI attention heatmaps

```bash
python "Downstream analysis/WSI level attention mapping/visualize_wsi_heatmap.py" \
  --h5d_image ./dataset/H5D/CHCAMS_Discovery.hdf5 \
  --h5d_csv ./dataset/H5D/CHCAMS_Discovery.csv \
  --weight_csv ./Visual/Log/Patchweight_Train.csv \
  --train_csv ./Log/Result_Train.csv \
  --test_csv ./Log/Result_Test.csv \
  --threshold 0.4632 \
  --output_dir ./Result/wsi_heatmap
```

To visualize one patient only:

```bash
python "Downstream analysis/WSI level attention mapping/visualize_wsi_heatmap.py" \
  --h5d_image ./dataset/H5D/CHCAMS_Discovery.hdf5 \
  --h5d_csv ./dataset/H5D/CHCAMS_Discovery.csv \
  --weight_csv ./Visual/Log/Patchweight_Train.csv \
  --train_csv ./Log/Result_Train.csv \
  --patient_id 100001 \
  --output_dir ./Result/wsi_heatmap
```

#### 05.2 Analyze attention-weight distributions

```bash
python "Downstream analysis/WSI level attention mapping/analyze_attention_distribution.py" \
  --h5d_image ./dataset/H5D/CHCAMS_Discovery.hdf5 \
  --h5d_csv ./dataset/H5D/CHCAMS_Discovery.csv \
  --weight_csv ./Visual/Log/Patchweight_Train.csv \
  --hazard_csv ./Log/Result_Train.csv \
  --threshold 0.6585 \
  --output_dir ./Result/attention_distribution
```

This module generates summary plots of patch-level attention weights, including histograms, density curves, percentile thresholds, and descriptive statistics.

#### 05.3 Export high- and low-attention patch grids

```bash
python "Downstream analysis/WSI level attention mapping/key_patch_grid_visualization.py" \
  --patient-id 100001 \
  --h5d-image-path ./dataset/H5D/CHCAMS_Discovery.hdf5 \
  --h5d-csv-path ./dataset/H5D/CHCAMS_Discovery.csv \
  --wsi-weight-train ./Visual/Log/Patchweight_WSIModel_Train.csv \
  --wsi-weight-test ./Visual/Log/Patchweight_WSIModel_Test.csv \
  --multi-weight-train ./Visual/Log/Patchweight_MultiModel_Train.csv \
  --multi-weight-test ./Visual/Log/Patchweight_MultiModel_Test.csv \
  --output-dir ./Result/key_patch_grids
```

This script exports patch montages for high- and low-weight regions from both WSI-only and multimodal models.

#### 05.4 Visualize model-discordant WSI regions

```bash
python "Downstream analysis/WSI level attention mapping/mismatch_region_patch_visualization.py" \
  --patient-id 100001 \
  --h5d-image-path ./dataset/H5D/CHCAMS_Discovery.hdf5 \
  --h5d-csv-path ./dataset/H5D/CHCAMS_Discovery.csv \
  --wsi-weight-train ./Visual/Log/Patchweight_WSIModel_Train.csv \
  --wsi-weight-test ./Visual/Log/Patchweight_WSIModel_Test.csv \
  --multi-weight-train ./Visual/Log/Patchweight_MultiModel_Train.csv \
  --multi-weight-test ./Visual/Log/Patchweight_MultiModel_Test.csv \
  --output-dir ./Result/wsi_mismatch_regions
```

This script compares WSI-only and multimodal patch weights, computes their difference, and uses a KDTree-based circular neighborhood search to identify local regions with the largest positive and negative weight shifts.

#### 05.5 Generate WSI-only model heatmaps

```bash
python "Downstream analysis/WSI level attention mapping/wsi_weight_heatmap_visualization.py" \
  --patient-id 100001 \
  --h5d-image-path ./dataset/H5D/CHCAMS_Discovery.hdf5 \
  --h5d-csv-path ./dataset/H5D/CHCAMS_Discovery.csv \
  --wsi-weight-train ./Visual/Log/Patchweight_WSIModel_Train.csv \
  --wsi-weight-test ./Visual/Log/Patchweight_WSIModel_Test.csv \
  --analysis-dir ./Analysis \
  --output-dir ./Result/wsi_weight_heatmap
```

This module generates side-by-side WSI patch mosaics and WSI-model attention heatmaps.

### 06 Protein-level attribution and protein-conditioned WSI visualization

Protein-level attribution analysis is used to connect PROMISE-derived risk stratification with protein-specific or pathway-specific histologic patterns. The provided script reconstructs WSI patch mosaics and generates correlation-derived patch-weight maps conditioned on selected protein or pathway signals.

```bash
python "Downstream analysis/Protein level attribution analysis/visualize_wsi_protein_interaction.py"
```

Before running this script, update the configuration constants at the top of the file, including:

- HDF5 patch archive path.
- HDF5 patch CSV path.
- Patch-weight CSV path.
- Proteomic expression matrix path.
- Pathway-to-protein mapping file.
- Contrastive-learning output files.
- Output directory.

This analysis can be used to visualize how immune-related, metabolic, adhesion, migration, or other protein-associated programs localize to distinct histologic compartments.

### 07 Cell-level morphologic and textural profiling

This stage supports cell-level interpretation of PROMISE-highlighted WSI regions. The repository includes a HoVer-Net-based segmentation workflow and a downstream aggregation script for morphometric, textural, and intensity features.

A typical pipeline is:

1. Run HoVer-Net nucleus segmentation and cell-type classification on selected WSI patches.
2. Extract nucleus-level morphology, texture, and intensity features.
3. Aggregate nucleus-level features to patient-level summaries by cell type.

The aggregation script expects one per-nucleus CSV for low-risk regions and one per-nucleus CSV for high-risk regions:

```bash
python "Downstream analysis/Cell level morphologic and textural profiling/Textural features/Main.py" \
  --low_csv ./Result/cell_features_low.csv \
  --high_csv ./Result/cell_features_high.csv \
  --output ./Result/patient_level_cell_features.csv
```

The output contains one row per patient and cell type, including:

- Cell-type proportion.
- Mean morphometric features.
- Mean GLCM textural features.
- Mean first-order intensity features.
- Risk-group label.

## Expected Input Tables

### Clinical survival table

The modeling scripts expect clinical tables linking patients to OS and DFS outcomes. Column names should be consistent with the dataset loaders and may include:

```text
PatientID,OS,OSState,DFS,DFSState
patient_001,54,0,54,0
patient_002,17,1,12,1
patient_003,36,0,24,1
```

### Proteomic matrix

The proteomic matrix should contain patient identifiers and protein-level measurements. The default architecture assumes 6,646 protein features, matching the study preprocessing workflow.

```text
PatientID,Protein_0001,Protein_0002,Protein_0003,...
patient_001,8.31,5.22,2.14,...
patient_002,8.10,5.01,2.39,...
patient_003,7.94,5.17,2.21,...
```

### Patch feature table

Patch-level feature tables should preserve patch identifiers that encode slide or patient identity and spatial coordinates.

```text
patch_name                         feat_0001  feat_0002  ...  feat_1024
patient_001_(0,0).jpg              0.1532    -0.2910    ...  0.0084
patient_001_(224,0).jpg            0.1321    -0.1442    ...  0.0213
patient_001_(448,0).jpg            0.0954    -0.1887    ...  0.0149
```

### HDF5 patch archive

Several visualization scripts expect an HDF5 patch archive with a `patches` node and a corresponding CSV file containing `Patch_Name` values.

```text
dataset/
`-- H5D/
    |-- CHCAMS_Discovery.hdf5
    `-- CHCAMS_Discovery.csv
```

## Module Guide

### WSI digitization and pre-processing

- `tiling_WSI_multi_thread.py`  
  Tiles WSIs into 224 x 224-pixel patches at the target magnification, applies Otsu tissue filtering, and supports multi-threaded slide processing.

- `get_foundation_model_features.py`  
  Extracts patch-level embeddings using selected pathology foundation models and writes slide-level Feather feature files.

### Proteomic data processing

- `proteomics_processing.R`  
  Provides a template for proteomic preprocessing, including normalization, missingness filtering, and imputation before model development.

### PROMISE architecture / Pathomics-proteomics cross-modal representation learning

- `Train_WSI_TO_Protein.py`  
  Trains the WSI-to-protein representation learning module using paired WSI and proteomic data.

- `Val_WSI_TO_Protein.py`  
  Runs inference with a trained WSI-to-protein checkpoint and exports standardized predicted protein representations for training, validation, and external cohorts.

- `models/Model_WSI_TO_Protein.py`  
  Defines the cross-modal network used to align WSI-derived features with proteomic representations.

- `models/dataset_WSI_TO_Protein.py`  
  Builds cohort datasets for WSI-to-protein training and validation.

### PROMISE architecture / Proteomics-guided multimodal fusion

- `Train_MultiModal_Survival.py`  
  Trains the final PROMISE survival model with WSI features and predicted protein representations.

- `Eval_MultiModal_Survival.py`  
  Evaluates PROMISE on discovery and external cohorts and extracts patch-level attention weights for interpretability.

- `models/model_multimodel_Pre.py`  
  Defines the predicted-protein-guided multimodal survival network with gated MIL attention, cross-attention, and Cox survival loss.

- `models/Attention.py`  
  Implements gated-attention pooling for multiple-instance learning over WSI patches.

- `models/DimReduction.py`  
  Projects high-dimensional patch features into the latent space used by the fusion model.

- `models/Survival.py` and `utils/Survival.py`  
  Provide survival-analysis utilities such as Cox loss, C-index, and log-rank-related evaluation.

### Downstream analysis / WSI level attention mapping

- `visualize_wsi_heatmap.py`  
  Reconstructs WSI patch mosaics and overlays model attention weights as heatmaps.

- `analyze_attention_distribution.py`  
  Summarizes the distribution of patch-level attention weights across patient risk groups.

- `wsi_weight_heatmap_visualization.py`  
  Generates WSI-model attention heatmaps for selected patients.

- `key_patch_grid_visualization.py`  
  Exports high- and low-attention patch montages for WSI-only and multimodal models.

- `mismatch_region_patch_visualization.py`  
  Identifies local WSI regions where multimodal and WSI-only models differ most strongly in patch-level attention.

### Downstream analysis / Protein level attribution analysis

- `visualize_wsi_protein_interaction.py`  
  Generates protein- or pathway-conditioned WSI visualizations by linking protein-associated signals with patch-level histologic patterns.

### Downstream analysis / Cell level morphologic and textural profiling

- `hover_net-master/`  
  Contains the HoVer-Net-based nucleus segmentation and cell-type classification workflow used before cell-level feature analysis.

- `Textural features/Main.py`  
  Aggregates nucleus-level morphometric, textural, and intensity descriptors to patient-level feature tables by cell type and risk group.

### Downstream analysis / Functional enrichment analysis

This directory is intended for functional interpretation of PROMISE-associated proteins and pathway programs. In the study workflow, enrichment analyses were used to compare biological programs associated with low- and high-risk PROMISE groups.

## Practical Tips

- Run training scripts from their own module directories because they load `./config/config.yaml` using a relative path.
- Update all placeholder paths such as `<DATA_ROOT>` and `<RESULT_ROOT>` before running.
- Use consistent patient identifiers across clinical tables, proteomic matrices, patch names, and feature files.
- Keep train/test/external cohort definitions fixed after generating the WSI-to-protein predictions so that downstream survival modeling uses aligned inputs.
- Remove local caches, large private data, `.pyc` files, and experiment-specific logs before making a public GitHub release.
- For GitHub readability, consider renaming directories with spaces to underscore-separated names in a future cleanup, or keep quotes around paths in all shell commands.

## Citation

If you use PROMISE in academic work, please cite the associated study:

**Proteomics-guided interpretable multimodal AI model for prognosis and risk stratification in surgically resected small cell lung cancer.**

A formal citation can be added here once the manuscript or preprint is publicly available.
