# =============================================================================
# Inference_WSI_TO_Protein.py
#
# Inference pipeline for the WSI-to-Protein multi-modal model.
# This script loads a trained checkpoint, runs forward inference on all
# cohorts (discovery train/test + external validation), standardizes the
# predicted protein features using the training set statistics, and saves
# the results to CSV files.
#
# Usage:
#   python Inference_WSI_TO_Protein.py --config ./config/config.yaml
# =============================================================================

# ---- Standard library ----
import math
import random
import warnings

warnings.filterwarnings('ignore')

# ---- Third-party libraries ----
import numpy as np
import pandas as pd
import yaml
import argparse

import torch
import torch.optim
import torch.cuda.amp as amp
import torchvision
import tables
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---- Project-specific modules ----
from dataset_WSI_TO_Protein import *
from models import resnet, Model_WSI_TO_Protein
import Val_Data_WSI_TO_Protein
import Survival
from utils import yaml_config_hook, save_model

# Global flag for automatic mixed precision
is_amp = True


# =============================================================================
# Configuration Utilities
# =============================================================================

def yaml_config_hook(config_file):
    """Load a YAML configuration file and return a dictionary.

    Args:
        config_file: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration key-value pairs.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_learning_rate(optimizer):
    """Retrieve the current learning rate from the optimizer.

    Args:
        optimizer: PyTorch optimizer instance.

    Returns:
        Current learning rate (float).
    """
    return optimizer.param_groups[0]['lr']


# =============================================================================
# Batch Initialization
# =============================================================================

def initialize(batch):
    """Transfer all tensors in a batch dictionary to GPU (CUDA).

    Moves 'feature', 'OS', 'DFS', 'OSState', 'DFSState' to CUDA.
    'Protein' is moved only when it is not None.

    Args:
        batch: Dictionary produced by the DataLoader / collate function.

    Returns:
        The same batch dictionary with tensors on CUDA.
    """
    batch['feature']  = batch['feature'].cuda()
    batch['OS']       = batch['OS'].cuda()
    batch['DFS']      = batch['DFS'].cuda()
    batch['OSState']  = batch['OSState'].cuda()
    batch['DFSState'] = batch['DFSState'].cuda()

    if batch['Protein'] != None:
        batch['Protein'] = batch['Protein'].cuda()
    else:
        batch['Protein'] = batch['Protein']

    return batch


# =============================================================================
# Inference (Validation)
# =============================================================================

def Model_Val(net, dataset_, args):
    """Run inference on a given dataset and collect per-patient predictions.

    The model is set to eval mode with 'inference' output only (no loss
    computation). For every sample the function records clinical outcomes
    (OS, DFS) and the WSI-predicted protein expression vector.

    Args:
        net:      The MultiModal network instance (already on GPU).
        dataset_: PyTorch Dataset to evaluate.
        args:     Namespace with hyperparameters (e.g., workers).

    Returns:
        My_Final_Result: DataFrame whose first 5 columns are
                         [Patient, OS, OSState, DFS, DFSState] followed by
                         the predicted protein feature columns.
    """
    # -- Build DataLoader (batch_size=1, no shuffle for deterministic order) --
    loader = DataLoader(
        dataset_,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=int(args.workers),    # Set to 0 for debugging
        pin_memory=False,
        collate_fn=null_collate
    )

    # -- Switch model to evaluation / inference mode --
    net.eval()
    net.output_type = ['inference']

    results = []
    List_Patient_Protein = []
    Loss = 0

    # -- Iterate over all samples --
    for t, batch in enumerate(loader):
        batch = initialize(batch)

        with torch.no_grad():
            output = net(batch)

            # Collect clinical and survival information for this patient
            result = {
                'Patient':  batch['patient_name'],                            # Patient identifier
                'OS':       batch['OS'].detach().cpu().numpy().squeeze(),      # Overall survival time
                'OSState':  batch['OSState'].detach().cpu().numpy().squeeze(), # OS event indicator
                'DFS':      batch['DFS'].detach().cpu().numpy().squeeze(),     # Disease-free survival
                'DFSState': batch['DFSState'].detach().cpu().numpy().squeeze(),# DFS event indicator
            }

            results.append(pd.DataFrame(result))
            List_Patient_Protein.append(
                output['WSI_Protein'].detach().cpu().numpy().squeeze()
            )

    # -- Aggregate patient-level results into a single DataFrame --
    MyResult        = pd.concat(results, ignore_index=True)
    MyProtein       = pd.DataFrame(List_Patient_Protein)
    My_Final_Result = pd.concat([MyResult, MyProtein], axis=1)

    return My_Final_Result


# =============================================================================
# Main Inference Pipeline
# =============================================================================

if __name__ == '__main__':

    # -- Set global reproducibility seed --
    set_seed(42)

    # -----------------------------------------------------------------
    # 1. Parse hyperparameters from YAML config + command-line overrides
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # Extract key hyperparameters for readability
    out_dir            = args.Model_Out
    initial_checkpoint = args.Locked_checkpoint
    start_lr           = float(args.start_lr)
    batch_size         = int(args.batch_size)
    Epoch              = int(args.Epoch)

    # -----------------------------------------------------------------
    # 2. Prepare validation datasets for all cohorts
    # -----------------------------------------------------------------
    Cohorts, dataset_val = Val_Data_WSI_TO_Protein.Discovery_Cohort(args)
    args.dim = Cohorts['dim']

    # -----------------------------------------------------------------
    # 3. Build model and move to GPU
    # -----------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net    = Model_WSI_TO_Protein.MultiModal(arg=args).to(device)

    # -----------------------------------------------------------------
    # 4. Load trained checkpoint
    # -----------------------------------------------------------------
    if initial_checkpoint != 'None':
        f          = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict, strict=False)
    else:
        start_iteration = 0
        start_epoch     = 0

    # -----------------------------------------------------------------
    # 5. Run inference on the discovery training set
    #    Fit a StandardScaler on the training predictions so that all
    #    cohorts are standardized with the same statistics.
    # -----------------------------------------------------------------
    Train_PreProtein = Model_Val(net, dataset_val['train_dataset'], args)

    # Fit StandardScaler on predicted protein features (columns 5 onward)
    scaler = StandardScaler()
    Train_Features = Train_PreProtein.values[:, 5:]
    Train_Features_scaled = scaler.fit_transform(Train_Features)
    Train_PreProtein.iloc[:, 5:] = Train_Features_scaled

    # -----------------------------------------------------------------
    # 6. Run inference on the discovery test set
    #    Apply the SAME scaler fitted on the training set.
    # -----------------------------------------------------------------
    Test_PreProtein = Model_Val(net, dataset_val['test_dataset'], args)
    Test_PreProtein.iloc[:, 5:] = scaler.transform(
        Test_PreProtein.values[:, 5:]
    )

    # -----------------------------------------------------------------
    # 7. Run inference on external validation cohorts
    #    All external cohorts are standardized with the training scaler.
    # -----------------------------------------------------------------

    # -- HMUCH external cohort --
    HMUCH_PreProtein = Model_Val(net, dataset_val['HMUCH_dataset'], args)
    HMUCH_PreProtein.iloc[:, 5:] = scaler.transform(
        HMUCH_PreProtein.values[:, 5:]
    )

    # -- TMUGH external cohort --
    TMUGH_PreProtein = Model_Val(net, dataset_val['TMUGH_dataset'], args)
    TMUGH_PreProtein.iloc[:, 5:] = scaler.transform(
        TMUGH_PreProtein.values[:, 5:]
    )

    # -- CHCAMS external cohort --
    CHCAMS_PreProtein = Model_Val(net, dataset_val['CHCAMS_External_dataset'], args)
    CHCAMS_PreProtein.iloc[:, 5:] = scaler.transform(
        CHCAMS_PreProtein.values[:, 5:]
    )

    # -----------------------------------------------------------------
    # 8. Save predicted protein results to CSV
    #    Uncomment the lines below to export each cohort's predictions.
    # -----------------------------------------------------------------
    # Train_PreProtein.to_csv(args.Train_PreProtein, index=False)
    # Test_PreProtein.to_csv(args.Test_PreProtein, index=False)
    # HMUCH_PreProtein.to_csv(args.HMUCH_PreProtein, index=False)
    # TMUGH_PreProtein.to_csv(args.TMUGH_PreProtein, index=False)
    # CHCAMS_PreProtein.to_csv(args.CHCAMS_PreProtein, index=False)

