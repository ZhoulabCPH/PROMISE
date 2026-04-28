# =============================================================================
# Eval_MultiModal_Survival.py
#
# Evaluation and visualization pipeline for the multi-modal survival model.
# This script loads a trained checkpoint, runs inference on all cohorts
# (discovery train/test + external validation), computes the concordance
# index (C-Index) and log-rank p-value for each cohort, and optionally
# exports per-patch attention weights for interpretability analysis.
#
# Usage:
#   python Eval_MultiModal_Survival.py --config ./config/config.yaml
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

# ---- Project-specific modules ----
from dataset import *
from models import resnet, model_multimodel_Pre
import Val_Data
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

    Moves 'feature', 'OS', 'DFS', 'OSState', 'DFSState', 'PreProtein' to
    CUDA. 'Protein' is moved only when it is not None.

    Args:
        batch: Dictionary produced by the DataLoader / collate function.

    Returns:
        The same batch dictionary with tensors on CUDA.
    """
    batch['feature']    = batch['feature'].cuda()
    batch['OS']         = batch['OS'].cuda()
    batch['DFS']        = batch['DFS'].cuda()
    batch['OSState']    = batch['OSState'].cuda()
    batch['DFSState']   = batch['DFSState'].cuda()
    batch['PreProtein'] = batch['PreProtein'].cuda()

    if batch['Protein'] != None:
        batch['Protein'] = batch['Protein'].cuda()
    else:
        batch['Protein'] = batch['Protein']

    return batch


# =============================================================================
# Survival Evaluation
# =============================================================================

def Model_Val(net, dataset_, args):
    """Run inference and compute survival metrics on a given dataset.

    For each patient the function collects clinical outcomes and the predicted
    hazard score. After iterating over all samples it computes the C-Index and
    the log-rank p-value.

    Args:
        net:      The MultiModal network instance (already on GPU).
        dataset_: PyTorch Dataset to evaluate.
        args:     Namespace with hyperparameters (e.g., workers).

    Returns:
        MyResult: DataFrame with columns [Patient, OS, OSState, DFS, DFSState, hazards].
        CIndex:   Concordance index for OS.
        Pvalues:  Log-rank p-value for OS.
    """
    loader = DataLoader(
        dataset_,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=int(args.workers),    # Set to 0 for debugging
        pin_memory=False,
        collate_fn=null_collate
    )

    net.eval()
    net.output_type = ['inference']

    results = []
    for t, batch in enumerate(loader):
        batch = initialize(batch)

        with torch.no_grad():
            output = net(batch)

            result = {
                'Patient':  batch['patient_name'],                            # Patient identifier
                'OS':       batch['OS'].detach().cpu().numpy().squeeze(),      # Overall survival time
                'OSState':  batch['OSState'].detach().cpu().numpy().squeeze(), # OS event indicator
                'DFS':      batch['DFS'].detach().cpu().numpy().squeeze(),     # Disease-free survival
                'DFSState': batch['DFSState'].detach().cpu().numpy().squeeze(),# DFS event indicator
                'hazards':  output['hazards'].detach().cpu().numpy().squeeze() # Predicted hazard
            }
            results.append(pd.DataFrame(result))

    MyResult = pd.concat(results, ignore_index=True)

    # Compute C-Index and log-rank p-value for overall survival
    CIndex  = Survival.CIndex(
        hazards=MyResult['hazards'],
        labels=MyResult['OSState'],
        survtime_all=MyResult['OS']
    )
    Pvalues = Survival.cox_log_rank(
        hazardsdata=MyResult['hazards'],
        labels=MyResult['OSState'],
        survtime_all=MyResult['OS']
    )

    return MyResult, CIndex, Pvalues


# =============================================================================
# Patch-Level Attention Weight Extraction
# =============================================================================

def Model_Val_PatchWeight(net, dataset_, args):
    """Extract per-patch attention weights for interpretability / visualization.

    The function checks the checkpoint name to determine whether the model is
    a full MultiModal variant (which outputs both MIL and cross-attention
    weights) or a simpler model (which outputs MIL weights only).

    Args:
        net:      The MultiModal network instance (already on GPU).
        dataset_: PyTorch Dataset to evaluate.
        args:     Namespace with hyperparameters.

    Returns:
        MyResult: DataFrame with columns [Patch_name, Patch_weight] and
                  optionally [InterProteinWSI_weight] for full MultiModal.
    """
    loader = DataLoader(
        dataset_,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=int(args.workers),    # Set to 0 for debugging
        pin_memory=False,
        collate_fn=null_collate
    )

    net.eval()
    net.output_type = ['inference']
    results = []

    # Determine model variant from checkpoint filename
    is_multimodel = (
        args.initial_checkpoint.split("_")[-1].split(".")[0] == 'MultiModel'
    )

    if is_multimodel:
        # Full MultiModal: extract both MIL and cross-attention weights
        for t, batch in enumerate(loader):
            batch = initialize(batch)
            with torch.no_grad():
                output = net(batch)
                result = {
                    'Patch_name':           batch['patch_name'][0],        # Patch file identifier
                    'Patch_weight':         output['patch_weight'],        # MIL attention weight
                    'InterProteinWSI_weight': output['InterProteinWSI_weight']  # Cross-attention weight
                }
                results.append(pd.DataFrame(result))
    else:
        # Other models: extract MIL attention weights only
        for t, batch in enumerate(loader):
            batch = initialize(batch)
            with torch.no_grad():
                output = net(batch)
                result = {
                    'Patch_name':   batch['patch_name'][0],               # Patch file identifier
                    'Patch_weight': output['patch_weight']                # MIL attention weight
                }
                results.append(pd.DataFrame(result))

    MyResult = pd.concat(results, ignore_index=True)
    return MyResult


# =============================================================================
# Main Evaluation Pipeline
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

    out_dir            = args.Model_Out
    initial_checkpoint = args.initial_checkpoint
    start_lr           = float(args.start_lr)
    batch_size         = int(args.batch_size)
    Epoch              = int(args.Epoch)

    # -----------------------------------------------------------------
    # 2. Prepare validation datasets for all cohorts
    # -----------------------------------------------------------------
    Cohorts, dataset_val = Val_Data.Discovery_Cohort(args)
    args.dim = Cohorts['dim']

    # -----------------------------------------------------------------
    # 3. Build model and move to GPU
    #    Switch between model variants by changing the import below:
    #      model_multimodel_Pre  – Predicted-protein + WSI multi-modal
    #      model_multimodel      – True-protein + WSI multi-modal
    #      model_WSI             – WSI-only baseline
    #      model_Protein         – Protein-only baseline
    # -----------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net    = model_multimodel_Pre.MultiModal(arg=args).to(device)

    # -----------------------------------------------------------------
    # 4. Load trained checkpoint
    # -----------------------------------------------------------------
    if initial_checkpoint != 'None':
        f           = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict, strict=False)
    else:
        start_iteration = 0
        start_epoch     = 0

    # -----------------------------------------------------------------
    # 5. Evaluate survival on discovery cohort (train / test)
    # -----------------------------------------------------------------
    Train_External_Result, Train_External_CIndex, Train_External_Pvalues = \
        Model_Val(net, dataset_val['train_dataset'], args)

    Test_External_Result, Test_External_CIndex, Test_External_Pvalues = \
        Model_Val(net, dataset_val['test_dataset'], args)

    # -----------------------------------------------------------------
    # 6. Extract patch-level attention weights for visualization
    # -----------------------------------------------------------------
    Train_patchweight = Model_Val_PatchWeight(net, dataset_val['train_dataset'], args)
    Train_patchweight.to_csv("./Visual/Log/Patchweight_Train.csv", index=None)

    Test_patchweight = Model_Val_PatchWeight(net, dataset_val['test_dataset'], args)
    Test_patchweight.to_csv("./Visual/Log/Patchweight_Test.csv", index=None)

    # -----------------------------------------------------------------
    # 7. Evaluate survival on external validation cohorts
    # -----------------------------------------------------------------

    # -- HMUCH external cohort --
    HMUCH_External_Result, HMUCH_External_CIndex, HMUCH_External_Pvalues = \
        Model_Val(net, dataset_val['HMUCH_dataset'], args)

    # -- TMUGH external cohort --
    TMUGH_External_Result, TMUGH_External_CIndex, TMUGH_External_Pvalues = \
        Model_Val(net, dataset_val['TMUGH_dataset'], args)

    # -- CHCAMS external cohort --
    CHCAMS_External_Result, CHCAMS_External_CIndex, CHCAMS_External_Pvalues = \
        Model_Val(net, dataset_val['CHCAMS_External_dataset'], args)

    # -----------------------------------------------------------------
    # 8. Extract patch weights for external cohorts (optional)
    # -----------------------------------------------------------------
    # Train_patchweight = Model_Val_PatchWeight(net, dataset_val['train_dataset'], args)
    # Train_patchweight.to_csv("./Visual/Log/Patchweight_PreMultiModel_Train.csv", index=None)
    # Test_patchweight = Model_Val_PatchWeight(net, dataset_val['test_dataset'], args)
    # Test_patchweight.to_csv("./Visual/Log/Patchweight_PreMultiModel_Test.csv", index=None)

    # -----------------------------------------------------------------
    # 9. Save evaluation results (uncomment as needed)
    # -----------------------------------------------------------------
    # Train_External_Result.to_csv("./Log/Result_Train.csv", index=None)
    # Test_External_Result.to_csv("./Log/Result_Test.csv", index=None)
    # HMUCH_External_Result.to_csv("./Log/Result_HMUCH.csv", index=None)
    # TMUGH_External_Result.to_csv("./Log/Result_TMUGH.csv", index=None)
    # CHCAMS_External_Result.to_csv("./Log/Result_CHCAMS_External.csv", index=None)

