# =============================================================================
# Train_MultiModal_Survival.py
#
# Training pipeline for the multi-modal survival prediction model.
# The model fuses WSI patch features with predicted protein expression to
# predict patient hazard scores optimized via Cox partial-likelihood loss.
#
# During training the script periodically evaluates the concordance index
# (C-Index) and log-rank p-value on the discovery train/test split and all
# external validation cohorts, and saves checkpoints every epoch.
#
# Usage:
#   python Train_MultiModal_Survival.py --config ./config/config.yaml
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
import Train_Data
import Survival
from utils import yaml_config_hook, save_model

# Global flag for automatic mixed precision training
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
# Validation
# =============================================================================

def Model_Val(net, dataset_, args):
    """Run inference and compute survival metrics on a given dataset.

    For each patient the function collects clinical outcomes and the predicted
    hazard score, then computes the C-Index and log-rank p-value.

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
# Main Training Loop
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
    # 2. Prepare datasets (train loader + validation splits)
    # -----------------------------------------------------------------
    train_loader, Cohorts, dataset_val = Train_Data.Discovery_Cohort(args)
    args.dim = Cohorts['dim']

    # -----------------------------------------------------------------
    # 3. Build model and move to GPU
    # -----------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net    = model_multimodel_Pre.MultiModal(arg=args).to(device)

    # -----------------------------------------------------------------
    # 4. Optionally resume from a saved checkpoint
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
    # 5. Set up optimizer and learning rate scheduler
    # -----------------------------------------------------------------
    num_iteration = Epoch * len(train_loader)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=start_lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # -----------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------
    for i in range(Epoch):

        # ---- Train for one epoch ----
        for t, batch in enumerate(train_loader):
            rate       = get_learning_rate(optimizer)
            batch_size = len(batch['index'])
            batch      = initialize(batch)

            # Forward pass
            net.train()
            net.output_type = ['loss', 'inference']
            output = net(batch)
            loss   = output['All_Loss'].mean()

            # Backward pass with mixed-precision scaling
            optimizer.zero_grad()
            current_lr = lr_scheduler.get_last_lr()[0]
            optimizer.param_groups[0].update(lr=current_lr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        # ---- Periodic validation (every epoch after epoch 1) ----
        if i > 1 and i % 1 == 0 and i != 0:

            # Evaluate on discovery train / test split
            Train_External_Result, Train_External_CIndex, Train_External_Pvalues = \
                Model_Val(net, dataset_val['train_dataset'], args)

            Test_External_Result, Test_External_CIndex, Test_External_Pvalues = \
                Model_Val(net, dataset_val['test_dataset'], args)

            # Compute DFS C-Index and p-value on the test set
            Test_DFS_CIndex = Survival.CIndex(
                hazards=Test_External_Result['hazards'],
                labels=Test_External_Result['DFSState'],
                survtime_all=Test_External_Result['DFS']
            )
            Test_DFS_Pvalues = Survival.cox_log_rank(
                hazardsdata=Test_External_Result['hazards'],
                labels=Test_External_Result['DFSState'],
                survtime_all=Test_External_Result['DFS']
            )

            # Evaluate on external validation cohorts
            CHCAMS_External_Result, CHCAMS_External_CIndex, CHCAMS_External_Pvalues = \
                Model_Val(net, dataset_val['CHCAMS_External_dataset'], args)

            HMUCH_External_Result, HMUCH_External_CIndex, HMUCH_External_Pvalues = \
                Model_Val(net, dataset_val['HMUCH_dataset'], args)

            TMUGH_External_Result, TMUGH_External_CIndex, TMUGH_External_Pvalues = \
                Model_Val(net, dataset_val['TMUGH_dataset'], args)

            # Print C-Index and p-value for all cohorts
            print(
                'Epoch:[{}/{}]'
                '\tTrain_CIndex:{},Pvalue:{}'
                '\tTest_CIndex:{},Pvalue:{}'
                '\tTest_CIndex_DFS:{},Pvalue:{}'
                '\tCHCAMS_External_CIndex:{},Pvalue:{}'
                '\tTMUGH_CIndex:{},Pvalue:{}'
                '\tHMUCH_CIndex:{},Pvalue:{}'
                .format(
                    i, Epoch,
                    str(round(Train_External_CIndex, 6)),
                    str(round(Train_External_Pvalues, 6)),
                    str(round(Test_External_CIndex, 6)),
                    str(round(Test_External_Pvalues, 6)),
                    str(round(Test_DFS_CIndex, 6)),
                    str(round(Test_DFS_Pvalues, 6)),
                    str(round(CHCAMS_External_CIndex, 6)),
                    str(round(CHCAMS_External_Pvalues, 6)),
                    str(round(TMUGH_External_CIndex, 6)),
                    str(round(TMUGH_External_Pvalues, 6)),
                    str(round(HMUCH_External_CIndex, 6)),
                    str(round(HMUCH_External_Pvalues, 6))
                )
            )

            # Save checkpoint
            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'epoch': i,
                },
                out_dir + '/10X_UNI_Model/UNI_10X_PreMultiModel_{}.pth'.format(str(i))
            )

        # ---- Step the learning rate scheduler and clean up GPU memory ----
        lr_scheduler.step()
        start_epoch = i
        torch.cuda.empty_cache()

