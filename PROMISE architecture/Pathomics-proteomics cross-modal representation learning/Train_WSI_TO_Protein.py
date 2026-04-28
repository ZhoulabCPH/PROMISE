# =============================================================================
# Train_WSI_TO_Protein.py
#
# Training and validation pipeline for the WSI-to-Protein multi-modal model.
# The script reads hyperparameters from a YAML config, builds the dataset and
# model, trains with mixed-precision (AMP) and cosine annealing LR, and
# periodically validates / saves checkpoints.
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
from dataset_WSI_TO_Protein import *
from models import Model_WSI_TO_Protein
import Train_Data_WSI_TO_Protein
import Survival
from utils import yaml_config_hook, save_model


# =============================================================================
# Configuration Utilities
# =============================================================================

def yaml_config_hook(config_file):
    """Load a YAML configuration file and return a dictionary.

    Args:
        config_file: Path to the YAML file.

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

    The following keys are moved: 'feature', 'OS', 'DFS', 'OSState', 'DFSState'.
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
# Validation
# =============================================================================

def Model_Val(net, dataset_, args):
    """Run validation on a given dataset and collect per-patient predictions.

    The model is set to eval mode with both 'loss' and 'inference' outputs
    enabled. For every sample the function records clinical outcomes (OS, DFS)
    and the predicted protein expression from WSI.

    Args:
        net:      The MultiModal network instance.
        dataset_: PyTorch Dataset to evaluate.
        args:     Namespace with training hyperparameters (e.g., workers).

    Returns:
        My_Final_Result: DataFrame with patient info, survival data, and
                         predicted protein columns.
        My_Final_Loss:   Scalar average loss over all batches.
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

    # -- Switch model to evaluation mode --
    net.eval()
    net.output_type = ['loss', 'inference']

    results = []
    List_Patient_Protein = []
    Loss = 0

    # -- Iterate over all validation samples --
    for t, batch in enumerate(loader):
        batch = initialize(batch)

        with torch.no_grad():
            output = net(batch)

            # Collect clinical and survival information for this patient
            result = {
                'Patient':  batch['patient_name'],                           # Patient identifier
                'OS':       batch['OS'].detach().cpu().numpy().squeeze(),     # Overall survival time
                'OSState':  batch['OSState'].detach().cpu().numpy().squeeze(),# OS event indicator
                'DFS':      batch['DFS'].detach().cpu().numpy().squeeze(),    # Disease-free survival
                'DFSState': batch['DFSState'].detach().cpu().numpy().squeeze(),# DFS event indicator
            }

            # Accumulate loss across batches
            Loss += output['All_Loss']

            results.append(pd.DataFrame(result))
            List_Patient_Protein.append(
                output['WSI_Protein'].detach().cpu().numpy().squeeze()
            )

    # -- Aggregate results --
    My_Final_Loss   = Loss.detach().cpu().numpy() / len(loader)
    MyResult        = pd.concat(results, ignore_index=True)
    MyProtein       = pd.DataFrame(List_Patient_Protein)
    My_Final_Result = pd.concat([MyResult, MyProtein], axis=1)

    return My_Final_Result, My_Final_Loss


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

    # Extract key hyperparameters for readability
    out_dir             = args.Model_Out
    initial_checkpoint  = args.initial_checkpoint
    start_lr            = float(args.start_lr)
    batch_size          = int(args.batch_size)
    Epoch               = int(args.Epoch)

    # -----------------------------------------------------------------
    # 2. Prepare datasets
    # -----------------------------------------------------------------
    train_loader, Cohorts, dataset_val = Train_Data_WSI_TO_Protein.Discovery_Cohort(args)
    args.dim = Cohorts['dim']

    # -----------------------------------------------------------------
    # 3. Build model and move to GPU
    # -----------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler()
    net    = Model_WSI_TO_Protein.MultiModal(arg=args).to(device)

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

    # Early stopping trackers
    best_loss = 100
    delta     = 0.001

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

        # ---- Periodic validation (every 2 epochs, starting after epoch 2) ----
        if i > 2 and i % 2 == 0 and i != 0:
            Train_Protein, Train_Loss = Model_Val(net, dataset_val['train_dataset'], args)
            Test_Protein,  Test_Loss  = Model_Val(net, dataset_val['test_dataset'],  args)

            print(
                'Epoch:[{}/{}]'
                '\tTrain_Loss:{}'
                '\tTest_Loss:{}'
                .format(
                    i, Epoch,
                    str(round(Train_Loss, 6)),
                    str(round(Test_Loss, 6))
                )
            )

            # Save checkpoint if training loss did NOT improve by at least delta
            current_loss = Train_Loss
            if current_loss < best_loss - delta:
                best_loss = current_loss
            else:
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        'epoch': i,
                    },
                    out_dir + '/10X_UNI_Model/WSI_Pre_Protein_{}.pth'.format(str(i))
                )

        # ---- Step the learning rate scheduler and clean up GPU memory ----
        lr_scheduler.step()
        start_epoch = i
        torch.cuda.empty_cache()

