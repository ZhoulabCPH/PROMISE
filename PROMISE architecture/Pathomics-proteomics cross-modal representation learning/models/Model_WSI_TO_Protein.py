# =============================================================================
# Model_WSI_TO_Protein.py
#
# Multi-modal model for predicting protein expression from Whole Slide Images
# (WSI). The architecture uses gated attention over patch-level features,
# a shared encoder, a denoising autoencoder, and a composite loss combining
# L1 reconstruction losses with a contrastive instance loss.
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torchvision import models
import matplotlib.pyplot as plt

# Project-specific modules
from dataset_WSI_TO_Protein import *
import contrastive_loss
import DimReduction
import Attention

# Global flag for automatic mixed precision training
is_amp = True


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def dice_coeff(pred, target):
    """Compute the Dice coefficient between prediction and target tensors.

    Args:
        pred:   Predicted tensor of shape (N, ...).
        target: Ground-truth tensor of shape (N, ...).

    Returns:
        Scalar Dice coefficient with Laplace smoothing.
    """
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)          # Flatten prediction
    m2 = target.view(num, -1)        # Flatten target
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# -----------------------------------------------------------------------------
# Multi-Modal Network
# -----------------------------------------------------------------------------

class MultiModal(nn.Module):
    """Multi-modal model that maps WSI patch features to protein expression.

    Architecture overview:
        1. DimReduction   – Project high-dimensional patch features to a
                            compact representation.
        2. Attention      – Gated attention pooling over all patches to obtain
                            a single slide-level embedding.
        3. Encoder        – Map the slide-level embedding (or raw protein
                            vector) into a shared latent space.
        4. AutoEncoder    – Bottleneck denoising autoencoder applied to the
                            masked latent representation.
        5. Loss           – Weighted combination of two L1 reconstruction
                            losses and one contrastive instance loss.
    """

    def __init__(self, arg):
        """Initialize all sub-modules.

        Args:
            arg: Namespace / config object with at least the following fields:
                - dim             : Input feature dimension of WSI patches.
                - Linears_list    : List of three hidden dimensions [L0, L1, L2].
                - Result_Dim      : Dimension of the final protein output space.
                - Model_Pretrained_Res : (Optional) Path to pretrained weights.
        """
        super(MultiModal, self).__init__()
        self.arg = arg

        # -- Dimension reduction for raw patch features --
        self.DimReduction = DimReduction.DimReduction(
            self.arg.dim,
            m_dim=self.arg.Linears_list[0]
        ).cuda()

        # -- Gated attention pooling (slide-level aggregation) --
        self.Attention = Attention.Attention_Gated(
            L=self.arg.Linears_list[0],
            D=128,
            K=1
        ).cuda()

        # -- Protein dimensionality reduction (single linear layer) --
        self.Protein_DimR = nn.Sequential(
            nn.Linear(self.arg.Linears_list[0], self.arg.Linears_list[1]),
        )

        # -- Shared encoder: maps slide embedding / protein vector to latent --
        self.Encoder = nn.Sequential(
            nn.Linear(self.arg.Linears_list[0], self.arg.Linears_list[1]),
            nn.ReLU(),
            nn.Linear(self.arg.Linears_list[1], self.arg.Linears_list[2]),
            nn.ReLU(),
            nn.Linear(self.arg.Linears_list[2], self.arg.Result_Dim),
        )

        # -- Denoising autoencoder: compress then reconstruct the latent --
        self.AutoEncoder = nn.Sequential(
            # Compress: Result_Dim -> Result_Dim // 4
            nn.Linear(self.arg.Result_Dim, self.arg.Result_Dim // 4),
            nn.BatchNorm1d(self.arg.Result_Dim // 4),
            nn.ReLU(),
            # Compress further: Result_Dim // 4 -> Result_Dim // 16
            nn.Linear(self.arg.Result_Dim // 4, self.arg.Result_Dim // 16),
            nn.BatchNorm1d(self.arg.Result_Dim // 16),
            nn.ReLU(),
            # Expand: Result_Dim // 16 -> Result_Dim // 4
            nn.Linear(self.arg.Result_Dim // 16, self.arg.Result_Dim // 4),
            nn.BatchNorm1d(self.arg.Result_Dim // 4),
            nn.ReLU(),
            # Expand back: Result_Dim // 4 -> Result_Dim
            nn.Linear(self.arg.Result_Dim // 4, self.arg.Result_Dim),
        )

        # -- Learnable loss weights (softmax-normalized) --
        self.weight_loss = nn.Sequential(
            nn.Linear(3, 3),
            nn.Softmax()
        ).cuda()

    # -----------------------------------------------------------------
    # Pretrained weight loading
    # -----------------------------------------------------------------

    def load_pretrain(self):
        """Load pretrained weights from disk into self.res (backbone)."""
        print('loading %s ...' % self.arg.Model_Pretrained_Res)
        checkpoint = torch.load(
            self.arg.Model_Pretrained_Res,
            map_location=lambda storage, loc: storage
        )
        print(self.res.load_state_dict(checkpoint, strict=False))

    # -----------------------------------------------------------------
    # Masking utility
    # -----------------------------------------------------------------

    def generate_mask(self, x, mask_ratio=0.5):
        """Generate a random binary mask for denoising.

        Args:
            x:          Input tensor whose shape determines the mask size.
            mask_ratio: Fraction of elements to drop (set to zero).

        Returns:
            Boolean tensor of the same shape as x (True = keep).
        """
        mask = torch.rand(x.shape) > mask_ratio
        return mask

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, batch):
        """Run the forward pass for training or inference.

        Args:
            batch: Dictionary containing:
                - 'index'   : Sample indices (used to infer batch size).
                - 'feature' : WSI patch features, shape (B, N_patches, dim).
                - 'Protein' : Ground-truth protein expression (training only).

        Returns:
            output: Dictionary that may contain:
                - 'All_Loss'        : Scalar total loss          (if 'loss' mode).
                - 'WSI_Protein'     : Predicted protein from WSI (if 'inference' mode).
                - 'Protein_Protein' : Reconstructed protein      (if 'inference' mode).
        """
        batch_size = len(batch['index'])

        # ---- Step 1: Patch-level dimension reduction ----
        Dim_Feature = self.DimReduction(batch['feature'])

        # ---- Step 2: Gated attention aggregation over patches ----
        # Attention weights: (B, 1, N_patches)
        WSI_Atten_Feature = self.Attention(Dim_Feature)
        # Transpose to (B, 1, N_patches) for matrix multiplication
        WSI_Feature = WSI_Atten_Feature.permute(0, 2, 1)
        # Weighted sum of patch features -> slide-level embedding (B, L0)
        WSI_Features = torch.matmul(WSI_Feature, Dim_Feature).view(batch_size, -1)

        # ---- Step 3: Encode WSI embedding into shared latent space ----
        WSI_Latent = self.Encoder(WSI_Features)

        # ---- Step 4: Apply random mask and denoise via AutoEncoder ----
        WSI_Nose_Index = self.generate_mask(WSI_Latent).cuda()
        WSI_Nose = WSI_Latent * WSI_Nose_Index
        WSI_DlNose = self.AutoEncoder(WSI_Nose)

        # ---- Build output dictionary ----
        output = {}

        # ---- Loss computation (training mode) ----
        if 'loss' in self.output_type:
            # Encode ground-truth protein expression into the same latent space
            Protein_real = batch['Protein'].view(batch_size, -1)
            Protein_Latent = self.Encoder(Protein_real)

            # Apply random mask and denoise the protein latent
            Protein_Nose_Index = self.generate_mask(Protein_Latent).cuda()
            Protein_Nose = Protein_Latent * Protein_Nose_Index
            Protein_DlNose = self.AutoEncoder(Protein_Nose)

            # L1 reconstruction loss between protein latent and denoised WSI
            loss_fn = nn.L1Loss(reduction='mean')
            LrecProteinDLWSI = loss_fn(Protein_Latent, WSI_DlNose)

            # L1 reconstruction loss between protein latent and denoised protein
            LrecDlProteinProtein = loss_fn(Protein_Latent, Protein_DlNose)

            # Contrastive instance loss between denoised WSI and denoised protein
            conloss = contrastive_loss.InstanceLoss(
                batch_size=batch_size,
                temperature=1.0,
                device='cuda'
            )
            Lcon = conloss(WSI_DlNose, Protein_DlNose)

            # Learnable softmax-normalized weights for the three loss terms
            Ori_weight = torch.tensor([1, 1, 1], dtype=torch.float32).cuda()
            weight = self.weight_loss(Ori_weight)

            # Total loss: weighted sum of reconstruction + contrastive losses
            Loss = (weight[0] * LrecProteinDLWSI
                    + weight[1] * LrecDlProteinProtein
                    + weight[2] * Lcon)

            output['All_Loss'] = Loss

        # ---- Inference mode: return predicted protein expressions ----
        if 'inference' in self.output_type:
            output['WSI_Protein'] = WSI_DlNose
            output['Protein_Protein'] = Protein_DlNose

        return output

