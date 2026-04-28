# =============================================================================
# Model_MultiModal_Survival.py
#
# Multi-modal survival prediction model that fuses Whole Slide Image (WSI)
# features with predicted protein expression to estimate patient hazard.
#
# Architecture overview:
#   1. DimReduction          – Project high-dimensional patch features into a
#                              compact latent space.
#   2. Cross-attention       – Protein-guided attention over WSI patches
#                              (nn.MultiheadAttention: Q=protein, K/V=patches).
#   3. Gated MIL attention   – Standard gated attention pooling over patches.
#   4. Late fusion           – Concatenation of the three feature streams:
#                              [MIL-pooled WSI, cross-attended WSI, protein].
#   5. Classifier            – Single linear layer mapping the fused vector
#                              to a scalar hazard score.
#   6. Loss                  – Cox partial-likelihood loss for survival.
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torchvision import models
import matplotlib.pyplot as plt

# Project-specific modules
from dataset import *
import Attention
import DimReduction
import Survival

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
# Multi-Modal Survival Network
# -----------------------------------------------------------------------------

class MultiModal(nn.Module):
    """Multi-modal model that fuses WSI patch features with predicted protein
    expression for Cox-regression-based survival prediction.

    Three feature streams are concatenated before the final classifier:
        (1) MIL gated-attention-pooled WSI embedding,
        (2) Protein-guided cross-attention-pooled WSI embedding,
        (3) Predicted protein expression vector.
    """

    def __init__(self, arg):
        """Initialize all sub-modules.

        Args:
            arg: Namespace / config object with at least:
                - dim          : Input feature dimension of WSI patches.
                - Linears_list : List of hidden dimensions [L0, ...].
                - Laten_Dim    : Dimension of the shared latent space.
                - Patch_number : Number of patches per slide (used when B > 1).
        """
        super(MultiModal, self).__init__()
        self.arg = arg

        # -- Genomic / protein embedding (not used in forward but kept for
        #    checkpoint compatibility) --
        self.Genomic_Embedding = nn.Sequential(
            nn.Linear(self.arg.Linears_list[0], self.arg.Linears_list[0] // 64),
            nn.ReLU(),
            nn.Linear(self.arg.Linears_list[0] // 64, self.arg.Laten_Dim),
        )

        # -- Gated attention pooling (MIL slide-level aggregation) --
        self.Attention = Attention.Attention_Gated(L=512, D=128, K=1)

        # -- Patch-level dimension reduction --
        self.DimReduction = DimReduction.DimReduction(self.arg.dim)

        # -- Cross-attention: protein queries attend to WSI patch keys/values --
        self.coattn = nn.MultiheadAttention(
            embed_dim=self.arg.Laten_Dim,
            num_heads=1
        )

        # -- Survival classifier: maps fused features to a scalar hazard --
        self.classifier = nn.Sequential(
            nn.Linear(3 * self.arg.Laten_Dim, 1),
        )

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
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, batch):
        """Run the forward pass for training or inference.

        Args:
            batch: Dictionary containing:
                - 'index'      : Sample indices (used to infer batch size).
                - 'feature'    : WSI patch features, shape (B, N_patches, dim).
                - 'PreProtein' : Predicted protein expression, shape (B, Laten_Dim).
                - 'OS'         : Overall survival time in months.
                - 'OSState'    : Event indicator (1 = event, 0 = censored).

        Returns:
            output: Dictionary that may contain:
                - 'All_Loss'     : Cox partial-likelihood loss (if 'loss' mode).
                - 'hazards'      : Predicted hazard scores    (if 'inference' mode).
                - 'patch_weight' : MIL attention weights      (if 'inference' mode).
        """
        batch_size = len(batch['index'])

        # Determine the number of patches per slide:
        #   - When batch_size == 1, use the actual number of patches in this slide.
        #   - When batch_size > 1, all slides are padded/truncated to Patch_number.
        if batch_size == 1:
            Image_Feature = batch['feature']
            patch_numbers = Image_Feature.shape[1]
        else:
            patch_numbers = self.arg.Patch_number

        # ---- Step 1: Patch-level dimension reduction ----
        Dim_Feature = self.DimReduction(batch['feature'])

        # ---- Step 2: Predicted protein expression ----
        Protein_Pre = batch['PreProtein'].view(batch_size, -1)

        # ---- Step 3: Protein-guided cross-attention over WSI patches ----
        # Reshape for nn.MultiheadAttention: (sequence_len, batch, embed_dim)
        V_WSI     = Dim_Feature.view(patch_numbers, batch_size, -1)
        K_WSI     = Dim_Feature.view(patch_numbers, batch_size, -1)
        Q_Protein = Protein_Pre.view(1, batch_size, -1)

        # Cross-attention: protein queries attend to WSI patch keys/values
        _, Co_Attention_WSI = self.coattn(query=Q_Protein, key=K_WSI, value=V_WSI)

        # Weighted aggregation: attention weights x patch features
        # V_permuted: (B, embed_dim, N_patches)
        V_permuted        = V_WSI.permute(1, 2, 0)
        # Co_Attention_WSIs: (B, N_patches, 1) -> column vector for matmul
        Co_Attention_WSIs = Co_Attention_WSI.permute(0, 2, 1)
        # Result: (B, embed_dim) — protein-guided WSI embedding
        WSI_HAVE_AttentionWeight = torch.matmul(V_permuted, Co_Attention_WSIs).view(batch_size, -1)

        # ---- Step 4: Gated MIL attention pooling over WSI patches ----
        # Attention weights: (B, 1, N_patches)
        WSI_Atten_Feature = self.Attention(Dim_Feature)
        # Transpose to (B, 1, N_patches) for matrix multiplication
        WSI_Feature  = WSI_Atten_Feature.permute(0, 2, 1)
        # Weighted sum of patch features -> MIL slide-level embedding (B, Laten_Dim)
        WSI_Features = torch.matmul(WSI_Feature, Dim_Feature).view(batch_size, -1)

        # ---- Step 5: Late fusion — concatenate three feature streams ----
        # [MIL-pooled WSI | Cross-attended WSI | Predicted Protein]
        fused_tensor = torch.cat(
            (WSI_Features, WSI_HAVE_AttentionWeight, Protein_Pre), dim=1
        )

        # ---- Step 6: Hazard prediction ----
        hazards = self.classifier(fused_tensor)

        # ---- Build output dictionary ----
        output = {}

        # -- Loss computation (training mode) --
        if 'loss' in self.output_type:
            loss_cox = Survival.CoxLoss(
                survtime=batch['OS'] / 12.0,      # Convert months to years
                censor=batch['OSState'],
                hazard_pred=hazards
            )
            output['All_Loss'] = loss_cox

        # -- Inference mode: return hazards and attention weights --
        if 'inference' in self.output_type:
            output['hazards']      = hazards
            output['patch_weight'] = WSI_Atten_Feature.detach().cpu().numpy().squeeze()

        return output

