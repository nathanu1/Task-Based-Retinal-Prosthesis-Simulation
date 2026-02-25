"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """Hybrid loss: BCE + MSE + Dice + safety penalty."""

    def __init__(self, bce_w=0.4, mse_w=0.3, dice_w=0.2, safety_w=0.1):
        super().__init__()
        self.bce_w = bce_w
        self.mse_w = mse_w
        self.dice_w = dice_w
        self.safety_w = safety_w

    def forward(self, pred, target, safety_logits=None):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        mse = F.mse_loss(torch.sigmoid(pred), target)
        pred_sig = torch.sigmoid(pred)
        dice = 1 - (2 * (pred_sig * target).sum() + 1) / (pred_sig.sum() + target.sum() + 1)
        loss = self.bce_w * bce + self.mse_w * mse + self.dice_w * dice
        if safety_logits is not None and safety_logits.numel() > 0:
            safety_mse = F.mse_loss(torch.sigmoid(safety_logits), target)
            loss = loss + self.safety_w * safety_mse
        return loss
