"""
Shared loss functions for training CNN_LSTM and MobileNetTransformer.

FocalLoss here replaces the two DIFFERENT (and each individually incomplete)
FocalLoss implementations that used to live duplicated inside
train/train_cnn.ipynb and train/train_mobile_net.ipynb:

  - train_cnn.ipynb's version wrapped `nn.CrossEntropyLoss(weight=alpha)`
    with the DEFAULT reduction='mean', so `pt = exp(-ce_loss)` was computed
    from a single batch-mean scalar instead of per-sample. The
    `(1 - pt) ** gamma` modulation then collapsed to one number for the
    whole batch, defeating the purpose of focal loss (which needs a
    per-sample confidence to down-weight easy examples individually).
  - train_mobile_net.ipynb's version used reduction='none' correctly (so pt
    is per-sample) but did not support class weights (alpha) at all, even
    though the 32-class BISINDO dataset is imbalanced (12-35 samples/class,
    see utils/EDA.ipynb) and CNN_LSTM's training relied on class weights to
    handle that.

This class merges both fixes: per-sample reduction='none' AND optional
class weights (alpha), so both models can be trained with a single,
correct, well-tested implementation.
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        alpha : Tensor[num_classes] or None
            Per-class weight (e.g. from utils/dataloader.get_class_weights()).
            None disables class weighting.
        gamma : float
            Focusing parameter. gamma=0 reduces this to plain (optionally
            class-weighted) cross-entropy.
        """
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # (B,) per-sample
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
