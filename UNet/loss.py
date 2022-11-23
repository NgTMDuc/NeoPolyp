import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        from torchgeometry.losses import DiceLoss as DL
        return (DL(inputs, targets))
