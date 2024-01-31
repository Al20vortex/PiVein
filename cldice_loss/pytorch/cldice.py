# TAKEN FROM 

import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel
from train_utils import DiceLoss

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth=1e-7):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        smooth = self.smooth
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + smooth) / (torch.sum(skel_pred[:, 1:, ...]) + smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + smooth) / (torch.sum(skel_true[:, 1:, ...]) + smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    smooth = 1e-7
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2. * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1e-7):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        # dice = soft_dice(y_true, y_pred)
        dice = DiceLoss()(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        smooth = self.smooth
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + smooth) / (torch.sum(skel_pred[:, 1:, ...]) + smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + smooth) / (torch.sum(skel_true[:, 1:, ...]) + smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice
