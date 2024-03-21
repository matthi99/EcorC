# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:05:37 2022

@author: A0067501
"""

import torch
from torch import nn


class DiceLoss(torch.nn.Module):
    """
    Dice loss function for training a multi class segmentation network.
    The prediction of the network should be of type softmax(...) and the target should be one-hot encoded.
    """

    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.depth = kwargs.get("depth", 1)
        self.weights = kwargs.get("weights", self.depth * [1])
        self.smooth = kwargs.get("smooth", 1.)
        self.p = kwargs.get("p", 2)

    def _single_class(self, prediction, target):
        prediction=prediction[:,1:,...]
        target=target[:,1:,...]
        bs = prediction.size(0)
        cl = prediction.size(1)
        p = prediction.reshape(bs,cl, -1)
        t = target.reshape(bs,cl, -1)

        intersection = (p * t).sum(2)
        total = (p.pow(self.p) + t.pow(self.p)).sum(2)

        loss = - (2 * intersection + self.smooth) / (total + self.smooth)
        loss = torch.mean(loss,1)
        return loss.mean()

    def forward(self, prediction, target, deep_supervision=False):
        if deep_supervision:
            loss=0
            L=self.depth
            for c in range(L):
                loss += self._single_class(prediction[c], target) * self.weights[c]
            return loss / sum(self.weights)
        else:
            return self._single_class(prediction, target)

    
class CrossEntropyLoss(torch.nn.Module):
    """
    Dice loss function for training a multi class segmentation network.
    The prediction of the network should be of type softmax(...) and the target should be one-hot encoded.
    """

    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.depth = kwargs.get("depth", 1)
        self.weights = kwargs.get("weights", self.depth * [1])
        self._single_class = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, prediction, target, deep_supervision=False):
        target=torch.argmax(target,1).long()
        if deep_supervision:
            loss=0
            L=self.depth
            for c in range(L):
                loss += self._single_class(prediction[c], target) * self.weights[c]
            return loss / sum(self.weights)
        else:
            return self._single_class(prediction, target)


def get_loss(crit="error", **kwargs):
    if crit == "dice":
        return DiceLoss(**kwargs)
    elif crit =="ce":
        return CrossEntropyLoss()
    else:
        return print("wrong crit!")


