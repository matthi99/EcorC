# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:05:37 2022

@author: A0067501
"""

import torch
#import kornia
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

class NetReclassificationImprovement(torch.nn.Module):
    """
    reclassification
    """

    def __init__(self, **kwargs):
        super(NetReclassificationImprovement, self).__init__()
        self.smooth = kwargs.get("smooth", 1)


    def forward(self, prediction2d, prediction3d, target):
        bs = prediction2d.size(0)
        cl = prediction2d.size(1)
        p2 = prediction2d.reshape(bs,cl, -1)
        p3 = prediction3d.reshape(bs,cl, -1)
        t = target.reshape(bs,cl, -1)
        corr_rec_as_1 = (p3 * t * (1-p2)).sum(2)
        wro_rec_as_0 = ((1-p3) *t * p2).sum(2)
        corr_rec_as_0 = ((1-p3) * (1-t) * p2).sum(2)
        wro_rec_as_1 = (p3 * (1-t) * (1-p2)).sum(2)
        
        total_1 = t.sum(2)
        total_0 = (1-t).sum(2)
        
        class_1 = ((corr_rec_as_1-wro_rec_as_0)/(total_1+self.smooth))
        class_0 =((corr_rec_as_0-wro_rec_as_1)/(total_0))
        loss= - class_0- class_1
        loss = torch.mean(loss,1)
        return loss.mean()
    
    
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
        target=target.long()
        if deep_supervision:
            loss=0
            L=self.depth
            for c in range(L):
                loss += self._single_class(prediction[c], target) * self.weights[c]
            return loss / sum(self.weights)
        else:
            return self._single_class(prediction, target)




class BinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.eps = 1e-7

        #self.single_loss = torch.nn.BCELoss()
    def forward(self, prediction, target, alpha=0.5):
        
        loss= -alpha * target * prediction.clamp(min=self.eps).log() - (
                1 - alpha) * (1 - target) * (1 - prediction).clamp(min=self.eps).log()
        return torch.mean(loss)






def get_loss(crit="error", **kwargs):
    if crit == "bce":
        return BinaryCrossEntropyLoss(**kwargs)
    elif crit == "dice":
        return DiceLoss(**kwargs)
    elif crit =="ce":
        return CrossEntropyLoss()
    elif crit =="nri":
        return NetReclassificationImprovement()
    else:
        return print("wrong crit!")


