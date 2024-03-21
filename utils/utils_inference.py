# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:27:30 2024

@author: A0067501
"""


import yaml
from utils.architectures import get_network
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import os

def load_2dnet(path, device):
    params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
    net2d = get_network(architecture='unet2d', device=device, **params)
    net2d.load_state_dict(weights)
    net2d.eval()
    return net2d

def load_3dnet(path, device):
    params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
    net3d = get_network(architecture='unet', device=device, **params)
    net3d.load_state_dict(weights)
    net3d.eval()
    return net3d

def normalize(img):
    for i in range(img.shape[0]):
        img[i,...]=(img[i,...] - np.mean(img[i,...])) / np.std(img[i,...])
    return img

def predict2d(im,net2d, num_classes):
    in2d=torch.moveaxis(im,2,0)[:,0,...]
    temp=net2d(in2d)[0]
    temp=torch.moveaxis(temp,0,1)
    temp=torch.argmax(temp,0).long()
    temp=torch.nn.functional.one_hot(temp,num_classes)
    temp=torch.moveaxis(temp,-1,0)
    return temp

def predict3d(im, out2d, net3d, num_classes):
    out2d = out2d[3:,...]
    out2d = out2d[None,...]
    in3d=torch.cat((im, out2d),1)
    out3d=net3d(in3d)[0]
    out3d=torch.argmax(out3d,1).long()
    out3d=torch.nn.functional.one_hot(out3d,num_classes)
    out3d=torch.moveaxis(out3d,-1,1).float()[0,...]
    return out3d


def plot_prediction(img, result, patientnr, classes, savefolder):
    segmentation=np.ma.masked_where(result == 0, result)
    segmentation-=1
    for i in range(img.shape[0]):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img[i], cmap='gray')
        plt.gca().set_title(f"Case_{patientnr}_{i}")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img[i], cmap='gray')
        mat=plt.imshow(segmentation[i], 'jet', alpha=0.45, interpolation="none", vmin = 0, vmax = len(classes))
        plt.axis('off')
        plt.gca().set_title('Prediction')
            
        values = np.arange(len(classes))
        colors = [ mat.cmap(mat.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
        plt.savefig(os.path.join(savefolder,f"Case_{patientnr}", f"Slice_{i}.png"), bbox_inches='tight', dpi=500)
        plt.close()