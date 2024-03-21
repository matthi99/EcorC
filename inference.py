# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:03:27 2023

@author: matthias
"""

import torch
import os
from utils.architectures import get_network
import json
import numpy as np
import yaml
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import argparse
from utils.utils_training import get_logger
from utils.utils_inference import load_2dnet, load_3dnet, normalize, predict2d, predict3d, plot_prediction


parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('dataset_name', type=str, help="Dataset name. Possible choices are EMIDEC or MyoPS")
#parser.add_argument('--input_folder', type= str, default="")
#parser.add_argument('--output_folder', type= str, default="RESULTS_FOLDER")
parser.add_argument('--plots', type=bool, default=True)
args = parser.parse_args()

#define logger and device
logger= get_logger(args.dataset_name+"_inference")
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#load plans
if args.dataset_name =="EMIDEC":
    f=open("plans/plans_EMIDEC.json")
    plans= json.load(f)
    path_to_testdata= os.path.join(plans["data_folder"], "emidec-segmentation-testset-1.0.0")
    testfiles=[os.path.join(path_to_testdata,f, "images", f+".nii.gz") for f in os.listdir(path_to_testdata) if os.path.isdir(os.path.join(path_to_testdata,f))]
    WIDTH=48
elif args.dataset_name =="MyoPS":
    f=open("plans/plans_MyoPS.json")
    plans= json.load(f)
    path_to_testdata= os.path.join(plans["data_folder"], "MyoPS 2020 Dataset", "test20")
    testfiles=[os.path.join(path_to_testdata,f) for f in os.listdir(path_to_testdata) if f.endswith("DE.nii.gz")]
    WIDTH=192
else:
    logger.info("Wrong dataset_name! Possible names are EMIDEC or MyoPS.")
    

path_to_nets= os.path.join(plans["data_folder"], "RESULTS_FOLDER", args.dataset_name)
savefolder= os.path.join(plans["data_folder"], "RESULTS_FOLDER", args.dataset_name, "inference")


for file in testfiles:
    example = nib.load(file)
    img= example.get_fdata()
    shape= img.shape
    if args.dataset_name =="EMIDEC":
        patientnr= file.split("_")[-1][0:3]
    elif args.dataset_name =="MyoPS": 
        patientnr= file.split("_")[-2]
    center=(img.shape[0]//2, img.shape[1]//2)
    img=np.transpose(img,(2,0,1))
    im=img.copy()
    im=normalize(im[:,center[0]-WIDTH : center[0]+WIDTH, center[1]-WIDTH: center[1]+WIDTH])
    im = torch.from_numpy(im[None,None, ...].astype("float32")).to(device)
    result = torch.zeros(( len(plans["classes"]), im.shape[2], im.shape[3], im.shape[4])).to(device)
    for i in range(5):
        net2d=load_2dnet(os.path.join(path_to_nets, f"fold_{i}",  "2d-net"), device)
        net3d=load_3dnet(os.path.join(path_to_nets, f"fold_{i}",  "3d-net"), device)
    
        out2d = predict2d(im, net2d, len(plans["classes"]))
        result += predict3d(im, out2d, net3d, len(plans["classes"]))
        
        out2d = torch.flip(predict2d(torch.flip(im, dims=[3]), net2d, len(plans["classes"])), dims=[2])
        result += predict3d(im, out2d, net3d, len(plans["classes"]))
        
        out2d = torch.flip(predict2d(torch.flip(im, dims=[4]), net2d, len(plans["classes"])), dims=[3])
        result += predict3d(im, out2d, net3d, len(plans["classes"]))
        
    result= torch.argmax(result,0).cpu().detach().numpy()
    result=np.pad(result, ((0,0),(center[0]-WIDTH, shape[0]-(center[0]+WIDTH)), (center[1]-WIDTH, shape[1]-(center[1]+WIDTH))), 
                            constant_values=0)
    
    
    prediction = nib.Nifti1Image(result, example.affine, example.header)
    if not os.path.exists(os.path.join(savefolder, f"Case_{patientnr}")):
        os.makedirs(os.path.join(savefolder, f"Case_{patientnr}"))
    nib.save(prediction, os.path.join(savefolder,f"Case_{patientnr}", f"Case_{patientnr}.nii.gz"))
    if args.plots:
        plot_prediction(img, result, patientnr, plans["classes"][1:], savefolder)
    
    logger.info(f"Saved results for Case_{patientnr}")
    

logger.info(f"All predctions saved in {savefolder}")





