# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:27:15 2022

@author: Schwab Matthias
"""
import nibabel as nib
import numpy as np
import os
import argparse
import json
from config import Config_EMIDEC, Config_MyoPS

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help="Dataset name. Possible choices are EMIDEC or MyoPS")
parser.add_argument('data_folder', type=str, help="Path to folder where dataset is stored. ")
args = parser.parse_args()

if args.dataset_name == "EMIDEC":
    #3d data
    train_folder= os.path.join(args.data_folder, 'emidec-dataset-1.0.1/')
    preprocessed_folder= os.path.join(args.data_folder, 'EMIDEC_preprocessed')
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    
    save_folder= os.path.join(preprocessed_folder, 'traindata3d/')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    train_patients=[f for f in os.listdir(train_folder) if os.path.isdir(train_folder+f)] 
    
    for patient in train_patients:
        data={}
        #load images
        img  = nib.load(os.path.join(train_folder,patient,'Images',patient+'.nii.gz')).get_fdata()
        data['center']=(img.shape[0]//2, img.shape[1]//2)
        img=np.transpose(img,(2,0,1))
        data['img']=img
        
        #create masks
        cont  = nib.load(train_folder+patient+'/Contours/'+patient+'.nii.gz').get_fdata()
        cont = np.transpose(cont,(2,0,1))
        
        temp=np.copy(cont)
        heart=np.zeros(temp.shape)
        heart[temp==0]=1
        blood=np.zeros(temp.shape)
        blood[temp==1]=1
        muscle=np.zeros(temp.shape)
        muscle[temp==2]=1
        scar=np.zeros(temp.shape)
        scar[temp==3]=1
        mvo=np.zeros(temp.shape)
        mvo[temp==4]=1
    
        masks=np.concatenate((heart[...,None],blood[...,None], muscle[...,None], scar[...,None], mvo[...,None]), axis=3)
        data['masks']=masks
        
        #save
        np.save(os.path.join(save_folder,patient[0:5]+patient[-3:]+'.npy'), data)
        
    
    #2D data
    save_folder2d= os.path.join(preprocessed_folder, 'traindata2d/')
    if not os.path.exists(save_folder2d):
        os.makedirs(save_folder2d)
        
    for patient in train_patients:
        data=np.load(os.path.join(save_folder, patient[0:5]+patient[-3:]+'.npy'), allow_pickle=True).item()
        L=data['img'].shape[0]
        for i in range(L):
            data2d={}
            data2d['center']=data['center']
            data2d['img']=data['img'][i,...]
            data2d['masks']=data['masks'][i,...]
            np.save(os.path.join(save_folder2d,patient[0:5]+patient[-3:]+'_'+str(i)+'.npy'), data2d)
    
    #create plans for training        
    if not os.path.exists("plans/"):
        os.makedirs("plans/")

    plans={
        "data_folder": args.data_folder, 
        "batchsize2d": 32, 
        "batchsize3d":4, 
        "out_channels": 5,
        "in_channels3d": 3, 
        "lr2d": 5*1e-3, 
        "lr3d": 1e-2, 
        "dropout3d": 0.1,
        "perturbation": [0.1, 0.05, 0.1, 0.1, 0.02, 0.63], 
        "classes": ["bg", "blood", "muscle", "scar", "mvo"],
        "cross_validation": Config_EMIDEC.cross_validation,
        "train_data_setup_2d": Config_EMIDEC.train_data_setup_2d, 
        "train_data_setup_3d": Config_EMIDEC.train_data_setup_3d, 
        "val_data_setup": Config_EMIDEC.val_data_setup
        }
    json.dump(plans, open(os.path.join("plans","plans_EMIDEC.json"), "w"))
    
    print("EMIDEC data prepared!")

elif args.dataset_name == "MyoPS":
    #3d data
    train_folder= os.path.join(args.data_folder, 'MyoPS 2020 Dataset/')
    preprocessed_folder= os.path.join(args.data_folder, 'MyoPS_preprocessed')
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    
    save_folder= os.path.join(preprocessed_folder, 'traindata3d/')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    train_gt =os.path.join(train_folder, 'train25_myops_gd/')
    train_img =os.path.join(train_folder, 'train25/')

    
    files_gt=sorted([train_gt+f for f in os.listdir(train_gt)])
    files_imgs=sorted([train_img+f for f in os.listdir(train_img) if f.endswith("DE.nii.gz")])
    
    
    
    for gt,im in zip(files_gt, files_imgs):
        data={}
        #load images
        img  = nib.load(im).get_fdata()
        data['center']=(img.shape[0]//2, img.shape[1]//2)
        img=np.transpose(img,(2,0,1))
        data['img']=img
    

        #create masks
        cont  = nib.load(gt).get_fdata()
        cont = np.transpose(cont,(2,0,1))
        patient_nr=im.split("_")[-2]
        
        temp=np.copy(cont)
        bg=np.zeros(temp.shape)
        bg[temp==0]=1
        bg[temp==600]=1
        blood=np.zeros(temp.shape)
        blood[temp==500]=1
        muscle=np.zeros(temp.shape)
        muscle[temp==200]=1
        muscle[temp==1220]=1
        scar=np.zeros(temp.shape)
        scar[temp==2221]=1
        
        masks=np.concatenate((bg[...,None], blood[...,None], muscle[...,None], scar[...,None]), axis=3)
        data['masks']=masks
    
        #save
        np.save(os.path.join(save_folder, "Case_"+patient_nr+'.npy'), data)
    
    
    #2d data
    save_folder2d = os.path.join(preprocessed_folder, 'traindata2d/')
    if not os.path.exists(save_folder2d):
        os.makedirs(save_folder2d)
        
    train_patients= os.listdir(save_folder)
    slices=[]
    for patient in train_patients:
        data=np.load(save_folder+patient, allow_pickle=True).item()
        L=data['img'].shape[0]
        slices.append(L)
        for i in range(L):
            data2d={}
            data2d['center']=data['center']
            data2d['img']=data['img'][i,...]
            data2d['masks']=data['masks'][i,...]
            np.save(os.path.join(save_folder2d, patient[:-4]+'_'+str(i)+'.npy'), data2d)
            
    #create plans for training   
    plans={
        "data_folder": args.data_folder, 
        "batchsize2d": 6, 
        "batchsize3d":4, 
        "out_channels": 4,
        "in_channels3d": 2, 
        "dropout3d": 0.05,
        "perturbation": [0.1, 0.05, 0.1, 0.1, 0.1, 0.55], 
        "classes": ["bg", "blood", "muscle", "scar"],
        "cross_validation": Config_MyoPS.cross_validation,
        "train_data_setup_2d": Config_MyoPS.train_data_setup_2d, 
        "train_data_setup_3d": Config_MyoPS.train_data_setup_3d, 
        "val_data_setup": Config_MyoPS.val_data_setup
        }
    json.dump(plans, open(os.path.join("plans","plans_MyoPS.json"), "w"))
    
    print("MyoPS data prepared!")

else:
    print("Wrong dataset_name! Possible names are EMIDEC or MyoPS.")
        
    