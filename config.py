# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:43:42 2022

@author: A0067501
"""


class Config_EMIDEC:
    
    all_patients=[i for i in range(1,101)]
    
    cross_validation={} #same cross validationa as nnunet
    for i in range(5):
        cross_validation['fold_'+str(i)]={}
    cross_validation['fold_0']['val']=[5,10,14,21,26,29,34,36,41,43,50,52,61,64,66,70,76,79,87,89]
    cross_validation['fold_1']['val']=[1,3,7,9,18,27,31,38,40,46,47,48,51,53,55,58,72,77,88,93]
    cross_validation['fold_2']['val']=[13,17,19,20,22,23,25,28,33,49,59,63,69,75,80,84,85,86,90,96]
    cross_validation['fold_3']['val']=[4,6,16,24,32,45,54,56,57,62,65,67,68,71,73,91,92,94,95,97]
    cross_validation['fold_4']['val']=[2,8,11,12,15,30,35,37,39,42,44,60,74,78,81,82,83,98,99,100]
    
    for i in range(5):
        cross_validation['fold_'+str(i)]['train']=[]
        for p in all_patients:
            if p not in cross_validation['fold_'+str(i)]['val']:
                cross_validation['fold_'+str(i)]['train'].append(p)
    
    train_data_setup_2d = {
        "spatialtransform":{
            "do_elastic_deform":True, 
            "alpha":(0.,175.),
            },
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 48,
            "height": 48,
            "translation":5
            },
        "gamma": {
            "retain_stats":True,
            "gamma_range": (0.7, 1.5),
            "p_per_sample": 0.15
            },
        "lowres": {
            "scale_factor": (0.5, 1),
            "p_per_sample": 0.15,
            },
        "contrast": {
            "contrast_range": (0.65, 1.5),
            "p_per_sample": 0.15
            },
        "gaussianblur":{
            "blur_sigma": (0.5, 1.5),
            "p_per_sample": 0.1
            },
        "gaussian":{
            "sigma":0.1,
            "p_gaussian":0.15},
        "flip":True,
        }
    
    
    train_data_setup_3d = {
        "spatialtransform":{
            "do_elastic_deform":True, 
            "alpha":(0.,900.),
            },
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 48,
            "height": 48,
            "translation": 5
            },
        "gamma": {
            "retain_stats":True, 
            "gamma_range": (0.65, 1.55), 
            "p_per_sample": 0.15
            },
        "lowres": {
            "scale_factor": (0.45, 1),
            "p_per_sample": 0.15,
            },
        "contrast": {
            "contrast_range": (0.6, 1.55), 
            "p_per_sample": 0.15
            }, 
        "gaussianblur":{
            "blur_sigma": (0.5, 1.5), 
            "p_per_sample": 0.1
            }, 
        "gaussian":{
            "sigma":0.1, 
            "p_gaussian":0.15},
        "flip":False, 
        }
    
    val_data_setup = {
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "translation": 0
            },
    }  
    
     



class Config_MyoPS:
    all_patients = [i for i in range(101,126)]
    
    cross_validation={} #same cross validationa as nnunet
    for i in range(5):
        cross_validation['fold_'+str(i)]={}
    cross_validation['fold_0']['val']=[109,114,117,120,122]
    cross_validation['fold_1']['val']=[104,111,113,116,118]
    cross_validation['fold_2']['val']=[101,107,108,112,119]
    cross_validation['fold_3']['val']=[115,121,123,124,125]
    cross_validation['fold_4']['val']=[102,103,105,106,110]
    
    for i in range(5):
        cross_validation['fold_'+str(i)]['train']=[]
        for p in all_patients:
            if p not in cross_validation['fold_'+str(i)]['val']:
                cross_validation['fold_'+str(i)]['train'].append(p)
    
    
    train_data_setup_2d = {
        "spatialtransform":{
            "do_elastic_deform":True, 
            "alpha":(0.,175.),
            },
        
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 192,
            "height": 192,
            "translation": 4
            },
        "gamma": {
            "retain_stats":True,
            "gamma_range": (0.7, 1.5),
            "p_per_sample": 0.15
            },
        "lowres": {
            "scale_factor": (0.5, 1),
            "p_per_sample": 0.15,
            },
        "contrast": {
            "contrast_range": (0.65, 1.5),
            "p_per_sample": 0.15
            },
        "gaussianblur":{
            "blur_sigma": (0.5, 1.5),
            "p_per_sample": 0.1
            },
        "gaussian":{
            "sigma":0.1,
            "p_gaussian":0.15},
        "flip":True,
        }
    
    
    
    train_data_setup_3d = {
        "spatialtransform":{
            "do_elastic_deform":True, 
            "alpha":(0.,900.),
            },
       "normalize": {
           "mode": "mean"
           },
       "ROI": {
           "width": 192,
           "height": 192,
           "translation": 4
           },
       "gamma": {
           "retain_stats":True, 
           "gamma_range": (0.65, 1.55), 
           "p_per_sample": 0.15
           },
       "lowres": {
           "scale_factor": (0.45, 1),
           "p_per_sample": 0.15,
           },
       "contrast": {
           "contrast_range": (0.6, 1.55), 
           "p_per_sample": 0.15
           }, 
       "gaussianblur":{
           "blur_sigma": (0.5, 1.5), 
           "p_per_sample": 0.1
           }, 
       "gaussian":{
           "sigma":0.1, 
           "p_gaussian":0.15},
       "flip":False, 
       }

    
    
    val_data_setup = {
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 192,
            "height": 192,
            "translation": 0
            },
    }  





