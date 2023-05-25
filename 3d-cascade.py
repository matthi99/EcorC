# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:24:17 2023

@author: A0067501
"""

import torch

from utils.architectures import get_network
from utils.utils_training import Histogram, plot_examples, save_checkpoint, get_logger, CascadeAugmentation_all_channels
from dataloader import CardioDataset, CardioCollatorMulticlass
from utils.metrics import get_metric
from utils.losses import get_loss

import os
import argparse
import json
import numpy as np
import yaml
from config import Config


#Define parameters for training 
parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('--epochs', type=int, default=750)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--batchnorm', type=bool, default=True)
parser.add_argument('--start_filters', type=int, default=32)
parser.add_argument('--out_channels', type=int, default=5)
parser.add_argument('--activation', type=str, default="leakyrelu")
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--disruption', nargs='+', type=float, default= [0.05, 0.05, 0.05, 0.1, 0.05, 0.02, 0.68] )
parser.add_argument('--fold',  help= "On which fold should be trained (number between 0 and 4)", type=int, default=0)
parser.add_argument('--data3d', help= "Path to data folder", type=str, 
                    default="DATA/traindata/")
parser.add_argument('--savepath', help= "Path were resuts should get saved", type=str, 
                    default="RESULTS_FOLDER/")
parser.add_argument('--savefolder', type=str, default="3d-cascade_")
args = parser.parse_args()


#create save folders if they dont exist already
folder=args.savepath
savefolder=args.savefolder+str(args.fold)+"/"
if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists(os.path.join(folder,savefolder)):
    os.makedirs(os.path.join(folder,savefolder))
# if not os.path.exists(os.path.join(folder,savefolder,'plots')):
#     os.makedirs(os.path.join(folder,savefolder,'plots'))   
    

#define logger and get network configs
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger= get_logger("cascade")
logger.info(f"Training on {device}")
config = {
        "losses": ["dice"],
        "dice":{"p":2, "depth":3, "weights":[1,0.5, 0.25]},
        "ce": {"depth":3, "weights":[1, 0.5, 0.25]},
        "bce" :{},
        "metrics":["dice"],
    "network":{
        "activation": args.activation,
        "dropout": args.dropout,
        "batchnorm": args.batchnorm,
        "start_filters": args.start_filters,
        "in_channels":6,
        "out_channels": args.out_channels,
        "kernel_size": [[1, 3, 3],[1, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
        "padding": [[0, 1, 1],[0, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1]],
        "stride": [[1, 1, 1],[1, 2, 2],[1, 2, 2],[1, 2, 2],[1, 2, 2]],
        "residual": False, 
        "last_activation":"softmax"},
    
    "classes": ["blood","muscle", "scar", "mvo"],
    "best_metric":-float('inf')
}


    
setup_train=Config.train_data_setup_2d
setup_val=Config.val_data_setup
if args.fold>=0 and args.fold<5:
    logger.info(f"Training on fold {args.fold}")
    setup_train['patients']=Config.cross_validation[f"fold_{args.fold}"]['train']
    setup_val['patients']=Config.cross_validation[f"fold_{args.fold}"]['val']
else:
    logger.info("Error: Wrong number spezified in --fold!")
    

#get dataloaders         
cd_train = CardioDataset(folder=args.data3d,  **setup_train)
cd_val = CardioDataset(folder=args.data3d, validation=True, **setup_val)    
collator = CardioCollatorMulticlass(classes=("bg", "blood","muscle", "scar", "mvo"), classes_data= Config.classes)
dataloader_train = torch.utils.data.DataLoader(cd_train, batch_size=args.batchsize, collate_fn=collator,
                                                        shuffle=True)    
#for evaluation we read in the whole stack so it has to be batch_size=0 because of different z dimensions
dataloader_eval = torch.utils.data.DataLoader(cd_val, batch_size=1, collate_fn=collator,
                                                        shuffle=True) 


#get networks, optimizer, loss, metric and histogramm 
path_to_2d_net=open(f"paths/best_weights2d_{args.fold}.txt", "r").readline()
params= yaml.load(open(path_to_2d_net+"/config-best_weights.json", 'r'), Loader=yaml.SafeLoader)['network']
weights = torch.load(path_to_2d_net+"best_weights.pth",  map_location=torch.device(device))
net2d = get_network(architecture='unet2d', device=device, **params)
net2d.load_state_dict(weights)
net2d.eval()


net3d = get_network(architecture='unet', **config["network"]).to(device)
net3d.train()
    
opt = torch.optim.SGD(net3d.parameters(), 1e-2,  weight_decay=3e-5, momentum=0.99, nesterov=True)
lambda1 = lambda epoch: (1-epoch/args.epochs)**0.9
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    
losses = {loss: get_loss(crit=loss, **config[loss]) for loss in config['losses']}
metrics={metric: get_metric(metric=metric) for metric in config['metrics']} 
classes=config["classes"]
histogram=Histogram(classes, metrics)

#dont know yet what to do with disruptions!!!
aug = CascadeAugmentation_all_channels(probs= args.disruption)


#train
best_metric=-float('inf')
for epoch in  range(args.epochs):
    net3d.train()
    logger.info(f"Epoch {epoch}\{args.epochs}-------------------------------")
    steps = 0
    histogram.append_hist()
    for im,mask in dataloader_train:
        opt.zero_grad()
        loss=0
        gt= torch.cat([mask[key].float() for key in mask.keys()],1)
        if epoch > 100: 
            out2d= aug.augment(net2d, im, gt)
        else:
            out2d = aug.augment(net2d, im, gt, do_nothing=True)
        
        
        in3d=torch.cat((im,out2d),1)
        out=net3d(in3d)
        
        for l in losses:
            if l=='dice':
                loss+=losses[l](out, gt, deep_supervision=True)
            if l=='ce':
                loss+=losses[l](out, torch.argmax(gt,1), deep_supervision=True)
            if l=='bce':
                mask_mvo=mask["mvo"].float()
                out_mvo=out[0][:,4:5,...]
                loss+=losses[l](out_mvo, mask_mvo, alpha=0.999)
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net3d.parameters(), 12)
        opt.step()
        histogram.add_loss(loss)
        histogram.add_train_metrics(out[0],gt)
        steps += 1
    histogram.scale_train(steps)
    
    # if epoch%25==0:
    #     plot_examples(im,out[0],epoch,os.path.join(folder,savefolder,"plots"), train=True)
        
        
        
    #evaluate (we are evaluating on a per patient level)
    net3d.eval()
    steps = 0
    for im,mask in dataloader_eval:
        with torch.no_grad():
            in2d=torch.moveaxis(im,2,0)
            out2d=[]
            for b in range(im.shape[0]):
                with torch.no_grad():
                    temp=net2d(in2d[:,b,...])[0]
                temp=torch.moveaxis(temp,0,1)
                temp=torch.argmax(temp,0).long()
                temp=torch.nn.functional.one_hot(temp,5)
                temp=torch.moveaxis(temp,-1,0)
                out2d.append(temp)
            
            out2d=torch.stack(out2d,0)
            in3d=torch.cat((im,out2d),1)
            
            out=net3d(in3d)[0]
            gt= torch.cat([mask[key].float() for key in mask.keys()],1)
            histogram.add_val_metrics(out,gt)
            steps += 1
            
    histogram.scale_val(steps)
    histogram.plot_hist(os.path.join(args.savepath,savefolder))
    # if epoch%25==0:
    #         plot_examples(im,out,epoch,os.path.join(folder,savefolder,"plots"), train=False)
    
   
    #ceck for improvement and save best model
    val_metric=0
    for cl in ["muscle", "scar", "mvo"]:
        val_metric+=histogram.hist[config['metrics'][0]][f"val_{cl}"][-1]
    val_metric=val_metric/3
    if val_metric > best_metric:
        best_metric=val_metric
        config["best_metric"]=best_metric
        logger.info(f"New best Metric {best_metric}")
        histogram.print_hist(logger)
        save_checkpoint(net3d, os.path.join(folder, savefolder), args.fold, "best_weights", savepath=False)
        json.dump(config, open(os.path.join(folder,savefolder,"config-best_weights.json"), "w"))
        
    logger.info(scheduler.get_last_lr())
    scheduler.step()
    
        
np.save(os.path.join(folder, savefolder, "histogram.npy"),histogram.hist)
save_checkpoint(net3d, os.path.join(folder, savefolder), args.fold, "last_weights")
json.dump(config, open(os.path.join(folder, savefolder, "config-last_weights.json"), "w"))
logger.info("Training Finished!") 
    
    
    
    
