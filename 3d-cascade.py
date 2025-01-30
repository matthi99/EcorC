# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:24:17 2023

@author: A0067501
"""

import torch
from utils.utils_training import get_network, Histogram, save_checkpoint, get_logger, CascadeAugmentation, predict_cascade
from dataloader import CardioDataset, CardioCollatorMulticlass
from utils.metrics import get_metric
from utils.losses import get_loss

import os
import argparse
import json
import numpy as np
import yaml



#Define parameters for training 
parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('dataset_name', type=str, help="Dataset name. Possible choices are EMIDEC or MyoPS")
parser.add_argument('fold', type=int)
parser.add_argument('--epochs', type=int, default=750)
parser.add_argument('--batchnorm', type=bool, default=True)
parser.add_argument('--start_filters', type=int, default=32)
parser.add_argument('--activation', type=str, default="leakyrelu")
args = parser.parse_args()

#define logger and device
logger= get_logger(args.dataset_name+"_cascade")
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on {device}")

#load plans
if args.dataset_name =="EMIDEC":
    f=open("plans/plans_EMIDEC.json")
    plans= json.load(f)
elif args.dataset_name =="MyoPS":
    f=open("plans/plans_MyoPS.json")
    plans= json.load(f)
else:
    logger.info("Wrong dataset_name! Possible names are EMIDEC or MyoPS.")

#create save folders if they dont exist already
path_results=os.path.join(plans["data_folder"], "RESULTS_FOLDER")
savefolder=os.path.join(path_results, args.dataset_name, f"fold_{args.fold}","3d-net")
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

#get network configs and train setups
config = {
        "losses": ["dice"],
        "dice":{"p":2, "depth":3, "weights":[1,0.5, 0.25]},
        "ce": {"depth":3, "weights":[1, 0.5, 0.25]},
        "bce" :{},
        "metrics":["dice"],
    "network":{
        "activation": args.activation,
        "dropout": plans["dropout3d"],
        "batchnorm": args.batchnorm,
        "start_filters": args.start_filters,
        "in_channels":plans["in_channels3d"],
        "out_channels": plans["out_channels"],
        "kernel_size": [[1, 3, 3],[1, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
        "padding": [[0, 1, 1],[0, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1]],
        "stride": [[1, 1, 1],[1, 2, 2],[1, 2, 2],[1, 2, 2],[1, 2, 2]],
        "residual": False, 
        "last_activation":"softmax"},
    "best_metric":-float('inf')
}

setup_train=plans["train_data_setup_3d"]
setup_train["dataset"]= args.dataset_name
setup_val=plans["val_data_setup"]
if args.fold<5:
    logger.info(f"Training on fold {args.fold} of nnunet data split")
    setup_train['patients']=plans["cross_validation"][f"fold_{args.fold}"]['train']
    setup_val['patients']=plans["cross_validation"][f"fold_{args.fold}"]['val']
else:
    logger.info("Wrong fold number")

#get dataloaders         
cd_train = CardioDataset(folder=os.path.join(plans["data_folder"], args.dataset_name+"_preprocessed", "traindata3d"),
                         z_dim=True, **setup_train)
cd_val = CardioDataset(folder=os.path.join(plans["data_folder"], args.dataset_name+"_preprocessed", "traindata3d"),
                       z_dim=True, validation=True, **setup_val)    
collator = CardioCollatorMulticlass(classes=plans["classes"], classes_data= plans["classes"])
dataloader_train = torch.utils.data.DataLoader(cd_train, batch_size=plans["batchsize3d"], collate_fn=collator,
                                                        shuffle=True)    
#for evaluation we read in all slices of the patients so it has to be batch_size=1 because of different z dimensions
dataloader_eval = torch.utils.data.DataLoader(cd_val, batch_size=1, collate_fn=collator,
                                                        shuffle=True) 

#get networks, optimizer, loss, metric and histogramm 
path_to_2d_net= os.path.join(path_results, args.dataset_name, f"fold_{args.fold}","2d-net")
params= yaml.load(open(os.path.join(path_to_2d_net, "config.json"), 'r'), Loader=yaml.SafeLoader)['network']
weights = torch.load(os.path.join(path_to_2d_net, "best_weights.pth"),  map_location=torch.device(device))
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
histogram=Histogram(plans["classes"][1:], metrics)

#create perturbation module
aug = CascadeAugmentation(probs = plans["perturbation"], data_set=args.dataset_name)


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
            out2d= aug.augment(net2d, im, gt, epoch)
        else:
            out2d = aug.augment(net2d, im, gt, epoch, do_nothing=True)
        in3d=torch.cat((im,out2d),1)
        out=net3d(in3d)
        for l in losses:
            loss+=losses[l](out, gt, deep_supervision=True)   
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net3d.parameters(), 12)
        opt.step()
        histogram.add_loss(loss)
        histogram.add_train_metrics(out[0],gt)
        steps += 1
    histogram.scale_train(steps)
        
    #evaluate 
    net3d.eval()
    steps = 0
    for im,mask in dataloader_eval:
        with torch.no_grad():
            out = predict_cascade(im, net2d, net3d, len(plans["classes"]))
            gt= torch.cat([mask[key].float() for key in mask.keys()],1)
            histogram.add_val_metrics(out,gt)
            steps += 1
    histogram.scale_val(steps)
    histogram.plot_hist(savefolder)
   
    #ceck for improvement and save best model
    val_metric=0
    for cl in plans["classes"][1:]:
        val_metric+=histogram.hist[config['metrics'][0]][f"val_{cl}"][-1]
    val_metric=val_metric/len(plans["classes"][1:])
    if val_metric > best_metric:
        best_metric=val_metric
        config["best_metric"]=float(best_metric)
        logger.info(f"New best Metric: Avg. = {best_metric}")
        histogram.print_hist(logger)
        save_checkpoint(net3d, savefolder, name = "best_weights")
        
    #adapt learning rate
    scheduler.step()
    
#save training Histogram and last network        
np.save(os.path.join(savefolder, "histogram.npy"), histogram.hist)
save_checkpoint(net3d, savefolder, name="last_weights")
json.dump(config, open(os.path.join(savefolder, "config.json"), "w"))
logger.info("Training Finished!") 
    
    
    
    
