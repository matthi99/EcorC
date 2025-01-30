# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:54:10 2022

@author: A0067501
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sys
import random
from skimage import measure
from utils.unet import UNet, UNet2D


def get_network(architecture="unet", device="cuda:0", **kwargs):
    architecture = architecture.lower()
    if architecture == "unet":
        net = UNet(**kwargs)
    elif architecture == "unet2d":
        net = UNet2D(**kwargs)

    else:
        net = UNet(**kwargs)
    return net.to(device=device)


class Histogram():
    def __init__(self, classes, metrics):
        self.classes = classes
        self.metrics = metrics
        self.hist = self.prepare_hist()
        
    def prepare_hist(self):
        hist={}
        hist['loss'] = []
        for m in self.metrics:
            hist[m]={}
            hist[m]["train_mean"] = [] 
            hist[m]["val_mean"] = []
            for cl in self.classes:
                hist[m][f"train_{cl}"] = [] 
                hist[m][f"val_{cl}"] = []
        return hist
    
    def append_hist(self):
        self.hist['loss'].append(0)
        for m in self.metrics:
            self.hist[m]["train_mean"].append(0)
            self.hist[m]["val_mean"].append(0)
            for cl in self.classes:
                self.hist[m][f"train_{cl}"].append(0) 
                self.hist[m][f"val_{cl}"].append(0)
        
    def add_loss(self, loss):
        self.hist['loss'][-1]+=loss.item()
    
    def add_train_metrics(self, out, gt):
        #make output binary
        with torch.no_grad():
            temp=torch.argmax(out,1).long()
            temp=torch.nn.functional.one_hot(temp,len(self.classes)+1)
            out=torch.moveaxis(temp, -1, 1)
        for m in self.metrics:
            with torch.no_grad():
                values=self.metrics[m](out,gt).cpu().detach().numpy()
            if len(values)== len(self.classes):
                self.hist[m]["train_mean"][-1]+=values.mean()
                for cl, i in zip(self.classes, range(len(values))):
                    self.hist[m][f"train_{cl}"][-1]+=values[i]
            else:
                self.hist[m]["train_mean"][-1]+=values.mean()
                
    def add_val_metrics(self, out, gt):
        #make output binary
        with torch.no_grad():
            temp=torch.argmax(out,1).long()
            temp=torch.nn.functional.one_hot(temp,len(self.classes)+1)
            out=torch.moveaxis(temp, -1, 1)
        for m in self.metrics:
            with torch.no_grad():
                values=self.metrics[m](out,gt).cpu().detach().numpy()
            if len(values)== len(self.classes):
                self.hist[m]["val_mean"][-1]+=values.mean()
                for cl, i in zip(self.classes, range(len(values))):
                    self.hist[m][f"val_{cl}"][-1]+=values[i]
            else:
                self.hist[m]["val_mean"][-1]+=values.mean()
                
    def scale_train(self,steps):
        self.hist["loss"][-1]/=steps
        for m in self.metrics:
            self.hist[m]["train_mean"][-1]/=steps
            for cl in self.classes:
                self.hist[m][f"train_{cl}"][-1]/=steps
    
    def scale_val(self,steps):
        for m in self.metrics:
            self.hist[m]["val_mean"][-1]/=steps
            for cl in self.classes:
                self.hist[m][f"val_{cl}"][-1]/=steps
    
    def print_hist(self,logger):
        logger.info("Training:")
        loss=self.hist["loss"][-1]
        logger.info(f"Loss = {loss}")
        for m in self.metrics:
            value = self.hist[m]["train_mean"][-1]
            logger.info(f"Mean {m} = {value}")
            for cl in self.classes:
                value = self.hist[m][f"train_{cl}"][-1]
                logger.info(f"{cl} {m} = {value}")
        logger.info("Validation:")
        for m in self.metrics:
            value = self.hist[m]["val_mean"][-1]
            logger.info(f"Mean {m} = {value}")
            for cl in self.classes:
                value = self.hist[m][f"val_{cl}"][-1]
                logger.info(f"{cl} {m} = {value}")
    
    def plot_hist(self ,savefolder):
        #plot loss
        plt.figure()
        plt.plot(np.array(self.hist["loss"]))
        plt.title("Loss")
        plt.savefig(os.path.join(savefolder,"Loss.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        for m in self.metrics:
            # plot mean metric
            plt.figure()
            plt.plot(np.array(self.hist[m]["train_mean"]))
            plt.plot(np.array(self.hist[m]["val_mean"]))
            plt.title(f"Mean {m}")
            plt.savefig(os.path.join(savefolder,f"Mean {m}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            #plot metric for different classes
            for cl in self.classes:
                plt.figure()
                plt.plot(np.array(self.hist[m][f"train_{cl}"]))
                plt.plot(np.array(self.hist[m][f"val_{cl}"]))
                plt.savefig(os.path.join(savefolder,f"{m} {cl}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
                plt.close()
            

def save_checkpoint(net, checkpoint_dir, name="weights"):
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(net.state_dict(), checkpoint_path)
    

class CascadeAugmentation():
    def __init__(self, probs, data_set):
        self.probs = probs
        self.data_set = data_set
        
    
    def delete_class(self, out2d):
        sl=np.random.randint(0,out2d.shape[1])
        if self.data_set == "EMIDEC":
            if np.random.uniform() < 0.4:
                out2d[0,sl,...]=0
            else:
                out2d[1,sl,...]=0
            return out2d
        elif self.data_set == "MyoPS":
            out2d[0,sl,...]=0
            return out2d
        
    @staticmethod
    def delete_slices(out2d):
        slice_nrs=random.choice(range(1,4))
        slices=random.sample(range(out2d.shape[1]),slice_nrs)
        for sl in slices:
            out2d[:,sl,...]=0
        return out2d
        
    @staticmethod
    def delete_all(out2d):
        out2d=torch.zeros_like(out2d)
        return out2d
    
    @staticmethod
    def add_scar(out2d, muscle):
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sl=np.random.randint(0,out2d.shape[1])
        muscle= muscle[sl].cpu().detach().numpy()
        temp=muscle[muscle!=0]
        if len(temp)!=0:
            cut = np.random.uniform(70,90)
            per = np.percentile(temp, cut)
            muscle[muscle<per]=0
            muscle[muscle!=0]=1
            labels = measure.label(muscle)
            assert( labels.max() != 0 ) # assume at least 1 CC
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
            created_scar = largestCC*1
            created_scar = torch.from_numpy(created_scar).to(device)
            out2d[0,sl,...]+=((1-out2d[0,sl,...])*created_scar)
        return out2d
    
    @staticmethod
    def add_mvo(out2d):
        scar= out2d[0,...].clone().detach()
        mvo = out2d[1,...].clone().detach()
        if torch.sum(scar)!=0:
            indices= scar.nonzero()
            index=indices[np.random.randint(0,len(indices))].cpu().detach().numpy()
            n=[]
            for i in range(-1,2):
                for j in range(-1,2):
                    temp=torch.tensor(index)
                    temp[-1]+=i
                    temp[-2]+=j
                    n.append(temp)
            for pixel in n:
                if np.random.uniform() < 0.5:
                    scar[pixel[0], pixel[1], pixel[2]]=0
                    mvo[pixel[0], pixel[1], pixel[2]]=1
            out2d[0,...]=scar
            out2d[1,...]=mvo
        return out2d
     
    @staticmethod
    def remove_scar(out2d, scar):
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        full_scar= scar.cpu().detach().numpy()
        
        sl=np.random.randint(0,out2d.shape[1])
        scar= full_scar[sl]
        temp=scar[scar!=0]
        if len(temp)!=0:
            cut = np.random.uniform(30,50)
            per = np.percentile(temp, cut)
            scar[scar>per]=0
            scar[scar!=0]=1
            labels = measure.label(scar)
            assert( labels.max() != 0 ) # assume at least 1 CC
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
            remove_scar = largestCC*1
            remove_scar = torch.from_numpy(remove_scar).to(device)
            
            out2d[0,sl,...]-= (out2d[0,sl,...]*remove_scar)
        return out2d    
        
    @staticmethod
    def nothing(out2d):
        return out2d
    
    def augment(self, net2d, im, gt, epoch, per_batch =True, do_nothing=False):
        in2d=torch.moveaxis(im,2,0)
        out2d=[]
        if self.data_set=="EMIDEC":
            disruption_list = [self.delete_class, self.delete_slices, self.delete_all, self.add_scar, self.add_mvo,  self.nothing]
            with torch.no_grad():
                for b in range(im.shape[0]):
                    temp=net2d(in2d[:,b,...])[0]
                    temp=torch.moveaxis(temp,0,1)
                    temp=torch.argmax(temp,0).long()
                    temp=torch.nn.functional.one_hot(temp,5)
                    temp=torch.moveaxis(temp,-1,0)[3:,...].float()
                    if do_nothing:
                        index=-1
                    else:
                        index=np.random.choice(np.arange(6), p=self.probs)
                    fun= disruption_list[index]
                    if index == 3:
                        muscle= gt[b,2,...]*im[b,0,...]
                        out2d.append(fun(temp,muscle))
                    else:
                        out2d.append(fun(temp))
                out2d=torch.stack(out2d,0)
            return out2d
        elif self.data_set == "MyoPS":
            disruption_list = [self.delete_class, self.delete_slices, self.delete_all, self.add_scar, self.remove_scar,  self.nothing]
            with torch.no_grad():
                for b in range(im.shape[0]):
                    temp=net2d(in2d[:,b,...])[0]
                    temp=torch.moveaxis(temp,0,1)
                    temp=torch.argmax(temp,0).long()
                    temp=torch.nn.functional.one_hot(temp,4)
                    temp=torch.moveaxis(temp,-1,0)[3:4,...].float()
                    if do_nothing:
                        index=-1
                    else:
                        index=np.random.choice(np.arange(6), p=self.probs)
                    fun= disruption_list[index]
                    if index == 3:
                        muscle= gt[b,2,...]*im[b,0,...]
                        out2d.append(fun(temp,muscle))
                    elif index == 4:
                        scar= gt[b,3,...]*im[b,0,...]
                        out2d.append(fun(temp,scar))
                    else:
                        out2d.append(fun(temp))

                out2d=torch.stack(out2d,0)
            return out2d
        else:
            print("Wrong dataset_name! Possible names are EMIDEC or MyoPS.")


def get_logger(name, level=logging.INFO, formatter = '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s'):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.handler_set = True
    return logger

def predict2d(im, net):
    out = net(im)[0]
    out += torch.flip(net(torch.flip(im, dims=[2]))[0], dims=[2])
    out +=  torch.flip(net(torch.flip(im, dims=[3]))[0], dims=[3])
    return out

def predict2d_cascade(im,net2d, classes):
    in2d=torch.moveaxis(im,2,0)
    out2d=[]
    for b in range(im.shape[0]):
        with torch.no_grad():
            temp=net2d(in2d[:,b,...])[0]
        temp=torch.moveaxis(temp,0,1)
        temp=torch.argmax(temp,0).long()
        temp=torch.nn.functional.one_hot(temp,classes)
        temp=torch.moveaxis(temp,-1,0)
        out2d.append(temp[3:classes])
    out2d=torch.stack(out2d,0)
    return out2d

def predict_cascade(im, net2d, net3d, classes):
    out2d = predict2d_cascade(im,net2d, classes)
    in3d=torch.cat((im,out2d),1)
    out=net3d(in3d)[0]
    
    out2d = torch.flip(predict2d_cascade(torch.flip(im, dims=[3]),net2d, classes), dims =[3])
    in3d=torch.cat((im,out2d),1)
    out+=net3d(in3d)[0]
    
    out2d = torch.flip(predict2d_cascade(torch.flip(im, dims=[4]),net2d, classes), dims =[4])
    in3d=torch.cat((im,out2d),1)
    out+=net3d(in3d)[0]
    return out
    