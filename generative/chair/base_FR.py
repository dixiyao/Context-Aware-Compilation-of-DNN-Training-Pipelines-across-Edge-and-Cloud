import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import socket
import copy
from queue import Queue

import numpy as np

import torchvision.models as backbones
from torchvision import datasets, transforms

from modules.model import getmodel
from Dataset import Dataset
import computation
import utils.loggers as logger

if __name__=="__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    models=getmodel()
    
    for model in models:
        model.train()
        model.cuda()
    optimizers=[]
    schedulers=[]
    for model in models:
        try:
            optim=torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-6)
            sch=torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, verbose=True)
            optimizers.append(optim)
            schedulers.append(sch)
        except:
            optimizers.append("FREE")
    #print(optimizers)
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()

    split_point=5
    point=4
    models_cloud=models[:split_point]
    models_edge=models[split_point:]

    train_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=True), batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=False), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    Log_all=logger.Logger('chair_stale5')
    all_log=[]
    epoch_all_log=[]
    loss_log=[]
    acc_log=[]

    stale_it=5

    for epoch in range(100):
        t=0
        c=0
        losses=0
        epoch_all_log.append(epoch)
        for model in models_edge:
            model=model.cuda()
            model.train()
        for model in models_cloud:
            model=model.cuda()
            model.train()
        Q_history=Queue()
        Q_error=Queue()
        for i in range(stale_it):
            Q_error.put("no")
        torch.cuda.synchronize()
        start = time.time()
        for i, data in enumerate(train_dataloader):
                target_image,target_mask,input_c,input_v,input_t=data
                #edge send input to cloud
                Q_history.put((input_c,input_v,input_t))
                #cloud forward
                feature=computation.cloud_forward(models_cloud,input_c,input_v,input_t)
                #edge forward and backward
                loss,gradients=computation.edge(models_edge,feature,target_image,target_mask,optimizers,point)
                #edge send gradient to cloud
                Q_error.put((i,gradients,loss))
                #cloud do feature replay
                gradients_=Q_error.get()
                if gradients_=="no":
                    continue
                it=gradients_[0]
                gradients=gradients_[1]
                loss=gradients_[2]
                if not (i-it)==stale_it:
                    print("cao")
                input_c,input_v,input_t=Q_history.get()
                computation.cloud_backward(models_cloud,gradients,input_c,input_v,input_t,optimizers,point)
                c+=input_c.shape[0]
                losses+=loss
                if i%100==0:
                    print("iter{}, loss: {}".format(i,losses/c))
        while not Q_error.empty():
            gradients_=Q_error.get()
            it=gradients_[0]
            gradients=gradients_[1]
            loss=gradients_[2]
            input_c,input_v,input_t=Q_history.get()
            computation.cloud_backward(models_cloud,gradients,input_c,input_v,input_t,optimizers,point)
            c+=input_c.shape[0]
            losses+=loss
        loss_log.append(losses/c)
        torch.cuda.synchronize()
        end = time.time()
        all_log.append(end-start)
        if epoch%1==0:
            for model in models_edge:
                model.eval()
            for model in models_cloud:
                model.eval()
            c=0
            losses=0
            for i, data in enumerate(test_dataloader):
                target_image,target_mask,input_c,input_v,input_t=data
                #forward
                target_image=target_image.cuda()
                target_mask=target_mask.cuda()
                input_c=input_c.cuda()
                input_v=input_v.cuda()
                input_t=input_t.cuda()
                x=(input_c,input_v,input_t)
                for model in models_cloud:
                    x=model(x)
                for model in models_edge:
                    x=model(x)
                out_image,out_mask=x
                loss1 = criterion1(out_image, target_image)
                loss2 = criterion2(out_mask, target_mask)
                loss = loss1 + 0.1 * loss2 
                losses+=loss.detach().cpu().data.item()
                c+=target_image.shape[0]
                if i%100==1:
                    print("iter{}, loss: {}".format(i,losses/c))
            print("epoch:{} test:{}".format(epoch,losses/c))
        for sch in schedulers:
            sch.step(loss_log[-1])   
        
    #End=time.time()
    #print(End-Start)
    items=['epoch','latency','loss']
    contents=[epoch_all_log,all_log,loss_log]
    Log_all.write(items,contents)
