import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import socket
import copy

import torchvision.models as backbones
from torchvision import datasets, transforms

import sys
from modules.model import getmodel
from Dataset import Dataset
import utils.loggers as logger

if __name__=="__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    models=getmodel()

    x=(torch.rand(128,809),torch.rand(128,4),torch.rand(128,12))
    for model in models[:10]:
        x=model(x)
        print(x.shape)
    x=models[10](x)
    a,b=x
    print(a.shape,b.shape)
    
    
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
    print(optimizers)
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()

    Start=time.time()
    train_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=True), batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=False), batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

    Log_all=logger.Logger('chair')
    all_log=[]
    epoch_all_log=[]
    loss_log=[]
    acc_log=[]

    for epoch in range(100):
        t=0
        c=0
        losses=0
        epoch_all_log.append(epoch)
        for model in models:
            #model=model.cuda()
            model.train()
        torch.cuda.synchronize()
        start = time.time()
        for i, data in enumerate(train_dataloader):
                target_image,target_mask,input_c,input_v,input_t=data
                #forward
                target_image=target_image.cuda(non_blocking=True)
                target_mask=target_mask.cuda(non_blocking=True)
                input_c=input_c.cuda(non_blocking=True)
                input_v=input_v.cuda(non_blocking=True)
                input_t=input_t.cuda(non_blocking=True)
                x=(input_c,input_v,input_t)
                for model in models:
                    x=model(x)
                out_image,out_mask=x
                loss1 = criterion1(out_image, target_image)
                loss2 = criterion2(out_mask, target_mask)
                loss = loss1 + 0.1 * loss2 
                for optim in optimizers:
                    if optim=="FREE":
                        continue
                    else:
                        optim.zero_grad()
                loss.backward()
                for optim in optimizers:
                    if optim=="FREE":
                        continue
                    else:
                        optim.step()
                #correct
                c+=target_image.shape[0]
                losses+=loss.detach().cpu().data.item()
                if i%100==1:
                    print("iter{}, loss: {}".format(i,losses/c))
        loss_log.append(losses/c)
        #for model in models:
        #    model=model.cpu()
        torch.cuda.synchronize()
        end = time.time()
        all_log.append(end-start)
        torch.cuda.empty_cache()
        if epoch%1==0:
            for model in models:
                model.eval()
            c=0
            losses=0
            for i, data in enumerate(test_dataloader):
                target_image,target_mask,input_c,input_v,input_t=data
                #forward
                target_image=target_image.cuda(non_blocking=True)
                target_mask=target_mask.cuda(non_blocking=True)
                input_c=input_c.cuda(non_blocking=True)
                input_v=input_v.cuda(non_blocking=True)
                input_t=input_t.cuda(non_blocking=True)
                x=(input_c,input_v,input_t)
                for model in models:
                    x=model(x)
                out_image,out_mask=x
                loss1 = criterion1(out_image, target_image)
                loss2 = criterion2(out_mask, target_mask)
                loss = loss1 + 0.1 * loss2 
                losses+=loss.detach().cpu().data.item()
                c+=target_image.shape[0]
                if i%100==1:
                    print("iter{}, acc: {}".format(i,losses/c))
            print("epoch: {} test:{}".format(epoch,losses/c))
            acc_log.append(losses/c)
        for sch in schedulers:
            sch.step(acc_log[-1])   
        
    End=time.time()
    print(End-Start)
    items=['epoch','latency','loss','acc']
    contents=[epoch_all_log,all_log,loss_log,acc_log]
    Log_all.write(items,contents) 
