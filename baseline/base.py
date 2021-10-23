import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
import pickle
import random
import socket
import copy
import argparse
import multiprocessing as mp
import queue as Queue

import torchvision.models as backbones
from torchvision import datasets, transforms

import sys
sys.path.append('../')
import models.sliced_model as sliced_model
import models.Models as Models
import models.DataLoaders as DataLoader
import hyper.Hypers as hypers
import computation as computation
import transmit.tcper as tcper
import Utils.loggers as logger
from decision.Kscheduler import Ksch

parser = argparse.ArgumentParser(description='Base method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#dataset
parser.add_argument('--data_path', type=str, default='../data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='imdb', help='Choose dataset {mnist, cifar10, tiny_imagenet, imdb}')
#split
parser.add_argument('--splits', type=int, default=5, help='splits net to multiple parts.')
parser.add_argument('--split_index', type=int, default=1, help='final index for server or start index for client')
# Optimization options
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--optim', type=str, default='imdb', help='predefined hyper-param setting {resnet,resnetpre,mobilenet,tinyimagenet,imdb}')
#stale limitation
parser.add_argument('--queue_limit', type=int, default=5, help='Limitation of Buffer Queue')
parser.add_argument('--stale_it', type=int, default=0, help='Limitation of stale epoch K*, set 0 is not using feature replay')
#EF
parser.add_argument('--use_EF', type=int, default=0, help='1 is use, 0 is not')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers (default: 2)')
#model
parser.add_argument('--model', type=str, default='BILSTM', help='model {resnet50, VGG19 ,MoblienetV3(small/large),resnet50_imgnet,VGG19_imgnet,BILSTM}')
parser.add_argument('--freeze', type=int, default=0, help='1 is type two transfer, 0 is not')
parser.add_argument('--pretrained', default='False', action='store_true', help='pretrained')
parser.add_argument('--num_classes', type=int, default=10, help='Classes of final layer')
parser.add_argument('--start_channel', type=int, default=3, help='Starting Channel')
#weight track
parser.add_argument('--track_weights', type=int, default=0, help='1 is use, 0 is not')
parser.add_argument('--track_point', nargs='+', type=int)
#Logger
parser.add_argument('--log_name', type=str, default='lstm_track', help='name of log')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
#report
parser.add_argument('--report_freq', type=int, default=100, help='Reporting frequency')

if __name__=="__main__":
    models=Models.get_model(args.model,args.splits,args.pretrained,args.num_classes,args.start_channel)

    models=models
    for model in models:
        model.train()
        model.cuda()
    optimizers=[]
    schedulers=[]
    for idx,model in enumerate(models):
        try:
            optim,sch=hypers.get_optim(model,args.optim)
            if args.freeze==1 and idx<10:
                optimizer.append('FREE')
                schedulers.append('FREE')
            else:
                optimizers.append(optim)
                schedulers.append(sch)
        except:
            optimizers.append("FREE")
    if args.model=='BILSTM':
        optimizers[0]='FREE'
        schedulers[0]='FREE'
    print(optimizers)

    split_point=args.split_index
    point=args.split_index-1
    models_edge=models[:args.split_index]
    models_cloud=models[args.split_index:]
    
    Start=time.time()
    train_dataloader = DataLoader.get_loader(args.dataset,args.data_path,args.batch_size,'train')    
    test_dataloader = DataLoader.get_loader(args.dataset,args.data_path,args.batch_size,'valid')

    Log_all=logger.Logger(args.log_name)
    all_log=[]
    epoch_all_log=[]
    loss_log=[]
    acc_log=[]

    stale_it=args.stale_it
    Kschedule=Ksch(args.dataset,args.stale_it)

    for epoch in range(args.epochs):
        if args.track_weights==1 and (epoch in args.track_point):
            #weight distribution
            params=torch.tensor([]).cuda()
            for model in models:
                for param in model.parameters():
                    params=torch.cat((params,param.data.flatten()))
            params=params.flatten()
            params=copy.copy(params).cpu().numpy()
            np.save('weight_dis/cifar10/weights_'+str(epoch)+'.npy',params)            
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
        Q_history=mp.Queue()
        Q_error=mp.Queue()
        for i in range(stale_it):
            Q_error.put("no")
        torch.cuda.synchronize()
        start = time.time()
        for i, (data, target) in enumerate(train_dataloader):
                #edge forward
                feature,E=computation.edge_forward(models_edge,data,args.use_EF)
                Q_history.put((data,target,E))
                if args.use_EF==1:
                    feature=pickle.loads(feature)
                    feature=torch.dequantize(feature)
                results,gradients=computation.cloud(models_cloud,feature,target,optimizers,point+1,args.use_EF)
                Q_error.put((i,gradients,results))
                gradients_=Q_error.get()
                if gradients_=="no":
                    continue
                it=gradients_[0]
                gradients=gradients_[1]
                results=gradients_[2]
                inputs,label,E=Q_history.get()
                if args.use_EF==1:
                    gradients=gradients-E*gradients*gradients
                if not (i-it)==stale_it:
                    print("stale fault")
                computation.edge_backward(models_edge,gradients,inputs,optimizers,point)
                tt,loss,cc=computation.summary_iter(results,label)
                t+=tt
                c+=cc
                losses+=loss
                if i%100==0:
                    print("iter{}, loss: {}, acc: {}".format(i,losses/c,t/c))
        while not Q_error.empty():
            gradients_=Q_error.get()
            it=gradients_[0]
            gradients=gradients_[1]
            results=gradients_[2]
            inputs,label,E=Q_history.get()
            if args.use_EF==1:
                gradients=gradients-E*gradients*gradients
            computation.edge_backward(models_edge,gradients,inputs,optimizers,point)
            tt,loss,cc=computation.summary_iter(results,label)
            t+=tt
            c+=cc
            losses+=loss
        print("epoch {}:{}".format(epoch,t/c))
        loss_log.append(losses/c)
        torch.cuda.synchronize()
        end = time.time()
        all_log.append(end-start)
        for sch in schedulers:
            sch.step()
        stale_it=Kschedule.step()
        if epoch%1==0:
            for model in models_edge:
                model.eval()
            for model in models_cloud:
                model.eval()
            t=0
            c=0
            for i, (data, target) in enumerate(test_dataloader):
                x=data.cuda()
                for model in models_edge:
                    x=model(x)
                for model in models_cloud:
                    x=model(x)
                result=x.detach().cpu()
                _,id=torch.max(result,1)
                correct=torch.sum(id==target.data)
                t+=correct.data.item()
                c+=target.shape[0]
                if i%100==1:
                    print("iter{}, acc: {}".format(i,t/c))
            print("test:{}".format(t/c))
            acc_log.append(t/c)
            
        
    End=time.time()
    print(End-Start)
    items=['epoch','latency','loss','acc']
    contents=[epoch_all_log,all_log,loss_log,acc_log]
    Log_all.write(items,contents)
