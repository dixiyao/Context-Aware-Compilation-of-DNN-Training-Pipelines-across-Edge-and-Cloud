import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import lz4.frame
import pickle
import random
import socket
import copy
import multiprocessing as mp
import queue as Queue
import numpy as np
import argparse
import logging

import torchvision.models as backbones

import sys
sys.path.append('../')
import models.DataLoaders as DataLoader
import models.Models as Models
import computation as computation
import transmit.tcper as tcper
import Utils.loggers as logger
import hyper.Hypers as hypers
import computation as computation
import Utils.utils as utils
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
parser.add_argument('--stale_it', type=int, default=4, help='Limitation of stale epoch K*')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers (default: 2)')
#model
parser.add_argument('--model', type=str, default='Mobilenetv3_large', help='model')
parser.add_argument('--pretrained', default='False', action='store_true', help='pretrained')
parser.add_argument('--num_classes', type=int, default=10, help='Classes of final layer')
parser.add_argument('--start_channel', type=int, default=3, help='Starting Channel')
#profile
parser.add_argument('--runtimefile', type=str, default='../profile/runtime',help='file profiling devices runtime')
parser.add_argument('--featuresizefile', type=str, default='../profile/featuresize',help='file profiling feature sizes')
parser.add_argument('--qtime', type=float,default='0')
#devices
parser.add_argument('--cloud_device', type=str, default='Cloud',help='devices used for server')
parser.add_argument('--edge_device', type=str, default='TX2',help='devices used for client')
#Logger
parser.add_argument('--log_name', type=str, default='dynamic_wiredmobilelarge', help='name of log')
parser.add_argument('--it_log_name', type=str, default='it_d', help='name of per it log')
#transfer
parser.add_argument('--ip', type=str, default='202.120.36.29', help='ip of server address')
parser.add_argument('--port', type=int, default=1883, help='TCP port of server')
parser.add_argument('--hyperport', type=int, default=1884, help='TCP port of server for model update')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
#report
parser.add_argument('--report_freq', type=int, default=100, help='Reporting frequency')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('log_'+args.log_name+'.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def download_edge(client,Q4,E4,fix,up,down):
    while True:
            head,epoch,iters,result,gradient,client_send,server_rec,client_send_size,server_send,rec_data_length=client.recieve_tensor()
            if head=='Train':
                rec_time=time.time()
                up.value=client_send_size/(server_rec-(client_send+fix))/1024/1024
                down.value=rec_data_length/(rec_time+fix-server_send)/1024/1024
            if head=='warmup':
                continue
            Q4.put((head,epoch,iters,result,gradient))
            E4.set()
            if head=='Termi':
                break
    time.sleep(5)

def upload_edge(client,Q1,E1,Efull):
    while True:
        if not Q1.empty():
            a,b,c,d,e,f=Q1.get()
            Efull.set()
            client.send_tensor(a,b,c,d,e,f)
            if a=='Termi':
                break
        else:
            E1.wait()
    time.sleep(5)
    
if __name__=="__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    global_models=Models.get_model(args.model,args.splits,args.pretrained,args.num_classes,args.start_channel)
    models=global_models[:args.split_index]
    for model in models:
        model.train()
        model=model.cuda()
    use_Q=True
   
    edge,cloud,feature_size=utils.get_profile(args.runtimefile,args.featuresizefile,args.model,args.cloud_device,args.edge_device)
    model_size=utils.count_models_size_in_MB(global_models)
    print(model_size)
    
    optims=[]
    schedulers=[]
    for model in global_models:
        try:
            optim,sch=hypers.get_optim(model,args.optim)
            optims.append(optim)
            schedulers.append(sch)
        except:
            optims.append("FREE")
            schedulers.append("FREE")
    if args.model=='BILSTM':
        optims[0]='FREE'
        schedulers[0]='FREE'
    print(optims)
    train_dataloader = DataLoader.get_loader(args.dataset,args.data_path,args.batch_size,'train')    
    test_dataloader = DataLoader.get_loader(args.dataset,args.data_path,args.batch_size,'valid')

    client=tcper.Client(args.ip,args.port)
    client_start=time.time()
    print(client_start)
    time.sleep(3)
    hyperclient=tcper.Client(args.ip,args.hyperport)

    a,b,c,d,e,f,g,h,i,j=client.recieve_tensor()
    server_start=g
    print(server_start)
    client_slower_server=server_start-client_start
    #shared memory
    Q1=mp.Queue(args.queue_limit)
    Q4=mp.Queue()
    Q_history=mp.Queue()
    E1=mp.Event()
    E4=mp.Event()
    Efull=mp.Event()
    Efull.set()
    up=mp.Value('f',0.0)
    down=mp.Value('f',0.0)

    pupload=mp.Process(target=upload_edge,args=(client,Q1,E1,Efull))
    pdownload=mp.Process(target=download_edge,args=(client,Q4,E4,client_slower_server,up,down))
    pdownload.start()
    pupload.start()

    #log
    Log_all=logger.Logger(args.log_name)
    Log_it=logger.Logger(args.it_log_name)
    epoch_log=[]
    latency_log=[]
    loss_log=[]
    acc_log=[]
    upband_log=[]
    downband_log=[]
    point_log=[]
    stale_log=[]
    uQ_log=[]
    
    for model in models:
        model=model.cuda()

    meter=utils.AvgrageMeter()
    stale_it=args.stale_it
    Kschedule=Ksch(args.dataset,args.stale_it)
    current_back=-1
    #slice 0,...,point is on edge
    point=args.split_index-1
    #used for control estimated remain
    history_remain=1
    remain=len(train_dataloader)
    beta=0.8
 
    for epoch in range(args.epochs):
        for model in models:
            model.train()
        epoch_log.append(epoch)
        torch.cuda.synchronize()
        start=time.time()
        latency_log.append(start)
        
        for i, (data, target) in enumerate(train_dataloader):
                s=time.time()
                if point==args.splits-1:
                    computation.edge(i,models,data,target,meter,optims)
                    utils.check_full(Q1,Efull)
                    Q1.put(('Edge',epoch,i,torch.tensor([-1.1]),torch.tensor([-1.1]),False))
                    E1.set()
                else:
                    upload_feature,E=computation.edge_forward(models,data,use_Q)
                    Q_history.put((epoch,i,data,target,E))
                    utils.check_full(Q1,Efull)
                    Q1.put(('Train',epoch,i,target,upload_feature,use_Q))
                    E1.set()
                #backward
                while True:
                    while not Q4.empty():
                        head,b_epoch,b_iter,result,download_gradient=Q4.get()
                        stale_log.append(i-b_iter)
                        if not (head=='Train' or head=="Edge"):
                            print(head)
                        item,topic=computation.edge_backprocess(head,b_epoch,b_iter,
                                                                result,download_gradient,
                                                                Q_history,meter,models,optims,point)
                        current_back=b_iter
        
                    if i-current_back<=stale_it:
                        break
                #dynamic decision
                upload=up.value
                download=down.value
                if upload==0 or download==0:
                    continue
                upband_log.append(upload)
                downband_log.append(download)
                point_log.append(point)
                logging.info('current epoch '+str(epoch)+'and point '+str(point))
                uQ_log.append(use_Q)
                if stale_it>0:
                    estimate_latency,new_point,use_Q=computation.dynamic_decision(upload,download,models,global_models,remain,
                                                       edge,cloud,feature_size,model_size,
                                                       args.stale_it,point,args.qtime)
                    
                    if not point==new_point:
                        history_remain=1
                        print("estimate latency: {}".format(estimate_latency))
                        print("current point: {}".format(new_point))
                        utils.check_full(Q1,Efull)
                        Q1.put(('change',point,new_point,torch.tensor([-1.1]),torch.tensor([-1.1]),False))
                        E1.set()
                        while True:
                            if not Q4.empty():
                                head,b_epoch,b_iter,result,download_gradient=Q4.get()
                                item,topic=computation.edge_backprocess(head,b_epoch,b_iter,
                                                                        result,download_gradient,
                                                                        Q_history,meter,models,optims,point)
                                if head=='Train':
                                    stale_log.append(i-b_iter)
                                if item=='EndTrain':        
                                    loss,acc,end=topic
                                    latency_log[b_epoch]=end-latency_log[b_epoch]
                                    loss_log.append(loss)
                                    print('train %03d %e %f'%(b_epoch,loss,acc))
                                if head=='change':
                                    break
                            else:
                                E4.wait()
                        _,check_point,_=computation.dynamic_decision(upload,download,models,global_models,remain,
                                                       edge,cloud,feature_size,model_size,
                                                       args.stale_it,point,args.qtime)
                        if not check_point==new_point:
                            new_point=point
                        else:
                            models=computation.dynamic_change(hyperclient,models,global_models,point,new_point)
                    else:
                        history_remain+=1
                    point=new_point
                    remain=remain*beta+(1-beta)*history_remain
        for sch in schedulers[:point+1]:
            if sch=="FREE":
                pass
            else:
                sch.step()
        stale_it=Kschedule.step()
        logging.info(epoch_log)
        logging.info(latency_log)
        utils.check_full(Q1,Efull)       
        Q1.put(('EndTrain',epoch,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),False))
        E1.set()
        while True:
            if not Q4.empty():
                    head,b_epoch,b_iter,result,download_gradient=Q4.get()
                    if head=='Train':
                        stale_log.append(i-b_iter)
                    if not (head=='Train'or head=='EndTrain'):
                        print(head)
                    item,topic=computation.edge_backprocess(head,b_epoch,b_iter,
                                                            result,download_gradient,
                                                            Q_history,meter,models,optims,point)
                    if item=='EndTrain':
                        loss,acc,end=topic
                        latency_log[b_epoch]=end-latency_log[b_epoch]
                        loss_log.append(loss)
                        print('train %03d %e %f'%(b_epoch,loss,acc))
                        break
            else:
                E4.wait()
        
        #test
        for model in models:
            model.eval()
        for i, (data, target) in enumerate(test_dataloader):
            if i>300:
                break
            Q_history.put((data,target))
            x=data.cuda()
            for model in models:
                x=model(x)
            x=copy.copy(x.detach().cpu())
            utils.check_full(Q1,Efull)
            Q1.put(('Valid',epoch,i,target,x.detach().cpu(),False))
            E1.set()
            if not Q4.empty():
                    head,b_epoch,b_iter,result,download_gradient=Q4.get()
                    if head=='Train'or head=='EndTrain':
                        print(head)
                    if head=='END':
                        break
                    else:
                        item,topic=computation.edge_backprocess(head,b_epoch,b_iter,
                                                            result,download_gradient,
                                                            Q_history,meter,models,optims,point)
                        if item=='EndValid':
                            acc=topic
                            acc_log.append(acc)
                            print('valid %03d %f'%(b_epoch,acc))
        utils.check_full(Q1,Efull)
        Q1.put(('EndValid',epoch,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),False))
        E1.set()
        utils.check_full(Q1,Efull)
        Q1.put(('END',-1,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),False))
        #E1.set()
        while True:
            if not Q4.empty():
                    head,b_epoch,b_iter,result,download_gradient=Q4.get()
                    if head=='Train'or head=='EndTrain':
                        print(head)
                    if head=='END':
                        break
                    else:
                        item,topic=computation.edge_backprocess(head,b_epoch,b_iter,
                                                            result,download_gradient,
                                                            Q_history,meter,models,optims,point)
                        if item=='EndValid':
                            acc=topic
                            acc_log.append(acc)
                            print('valid %03d %f'%(b_epoch,acc))
            else:
                E4.wait()

    items=['epoch','latency','loss','acc']
    contents=[epoch_log,latency_log,loss_log,acc_log]
    Log_all.write(items,contents)
    items=['up','down','point','stale','useQ']
    contents=[upband_log,downband_log,point_log,stale_log[:len(upband_log)],uQ_log]
    Log_it.write(items,contents)
    
    
    #terminate
    utils.check_full(Q1,Efull)
    Q1.put(('Termi',-1,0,torch.tensor([-1.1]),torch.tensor([-1.1]),False))
    E1.set()
    time.sleep(5)

