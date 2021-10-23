import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import socket
import copy
import multiprocessing as mp
import queue as Queue
import numpy as np
import argparse 

import torchvision.models as backbones

import sys
from modules.model import getmodel
from Dataset import Dataset
import utils.loggers as logger
import utils.utils as utils
import computation
import tcper

parser = argparse.ArgumentParser(description='Base method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#split
parser.add_argument('--split_index', type=int, default=1, help='final index for server or start index for client')
# Optimization options
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--data_path', type=str, default='../data', help='Choose dataset.')
#stale limitation
parser.add_argument('--stale_it', type=int, default=5, help='Limitation of stale epoch K*')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers (default: 2)')
#profile
parser.add_argument('--runtimefile', type=str, default='../profile/runtime',help='file profiling devices runtime')
parser.add_argument('--featuresizefile', type=str, default='../profile/featuresize',help='file profiling feature sizes')
parser.add_argument('--qtime', type=float,default='0.3')
#devices
parser.add_argument('--cloud_device', type=str, default='Cloud',help='devices used for server')
parser.add_argument('--edge_device', type=str, default='TX2',help='devices used for client')
#Logger
parser.add_argument('--log_name', type=str, default='dynamic_wiredmobilelarge', help='name of log')
#transfer
parser.add_argument('--ip', type=str, default='127.0.0.1', help='ip of server address')
parser.add_argument('--port', type=int, default=1883, help='TCP port of server')
parser.add_argument('--hyperport', type=int, default=1884, help='TCP port of server for model update')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
#report
parser.add_argument('--report_freq', type=int, default=100, help='Reporting frequency')

def download_edge(client,Q4,E4,fix,up,down):
    while True:
            head,epoch,iters,gradient,client_send,server_rec,client_send_size,server_send,rec_data_length=client.recieve_tensor()
            if head=='Train':
                rec_time=time.time()
                up.value=client_send_size/(server_rec-(client_send+fix))/1024/1024
                down.value=rec_data_length/(rec_time+fix-server_send)/1024/1024
                #print(iters,client_send_size,client_send,server_rec,rec_data_length, server_send,rec_time)
            if head=='warmup':
                continue
            Q4.put((head,epoch,iters,gradient))
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
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    global_models=getmodel()
    models=global_models[:args.split_index]
    for model in models:
        model.train()
        model=model.cuda()
    use_Q=True
   
    edge,cloud,feature_size=utils.get_profile(args.runtimefile,args.featuresizefile,'chair',args.cloud_device,args.edge_device)
    model_size=utils.count_models_size_in_MB(global_models)
    print(model_size)
    
    optims=[]
    schedulers=[]
    for model in global_models:
        try:
            optim=torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-6)
            sch=torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, verbose=True)
            optims.append(optim)
            schedulers.append(sch)
        except:
            optims.append("FREE")
            schedulers.append("FREE")
    print(optims)
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()
    
    train_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=False), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=False), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    server=tcper.Server(args.ip,args.port)
    client_start=time.time()
    print(client_start)
    a,b,c,e,f,g,h,i,j=server.recieve_tensor()
    server_start=g
    print(server_start)
    client_slower_server=server_start-client_start
    hyperserver=tcper.Server(args.ip,args.hyperport)

    
    #shared memory
    Q1=mp.Queue(2*args.stale_it)
    Q4=mp.Queue()
    Q_history=mp.Queue()
    E1=mp.Event()
    E4=mp.Event()
    Efull=mp.Event()
    Efull.set()
    up=mp.Value('f',0.0)
    down=mp.Value('f',0.0)

    pupload=mp.Process(target=upload_edge,args=(server,Q1,E1,Efull))
    pdownload=mp.Process(target=download_edge,args=(server,Q4,E4,client_slower_server,up,down))
    pdownload.start()
    pupload.start()


    
    for model in models:
        model=model.cuda()

    stale_it=args.stale_it
    current_back=-1
    #slice 0,...,point is on edge
    point=args.split_index-1
    #used for control estimated remain
    history_remain=1
    remain=390
    beta=0.8
 
    for epoch in range(args.epochs):
        for model in models:
            model.train()
        for i, data in enumerate(train_dataloader):
                target_image,target_mask,input_c,input_v,input_t=data
                index=-1
                image=target_image
                mask=target_mask
                c=input_c
                v=input_v
                t=input_t
                upload_feature,E=computation.cloud_forward(models,c,v,t,use_Q)
                #print(i,"forward",time.time()-s)
                Q_history.put((epoch,i,c,v,t,E))
                utils.check_full(Q1,Efull)
                Q1.put(('Train',epoch,i,index,upload_feature,use_Q))
                E1.set()
                #backward
                while True:
                    while not Q4.empty():
                        head,b_epoch,b_iter,download_gradient=Q4.get()
                        if not (head=='Train' or head=="Edge"):
                            print(head)
                        #print('back {}, now {}'.format(b_iter,i))
                        computation.cloud_backprocess(models,download_gradient,Q_history,optims,point)
                        current_back=b_iter
        
                    if i-current_back<=stale_it:
                        break
                #dynamic decision
                upload=up.value
                download=down.value
                #print(upload,download)
                if upload==0 or download==0:
                    continue
                
                estimate_latency,new_point,use_Q=computation.dynamic_decision(upload,download,models,global_models,remain,
                                                   edge,cloud,feature_size,model_size,
                                                   args.stale_it,point,args.qtime)
                upband_log.append(upload)
                downband_log.append(download)
                point_log.append(new_point)
                if not point==new_point:
                    history_remain=1
                    print("estimate latency: {}".format(estimate_latency))
                    print("current point: {}".format(new_point))
                    utils.check_full(Q1,Efull)
                    Q1.put(('change',point,new_point,-1,torch.tensor([-1.1]),False))
                    E1.set()
                    while True:
                        if not Q4.empty():
                            head,b_epoch,b_iter,result,download_gradient=Q4.get()
                            item,topic=computation.edge_backprocess(head,b_epoch,b_iter,download_gradient,
                                                                    Q_history,models,optims,point)
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
                        models=computation.dynamic_change(hyperserver,models,global_models,point,new_point)
                else:
                    history_remain+=1
                point=new_point
                remain=remain*beta+(1-beta)*history_remain
                
        utils.check_full(Q1,Efull)       
        Q1.put(('EndTrain',epoch,-1,-1,torch.tensor([-1.1]),False))
        E1.set()
        while True:
            if not Q4.empty():
                    head,b_epoch,b_iter,download_gradient=Q4.get()
                    if not (head=='Train'or head=='EndTrain'):
                        print(head)
                    #print('now {}'.format(b_iter))
                    computation.cloud_backprocess(models,download_gradient,Q_history,optims,point)
                    if item=='EndTrain':
                        break
            else:
                E4.wait()
    
    
    #terminate
    utils.check_full(Q1,Efull)
    Q1.put(('Termi',-1,0,-1,torch.tensor([-1.1]),False))
    E1.set()
    time.sleep(5)

