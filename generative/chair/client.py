import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
import random
import socket
import copy
import multiprocessing as mp
import queue as Queue
import numpy as np
import argparse 

import torchvision.models as backbones
from torchvision import datasets, transforms

import sys
from modules.model import getmodel
from Dataset import Dataset
import utils.loggers as logger
import utils.utils as utils
import computation
import tcper

parser = argparse.ArgumentParser(description='Base method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#model
parser.add_argument('--data_path', type=str, default='../data', help='Choose dataset.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
#split
parser.add_argument('--split_index', type=int, default=1, help='final index for server or start index for client')
#stale limitation
parser.add_argument('--queue_limit', type=int, default=5, help='Limitation of Buffer Queue')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
#transfer
parser.add_argument('--log_name', type=str, default='dynamic_wiredmobilelarge', help='name of log')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='ip of server address')
parser.add_argument('--port', type=int, default=1883, help='TCP port of server')
parser.add_argument('--hyperport', type=int, default=1884, help='TCP port of server for model update and sppedtest')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

def upload_cloud(server,Q2,E2):
    while True:
            head,epoch,iters,index,feature,useQ,server_rec,client_send,send_size=server.recieve_tensor()
            if useQ:
                feature=torch.dequantize(feature)
            Q2.put((head,epoch,iters,index,feature,useQ,server_rec,client_send,send_size))
            E2.set()
            if head=='Termi':
                break
    time.sleep(5)

def download_cloud(server,Q3,E3,Efull):
    while True:
        if not Q3.empty():
            a,b,c,d,e,f,g=Q3.get()
            Efull.set()
            server.send_tensor(a,b,c,d,e,f,g)
            if a=='Termi':
                break
        else:
            E3.wait()
    time.sleep(5)
        
if __name__=="__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    global_models=getmodel()
    models=global_models[args.split_index:]
    for model in models:
        model=model.cuda()
        model.train()
        
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

    train_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=True), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(Dataset('../data', is_train=False), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    train_d=iter(train_dataloader)
    
    client=tcper.Client(args.ip,args.port)
    server_start=time.time()
    print(server_start)
    client.send_tensor('begin',-1,-1,torch.tensor([-1.1]),server_start,1,1)
    time.sleep(3)
    hyperclient=tcper.Client(args.ip,args.hyperport)

    meter=utils.AvgrageMeter()

    #shared memory
    Q2=mp.Queue()
    Q3=mp.Queue(args.queue_limit)
    E2=mp.Event()
    E3=mp.Event()
    Efull=mp.Event()
    Efull.set()
    
    pupload=mp.Process(target=upload_cloud,args=(client,Q2,E2))
    pdownload=mp.Process(target=download_cloud,args=(client,Q3,E3,Efull))
    pdownload.start()
    pupload.start()

    #log
    Log_all=logger.Logger(args.log_name)
    epoch_log=[]
    latency_log=[]
    loss_log=[]
    
    for model in models:
        model=model.cuda()
    last_epoch=-1
    point=args.split_index
    #point,point+1,.... are on cloud
    torch.cuda.synchronize()
    start=time.time()
    while True:
        if not Q2.empty():
            #print("cloud")
            head,epoch,i,index,upload_feature,useQ,server_rec,client_send,send_size=Q2.get()
            print(head)
            if head=='change':
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,download_gradient,server_rec,client_send,send_size))
                E3.set()
                point,models=computation.cloud_dynamic_change_model(global_models,models,point,epoch,i,hyperclient)
                continue
            elif head=='Termi':
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
                break
            elif head=='Train':
                for model in models:
                    model.train()
                image,mask,c,v,t=next(train_d)
                if point==0:
                    upload_feature=c,v,t
                result,download_gradient=computation.egde(models,upload_feature,image,mask,optims,point,useQ)
                meter.update(result,result,c.shape[0])
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,download_gradient,server_rec,client_send,send_size))
                E3.set()
            elif head=="EndTrain":
                for sch in schedulers[point:]:
                    if sch=="FREE":
                        pass
                    else:
                        sch.step()
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
                torch.cuda.synchronize()
                epoch_log.append(epoch)
                loss_log.append(meter.losses/meter.cnt)
                latency_log.append(time.time()-start)
            else:
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
        else:
            E2.wait()
    time.sleep(5)

    items=['epoch','latency','loss']
    contents=[epoch_log,latency_log,loss_log]
    Log_all.write(items,contents)
