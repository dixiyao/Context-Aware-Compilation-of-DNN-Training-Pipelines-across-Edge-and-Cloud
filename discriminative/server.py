import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

import torchvision.models as backbones
from torchvision import datasets, transforms

import sys
sys.path.append('../')
import models.Models as Models
import computation as computation
import transmit.tcper as tcper
import hyper.Hypers as hypers
import Utils.utils as utils

parser = argparse.ArgumentParser(description='Base method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#model
parser.add_argument('--model', type=str, default='resnet50', help='model used')
parser.add_argument('--pretrained', default='True', action='store_true', help='pretrained')
parser.add_argument('--num_classes', type=int, default=10, help='Classes of final layer')
parser.add_argument('--start_channel', type=int, default=3, help='Starting Channel')
#split
parser.add_argument('--splits', type=int, default=5, help='splits net to multiple parts.')
parser.add_argument('--split_index', type=int, default=1, help='final index for server or start index for client')
# Optimization options
parser.add_argument('--optim', type=str, default='resnetpre', help='predefined hyper-param setting')
#stale limitation
parser.add_argument('--queue_limit', type=int, default=5, help='Limitation of Buffer Queue')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
#transfer
parser.add_argument('--ip', type=str, default='0.0.0.0', help='ip of server address')
parser.add_argument('--port', type=int, default=1883, help='TCP port of server')
parser.add_argument('--hyperport', type=int, default=1884, help='TCP port of server for model update and sppedtest')
parser.add_argument('--speedtestport', type=int, default=1885, help='TCP port of speed test')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

def upload_cloud(server,Q2,E2):
    while True:
            head,epoch,iters,target,feature,useQ,server_rec,client_send,send_size=server.recieve_tensor()
            if useQ:
                features=lz4.frame.decompress(feature)
                feature=pickle.loads(feature)
                feature=torch.dequantize(feature)
            Q2.put((head,epoch,iters,target,feature,useQ,server_rec,client_send,send_size))
            E2.set()
            if head=='Termi':
                break
    time.sleep(5)

def download_cloud(server,Q3,E3,Efull):
    while True:
        if not Q3.empty():
            a,b,c,d,e,f,g,h=Q3.get()
            Efull.set()
            server.send_tensor(a,b,c,d,e,f,g,h)
            if a=='Termi':
                break
        else:
            E3.wait()
    time.sleep(5)
        
if __name__=="__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    global_models=Models.get_model(args.model,args.splits,args.pretrained,args.num_classes,args.start_channel)
    models=global_models[args.split_index:]
    for model in models:
        model=model.cuda()
        model.train()
        
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

    server=tcper.Server(args.ip,args.port)
    server_start=time.time()
    print(server_start)
    hyperserver=tcper.Server(args.ip,args.hyperport)

    server.send_tensor('begin',-1,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),server_start,1,1)

    #shared memory
    Q2=mp.Queue()
    Q3=mp.Queue(args.queue_limit)
    E2=mp.Event()
    E3=mp.Event()
    Efull=mp.Event()
    Efull.set()
    
    pupload=mp.Process(target=upload_cloud,args=(server,Q2,E2))
    pdownload=mp.Process(target=download_cloud,args=(server,Q3,E3,Efull))
    pdownload.start()
    pupload.start()
    
    for model in models:
        model=model.cuda()
    last_epoch=-1
    point=args.split_index
    #point,point+1,.... are on cloud
    
    while True:
        if not Q2.empty():
            #print("cloud")
            head,epoch,i,label,upload_feature,useQ,server_rec,client_send,send_size=Q2.get()
            if head=='change':
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,result,download_gradient,server_rec,client_send,send_size))
                E3.set()
                point,models=computation.cloud_dynamic_change_model(global_models,models,point,epoch,i,hyperserver)
                continue
            elif head=='Termi':
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,torch.tensor([-2.1]),torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
                break
            elif head=='Valid':
                for model in models:
                    model.eval()
                outputs=Variable(upload_feature, requires_grad=False).cuda()
                x=outputs
                for model in models:
                    x=model(x)
                result=x.detach().cpu()
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,result,torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
            elif head=='Train':
                for model in models:
                    model.train()
                result,download_gradient=computation.cloud(models,upload_feature,label,optims,point,useQ)
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,result,download_gradient,server_rec,client_send,send_size))
                E3.set()
            elif head=="EndTrain":
                for sch in schedulers[point:]:
                    if sch=="FREE":
                        pass
                    else:
                        sch.step()
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,torch.tensor([-1.1]),torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
            else:
                utils.check_full(Q3,Efull)
                Q3.put((head,epoch,i,torch.tensor([-1.1]),torch.tensor([-1.1]),server_rec,client_send,send_size))
                E3.set()
        else:
            E2.wait()
    time.sleep(5)


