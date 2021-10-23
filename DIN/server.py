import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import pickle
import lz4.frame
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

from dataGenerator import dataGenerator
from model import DIN,Bottom
from config import Config
import tcper

import sys
sys.path.append('../')
import Utils.loggers as logger
import Utils.utils as utils

parser = argparse.ArgumentParser(description='Base method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
#transfer
parser.add_argument('--ip', type=str, default='0.0.0.0', help='ip of server address')
parser.add_argument('--portA', type=int, default=1883, help='TCP port of server')
parser.add_argument('--portB', type=int, default=1886, help='TCP port of server')
parser.add_argument('--hyperport', type=int, default=1884, help='TCP port of server for model update and sppedtest')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

def upload_cloud(server,Q2,E2):
    while True:
            head,epoch,iters,target,m,a,useQ,server_rec,client_send,send_size=server.recieve_tensor()
            print(head,iters,m.shape,useQ)
            if useQ==True:
                #m=pickle.loads(m)
                m=torch.dequantize(m)
                #a=pickle.loads(a)
                a=torch.dequantize(a)
            Q2.put((head,epoch,iters,target,m,a,useQ,server_rec,client_send,send_size))
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
    seed=1
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # parameters
    config = Config()
    # cuda environments
    dev = torch.device(config.cuda) if torch.cuda.is_available() else torch.device('cpu')

    topmodel=DIN(embeddingGroupInfo=config.embeddingGroups,
                MLPInfo=config.MLPInfo,
                attMLPInfo=config.AttMLPInfo,
                isUseBN=config.isUseBN,
                l2RegEmbedding=config.l2RegEmbedding,
                dropoutRate=config.dropoutRate,
                initStd=config.initStd,
                device=dev)
        
    optimizer=config.optimizer(topmodel.parameters(), lr=config.learningRate)
    scheduler=config.lrSchedule(optimizer, config.decay)
    print(optimizer)

    lossFunc = config.lossFunc
    metricFunc = config.metricFunc

    serverA=tcper.Server(args.ip,args.portA)
    server_start=time.time()
    print(server_start)

    serverA.send_tensor('begin',-1,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),server_start,1,1)

    serverB=tcper.Server(args.ip,args.portB)
    server_start=time.time()
    print(server_start)

    serverB.send_tensor('begin',-1,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),server_start,1,1)
    #shared memory
    Q2A=mp.Queue()
    Q3A=mp.Queue()
    E2A=mp.Event()
    E3A=mp.Event()
    EfullA=mp.Event()
    EfullA.set()
    
    puploadA=mp.Process(target=upload_cloud,args=(serverA,Q2A,E2A))
    pdownloadA=mp.Process(target=download_cloud,args=(serverA,Q3A,E3A,EfullA))
    pdownloadA.start()
    puploadA.start()

    Q2B=mp.Queue()
    Q3B=mp.Queue()
    E2B=mp.Event()
    E3B=mp.Event()
    EfullB=mp.Event()
    EfullB.set()
    
    puploadB=mp.Process(target=upload_cloud,args=(serverB,Q2B,E2B))
    pdownloadB=mp.Process(target=download_cloud,args=(serverB,Q3B,E3B,EfullB))
    pdownloadB.start()
    puploadB.start()
    last_epoch=-1

    impressionNum = 0.0
    impressAuc = 0.0

    Log_all=logger.Logger('VFL_DP')
    all_log=[]
    epoch_all_log=[]
    acc_log=[]
    all_log.append(0)
    epoch_all_log.append(-1)
    all_log.append(time.time())
    acc_log.append(0)

    while True:
        if (not Q2A.empty()) and (not Q2B.empty()):
            #print("cloud")
            head1,epoch1,i1,label,m1,a1,useQ1,server_rec1,client_send1,send_size1=Q2A.get()
            head2,epoch2,i2,label,m2,a2,useQ2,server_rec2,client_send2,send_size2=Q2B.get()
            head=head1
            if head=='Termi':
                utils.check_full(Q3A,EfullA)
                Q3A.put((head,epoch1,i1,torch.tensor([-1.1]),torch.tensor([-1.1]),server_rec1,client_send1,send_size1))
                E3A.set()
                utils.check_full(Q3B,EfullB)
                Q3B.put((head,epoch2,i2,torch.tensor([-1.1]),torch.tensor([-1.1]),server_rec2,client_send2,send_size2))
                E3B.set()
                break
            elif head=='Valid':
                if (not epoch1==epoch_all_log[-1]) and i1==0:
                    all_log[-1]=time.time()-all_log[-1]
                topmodel.eval()
                m1=m1.to(dev)
                m2=m2.to(dev)
                a1=a1.to(dev)
                a2=a2.to(dev)
                label=label.numpy()
                preds = topmodel.predict(m1,m2,a1,a2)
                auc = metricFunc(label, preds)
                impressionNum += 1
                impressAuc += auc
            elif head=='EndValid':
                if not epoch1==epoch_all_log[-1]:
                    epoch_all_log.append(epoch1)
                    if not epoch1==4:
                        all_log.append(time.time())
                acc_log.append(impressAuc/impressionNum)
                print("test:{}".format(impressAuc/impressionNum))
                impressionNum = 0.0
                impressAuc = 0.0
            elif head=='Train':
                topmodel.train()
                label=label.to(dev)
                m1=Variable(m1, requires_grad=True).to(dev)
                m2=Variable(m2, requires_grad=True).to(dev)
                a1=Variable(a1, requires_grad=True).to(dev)
                a2=Variable(a2, requires_grad=True).to(dev)
                m1.retain_grad()
                m2.retain_grad()
                a1.retain_grad()
                a2.retain_grad()
                loss=topmodel.loss(m1,m2,a1,a2,label,lossFunc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                gm1=m1.grad.detach().cpu()
                gm2=m2.grad.detach().cpu()
                ga1=a1.grad.detach().cpu()
                ga2=a2.grad.detach().cpu()
                utils.check_full(Q3A,EfullA)
                Q3A.put((head,epoch1,i1,gm1,ga1,server_rec1,client_send1,send_size1))
                E3A.set()
                utils.check_full(Q3B,EfullB)
                Q3B.put((head,epoch2,i2,gm2,ga2,server_rec2,client_send2,send_size2))
                E3B.set()
            else:
                utils.check_full(Q3A,EfullA)
                Q3A.put((head,epoch1,i1,torch.tensor([-1.1]),torch.tensor([-1.1]),server_rec1,client_send1,send_size1))
                E3A.set()
                utils.check_full(Q3B,EfullB)
                Q3B.put((head,epoch2,i2,torch.tensor([-1.1]),torch.tensor([-1.1]),server_rec2,client_send2,send_size2))
                E3B.set()
        else:
            if Q2A.empty():
                E2A.wait()
            if Q2B.empty():
                E2B.wait()
            
    time.sleep(5)
    items=['epoch','latency','acc']
    contents=[epoch_all_log,all_log,acc_log]
    Log_all.write(items,contents)

