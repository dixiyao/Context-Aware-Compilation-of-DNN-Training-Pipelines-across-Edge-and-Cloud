import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

from dataGenerator import dataGenerator
from model import DIN,Bottom
from config import Config
import tcper

import sys
sys.path.append('../')
import Utils.loggers as logger
import Utils.utils as utils


parser = argparse.ArgumentParser(description='Base method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#stale limitation
parser.add_argument('--stale_it', type=int, default=5, help='Limitation of stale epoch K*')
#Logger
parser.add_argument('--log_name', type=str, default='dynamic_wiredmobilelarge', help='name of log')
#transfer
parser.add_argument('--ip', type=str, default='127.0.0.1', help='ip of server address')
parser.add_argument('--port', type=int, default=1883, help='TCP port of server')
parser.add_argument('--hyperport', type=int, default=1884, help='TCP port of server for model update')
# random seed
parser.add_argument('--seed', type=int, default=10,help='manual seed')
args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
#report
parser.add_argument('--report_freq', type=int, default=100, help='Reporting frequency')

def quantize(t):
    dev=t.device
    feature=t.detach().cpu()
    #Q=feature
    #E=0
    min_range = max(torch.min(feature).item(),-128)
    max_range = min(torch.max(feature).item(),128)
    Q = torch.quantize_per_tensor(feature, scale=(max_range-min_range)/(2**8), zero_point=int(min_range), dtype=torch.qint8)
    E=torch.dequantize(Q)-feature
    Q=pickle.dumps(Q)
    return Q,E

def download_edge(client,Q4,E4,fix,up,down):
    while True:
            head,epoch,iters,result,gradient,client_send,server_rec,client_send_size,server_send,rec_data_length=client.recieve_tensor()
            if head=='Train':
                rec_time=time.time()
                #up.value=client_send_size/(server_rec-(client_send+fix))/1024/1024
                #down.value=rec_data_length/(rec_time+fix-server_send)/1024/1024
                #print(iters,client_send_size,client_send,server_rec,rec_data_length, server_send,rec_time)
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
            a,b,c,d,e,f,g=Q1.get()
            Efull.set()
            client.send_tensor(a,b,c,d,e,f,g)
            if a=='Termi':
                break
        else:
            E1.wait()
    time.sleep(5)
    
if __name__=="__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

    # parameters
    config = Config()
    # cuda environments
    dev = torch.device(config.cuda) if torch.cuda.is_available() else torch.device('cpu')

    bottom=Bottom(embeddingGroupInfo=config.embeddingGroups,
                MLPInfo=config.MLPInfo,
                attMLPInfo=config.AttMLPInfo,
                isUseBN=config.isUseBN,
                l2RegEmbedding=config.l2RegEmbedding,
                dropoutRate=config.dropoutRate,
                initStd=config.initStd,
                device=dev)
    #bottom= torch.load('./bottom.pth')
    optimizer=config.optimizer(bottom.parameters(), lr=config.learningRate)
    scheduler=config.lrSchedule(optimizer, config.decay)
    print(optimizer)
    
    dataset = dataGenerator(dataPath=config.dataPath, ratingBinThreshold=config.ratingBinThreshold,
                            maxSequenceLen=config.maxSequenceLen,
                            splitRatio=config.splitRatio, splitMehtod=config.splitMethod)
    # train dataset
    trainRowData, trainLabel = map(torch.tensor, (dataset.trainRowData, dataset.trainLabel))
    trainDataLoader = DataLoader(dataset=TensorDataset(trainRowData, trainLabel),
                                 batch_size=config.batchSize, shuffle=False)
    # test dataset
    testRowData, testLabel = map(torch.tensor, (dataset.testRowData, dataset.testLabel))
    testDataLoader = DataLoader(dataset=TensorDataset(testRowData, testLabel),
                                batch_size=1000, shuffle=False)
    # user features and movie features
    userFeatures = torch.tensor(dataset.userFeatures).to(dev)
    movieFeatures = torch.tensor(dataset.movieFeatures).to(dev)

    client=tcper.Client(args.ip,args.port)
    client_start=time.time()
    print(client_start)
    time.sleep(3)

    a,b,c,d,e,f,g,h,i,j=client.recieve_tensor()
    server_start=g
    print(server_start)
    client_slower_server=server_start-client_start
    #shared memory
    Q1=mp.Queue()
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

    Log_all=logger.Logger(args.log_name)
    stale_log=[]
    
    meter=utils.AvgrageMeter()
    stale_it=args.stale_it
    current_back=-1

    for epoch in range(5):
        bottom.train()
        for i, (data, target) in enumerate(trainDataLoader):
                rowData=data.to(dev)
                label=target.to(dev)
                movieIdSequence = rowData[:, 1: 5]
                ads = rowData[:, -1]
                movieSequenceEmbedding,adsEmbedding=bottom(movieIdSequence,ads,movieFeatures)
                Qm,Em=quantize(movieSequenceEmbedding)
                Qa,Ea=quantize(adsEmbedding)
                Q_history.put((epoch,i,data,target,Em,Ea))
                utils.check_full(Q1,Efull)
                Q1.put(('Train',epoch,i,target,Qm,Qa,True))
                E1.set()
                #backward
                while True:
                    while not Q4.empty():
                        head,b_epoch,b_iter,gm,ga=Q4.get()
                        if not (head=='Train' or head=="Edge"):
                            print(head)
                        if head=='Train':
                            stale_log.append(i-b_iter)
                        _,_,data,_,Em,Ea=Q_history.get()
                        rowData=data.to(dev)
                        movieIdSequence = rowData[:, 1: 5]
                        ads = rowData[:, -1]
                        out=bottom.forward_FR(movieIdSequence,ads,movieFeatures)
                        gm=gm.to(dev)
                        ga=ga.to(dev)
                        Em=Em.to(dev)
                        Ea=Ea.to(dev)
                        gm=gm-Em*gm*gm
                        ga=ga-Ea*ga*ga
                        optimizer.zero_grad()
                        out.backward(torch.cat((gm,ga),dim=0))
                        optimizer.step()
                        current_back=b_iter
                    if i-current_back<=stale_it:
                        break
        scheduler.step()
        utils.check_full(Q1,Efull)       
        Q1.put(('EndTrain',epoch,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),torch.tensor([-1.1]),False))
        E1.set()
        while True:
            if not Q4.empty():
                    head,b_epoch,b_iter,gm,ga=Q4.get()
                    if not (head=='Train'or head=='EndTrain'):
                        print(head)
                    if head=='Train':
                            stale_log.append(i-b_iter)
                    if head=='EndTrain':
                        break
                    _,_,data,_,Em,Ea=Q_history.get()
                    rowData=data.to(dev)
                    movieIdSequence = rowData[:, 1: 5]
                    ads = rowData[:, -1]
                    out=bottom.forward_FR(movieIdSequence,ads,movieFeatures)
                    gm=gm.to(dev)
                    ga=ga.to(dev)
                    Em=Em.to(dev)
                    Ea=Ea.to(dev)
                    gm=gm-Em*gm*gm
                    ga=ga-Ea*ga*ga
                    optimizer.zero_grad()
                    out.backward(torch.cat((gm,ga),dim=0))
                    optimizer.step()
            else:
                E4.wait()
        #test
        bottom.eval()
        for i, (data, target) in enumerate(testDataLoader):
            rowData=data.to(dev)
            label=target.to(dev)
            movieIdSequence = rowData[:, 1: 5]
            ads = rowData[:, -1]
            movieSequenceEmbedding,adsEmbedding=bottom(movieIdSequence,ads,movieFeatures)
            Qm=movieSequenceEmbedding.detach().cpu()
            Qa=adsEmbedding.detach().cpu()
            utils.check_full(Q1,Efull)
            Q1.put(('Valid',epoch,i,target,Qm,Qa,False))
            E1.set()
        utils.check_full(Q1,Efull)
        Q1.put(('EndValid',epoch,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),torch.tensor([-1.1]),False))
        E1.set()
        utils.check_full(Q1,Efull)
        Q1.put(('END',-1,-1,torch.tensor([-1.1]),torch.tensor([-1.1]),torch.tensor([-1.1]),False))
        #E1.set()
        while True:
            if not Q4.empty():
                    head,b_epoch,b_iter,result,download_gradient=Q4.get()
                    if head=='END':
                        break
            else:
                E4.wait()
    #terminate
    utils.check_full(Q1,Efull)
    Q1.put(('Termi',-1,0,torch.tensor([-1.1]),torch.tensor([-1.1]),torch.tensor([-1.1]),False))
    E1.set()
    time.sleep(5)

    items=['stale']
    contents=[stale_log]
    Log_all.write(items,contents)

