import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import time
import random
import socket
import copy
from queue import Queue

import numpy as np

import torchvision.models as backbones
from torchvision import datasets, transforms

import sys
sys.path.append('../')
import Utils.loggers as logger

from dataGenerator import dataGenerator
from model import DIN,Bottom
from config import Config
'''
def quantize_simu(t):
    dev=t.device
    feature=t.detach().cpu()
    min_range = max(torch.min(feature).item(),-128)
    max_range = min(torch.max(feature).item(),128)
    Q = torch.quantize_per_tensor(feature, scale=(max_range-min_range)/(2**8), zero_point=int(min_range), dtype=torch.qint8)
    E=torch.dequantize(Q)-feature
    Q=torch.dequantize(Q)
    return Q.to(dev),E.to(dev)
'''
def quantize_simu(t):
    dev=t.device
    feature=t.detach()
    Q=feature
    E=0
    #min_range = max(torch.min(feature).item(),-128)
    #max_range = min(torch.max(feature).item(),128)
    #Q = torch.quantize_per_tensor(feature, scale=(max_range-min_range)/(2**8), zero_point=int(min_range), dtype=torch.qint8)
    #E=torch.dequantize(Q)-feature
    #Q=pickle.dumps(Q)
    return Q,E

if __name__=="__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic=True
    
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
    bottomA=Bottom(embeddingGroupInfo=config.embeddingGroups,
                MLPInfo=config.MLPInfo,
                attMLPInfo=config.AttMLPInfo,
                isUseBN=config.isUseBN,
                l2RegEmbedding=config.l2RegEmbedding,
                dropoutRate=config.dropoutRate,
                initStd=config.initStd,
                device=dev)
    bottomA=torch.load('./bottom.pth')
    bottomB=Bottom(embeddingGroupInfo=config.embeddingGroups,
                MLPInfo=config.MLPInfo,
                attMLPInfo=config.AttMLPInfo,
                isUseBN=config.isUseBN,
                l2RegEmbedding=config.l2RegEmbedding,
                dropoutRate=config.dropoutRate,
                initStd=config.initStd,
                device=dev)
    
    optimizers=[]
    schedulers=[]
    for model in [topmodel,bottomA,bottomB]:
            optim=config.optimizer(model.parameters(), lr=config.learningRate)
            sch=config.lrSchedule(optim, config.decay)
            optimizers.append(optim)
            schedulers.append(sch)
    print(optimizers)

    Start=time.time()
    # ============== load data ============== #
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
                                batch_size=1000, shuffle=True)

    # user features and movie features
    userFeatures = torch.tensor(dataset.userFeatures).to(dev)
    movieFeatures = torch.tensor(dataset.movieFeatures).to(dev)

    Log_all=logger.Logger('VFL_EF')
    all_log=[]
    epoch_all_log=[]
    acc_log=[]

    stale_it=0
    lossFunc = config.lossFunc
    metricFunc = config.metricFunc

    for epoch in range(5):
        epoch_all_log.append(epoch)
        Q_history=Queue()
        Q_error=Queue()
        for i in range(stale_it):
            Q_error.put("no")
        bottomA.eval()
        bottomB.eval()
        topmodel.train()
        torch.cuda.synchronize()
        start = time.time()
        for i, (data, target) in enumerate(trainDataLoader):
                #edge forward
                Q_history.put((data,target))
                rowData=data.to(dev)
                label=target.to(dev)
                movieIdSequence1 = rowData[:, 1: 5]
                movieIdSequence2 = rowData[:, 5: -1]
                ads = rowData[:, -1]
                movieSequenceEmbedding1,adsEmbedding1=bottomA(movieIdSequence1,ads,movieFeatures)
                movieSequenceEmbedding2,adsEmbedding2=bottomB(movieIdSequence2,ads,movieFeatures)
                m1,Em1=quantize_simu(movieSequenceEmbedding1)
                m2,Em2=quantize_simu(movieSequenceEmbedding2)
                a1,Ea1=quantize_simu(adsEmbedding1)
                a2,Ea2=quantize_simu(adsEmbedding2)
                m1=Variable(m1, requires_grad=True)
                m2=Variable(m2, requires_grad=True)
                a1=Variable(a1, requires_grad=True)
                a2=Variable(a2, requires_grad=True)
                m1.retain_grad()
                m2.retain_grad()
                a1.retain_grad()
                a2.retain_grad()
                loss=topmodel.loss(m1,m2,a1,a2,label,lossFunc)
                optimizers[0].zero_grad()
                loss.backward()
                optimizers[0].step()
                gm1=m1.grad.detach()
                gm2=m2.grad.detach()
                ga1=a1.grad.detach()
                ga2=a2.grad.detach()
                #gm1=gm1-Em1*gm1*gm1
                #gm2=gm2-Em2*gm2*gm2
                #ga1=ga1-Ea1*ga1*ga1
                #ga2=ga2-Ea2*ga2*ga2
                gradients=(gm1,gm2,ga1,ga2)
                Q_error.put((i,gradients))
                gradients_=Q_error.get()
                if gradients_=="no":
                    continue
                it=gradients_[0]
                gradients=gradients_[1]
                if not (i-it)==stale_it:
                    print("cao")
                data,label=Q_history.get()
                bottomA.train()
                bottomB.train()
                rowData=data.to(dev)
                movieIdSequence1 = rowData[:, 1: 5]
                movieIdSequence2 = rowData[:, 5: -1]
                ads = rowData[:, -1]
                print('outout----------------------\n',gradients[0],gradients[2])
                input()
                out1=bottomA.forward_FR(movieIdSequence1,ads,movieFeatures)
                out2=bottomB.forward_FR(movieIdSequence2,ads,movieFeatures)
                optimizers[1].zero_grad()
                optimizers[2].zero_grad()
                out1.backward(torch.cat((gradients[0],gradients[2]),dim=0))
                out2.backward(torch.cat((gradients[1],gradients[3]),dim=0))
                optimizers[1].step()
                optimizers[2].step()
        while not Q_error.empty():
            gradients_=Q_error.get()
            if gradients_=="no":
                continue
            it=gradients_[0]
            gradients=gradients_[1]
            data,label=Q_history.get()
            bottomA.train()
            bottomB.train()
            rowData=data.to(dev)
            movieIdSequence1 = rowData[:, 1: 5]
            movieIdSequence2 = rowData[:, 5: -1]
            ads = rowData[:, -1]
            out1=bottomA.forward_FR(movieIdSequence1,ads,movieFeatures)
            out2=bottomB.forward_FR(movieIdSequence2,ads,movieFeatures)
            optimizers[1].zero_grad()
            optimizers[2].zero_grad()
            out1.backward(torch.cat((gradients[0],gradients[2]),dim=0))
            out2.backward(torch.cat((gradients[1],gradients[3]),dim=0))
            optimizers[1].step()
            optimizers[2].step()
        
        torch.cuda.synchronize()
        end = time.time()
        all_log.append(end-start)
        for sch in schedulers:
            sch.step()
        if epoch%1==0:
            bottomA.eval()
            bottomB.eval()
            topmodel.eval()
            impressionNum = 0.0
            impressAuc = 0.0
            for i, (data, target) in enumerate(testDataLoader):
                rowData=data.to(dev)
                label=target.numpy()
                movieIdSequence1 = rowData[:, 1: 5]
                movieIdSequence2 = rowData[:, 5: -1]
                ads = rowData[:, -1]
                movieSequenceEmbedding1,adsEmbedding1=bottomA(movieIdSequence1,ads,movieFeatures)
                movieSequenceEmbedding2,adsEmbedding2=bottomB(movieIdSequence2,ads,movieFeatures)
                preds = topmodel.predict(movieSequenceEmbedding1,movieSequenceEmbedding2,adsEmbedding1,adsEmbedding2)
                auc = metricFunc(label, preds)
                impressionNum += 1
                impressAuc += auc
            print("test:{}".format(impressAuc/impressionNum))
            acc_log.append(impressAuc/impressionNum)
            
        
    End=time.time()
    print(End-Start)
    items=['epoch','latency','acc']
    contents=[epoch_all_log,all_log,acc_log]
    Log_all.write(items,contents)
