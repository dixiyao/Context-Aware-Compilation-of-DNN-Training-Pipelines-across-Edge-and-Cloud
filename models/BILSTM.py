import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import *
import math

class lstm(nn.Module):
    def __init__(self,Blstm,dp):
        super(lstm, self).__init__()
        self.LSTM=Blstm

    def forward(self,x):
        x,_=self.LSTM(x)
        return x

class avgpool(nn.Module):
    def __init__(self):
        super(avgpool, self).__init__()

    def forward(self,x):
        x=F.avg_pool2d(x, (x.shape[1],1)).squeeze()
        return x

class LSTM(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(LSTM,self).__init__()
        self.layers=[]
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding=nn.Embedding(self.max_words, self.emb_size)
        self.LSTM=nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                          batch_first=True, bidirectional=True) #2层双向LSTM
        self.dp=nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)
        self.layers.append(self.Embedding)
        self.layers.append(lstm(self.LSTM,self.dp))
        self.layers.append(nn.Sequential(self.fc1,nn.ReLU()))
        self.layers.append(avgpool())
        self.layers.append(self.fc2)

class SlicedBlock(nn.Module):
  def __init__(self, model, layers, splits_id, num_splits,b_feature=False):
    super(SlicedBlock,self).__init__()

    self.splits_id = splits_id
    self.num_splits = num_splits

    #for binarize
    if not splits_id==num_splits-1:
        if b_feature:
            layers.append(nn.Tanh())
    self.layers = nn.Sequential(*layers)

  def forward(self, x):    
    x = self.layers(x)
      
    return x

def get_sliced_model(backbone,num_splits=2,b_feature=False):

    #layers=getLayers(backbone)
    layers=backbone.layers
    len_layers = len(layers)
    print(len_layers)
    if num_splits>len_layers:
        raise Exception("too many splits")
    split_depth = math.ceil(len_layers / num_splits)
    nets = []
    for splits_id in range(num_splits):
        left_idx = splits_id * split_depth
        right_idx = (splits_id+1) * split_depth
        if right_idx > len_layers:
            right_idx = len_layers
        net = SlicedBlock(backbone, layers[left_idx:right_idx], splits_id, num_splits,b_feature)
        nets.append(net)

    return nets
                           
def getmodel(max_words, emb_size, hid_size, dropout,device):
    model = LSTM(max_words, emb_size, hid_size, dropout).to(device)
    models=get_sliced_model(model,5)
    return models

