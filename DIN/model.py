import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, MLPInfo, activation='PReLU', PReLuInit=0.25, isUseBN=True, dropoutRate=0.0, initStd=0.0001):
        super(MLP, self).__init__()

        self.multiLayerPerceptron = nn.ModuleList()  # MLP
        for i in range(len(MLPInfo)-1):
            self.multiLayerPerceptron.append(nn.Linear(MLPInfo[i], MLPInfo[i + 1]))

            if isUseBN:
                self.multiLayerPerceptron.append(nn.BatchNorm1d(MLPInfo[i + 1]))

            actiFun = nn.PReLU(1, init=PReLuInit) if activation == 'PReLU' else Dice()
            self.multiLayerPerceptron.append(actiFun)

            self.multiLayerPerceptron.append(nn.Dropout(dropoutRate))

    def forward(self, x):
        for layer in self.multiLayerPerceptron:
            x = layer(x)
        return x

class Bottom(nn.Module):
    def __init__(self, embeddingGroupInfo, MLPInfo, attMLPInfo, activation='PReLU',
                 PReLuInit=0.25, isUseBN=True, l2RegEmbedding=1e-6,
                 dropoutRate=0.0, initStd=0.0001, device=torch.device('cpu')):
        super(Bottom, self).__init__()
        self.dev = device
        self.embeddingGroups = nn.ModuleDict()  # embedding group
        for key, value in embeddingGroupInfo.items():
            if key == 'MovieId' or key == 'Genre':
                self.embeddingGroups[key] = nn.Embedding(value[0], value[1], padding_idx=0)
            else:
                self.embeddingGroups[key] = nn.Embedding(value[0], value[1])

        self.sequenceMeanPooling = SequencePoolingLayer(mod='mean', device=self.dev)  # sequence pooling layer
        self.attentionActivationUnit = AttentionActivationUnit(attMLPInfo, activation,
                                                               PReLuInit, initStd)  # attention activation unit
        self.sequenceAttentionPooling = SequencePoolingLayer(mod='attention', device=self.dev)  # sequence pooling layer
        self.to(self.dev)

    def forward(self, movieIdSequence,ads, movieFeature):
        movieFeatSequence = movieFeature[movieIdSequence]
        adsFeat = movieFeature[ads]
               
        movieIdFeat = self.embeddingGroups['MovieId'](movieFeatSequence[:, :, 0])  # (B, SeqLen, 16)
        movieGenreFeat = self.embeddingGroups['Genre'](movieFeatSequence[:, :, 1:])  # (B, SeqLen, 6, 8)
        movieGenreFeat = self.sequenceMeanPooling(movieGenreFeat, movieFeatSequence[:, :, 1:] > 0)  # (B, SeqLen, 8)
        #print(movieGenreFeat)
        #input()
        adsIdFeat = self.embeddingGroups['MovieId'](adsFeat[:, 0])  # (B, 16)
        adsGenreFeat = self.embeddingGroups['Genre'](adsFeat[:, 1:])  # (B, 6, 8)
        adsGenreFeat = self.sequenceMeanPooling(adsGenreFeat, adsFeat[:, 1:] > 0)  # (B, 8)
        adsEmbedding = torch.cat((adsIdFeat, adsGenreFeat), dim=-1)  # (B, 24)
        
        movieEmbedding = torch.cat((movieIdFeat, movieGenreFeat), dim=-1)  # (B, SeqLen, 24)
        attentionWeights = self.attentionActivationUnit(movieEmbedding, adsEmbedding)  # (B, SeqLen, 1)
        movieSequenceEmbedding = self.sequenceAttentionPooling(movieEmbedding, attentionWeights)  # (B, 24)

        return movieSequenceEmbedding,adsEmbedding

    def forward_FR(self, movieIdSequence,ads, movieFeature):
        movieSequenceEmbedding,adsEmbedding=self.forward(movieIdSequence,ads, movieFeature)
        out=torch.cat((movieSequenceEmbedding,adsEmbedding),dim=0)
        return out
class DIN(nn.Module):
    def __init__(self, embeddingGroupInfo, MLPInfo, attMLPInfo, activation='PReLU',
                 PReLuInit=0.25, isUseBN=True, l2RegEmbedding=1e-6,
                 dropoutRate=0.0, initStd=0.0001, device=torch.device('cpu')):
        super(DIN, self).__init__()
        self.l2RegEmbeddding = l2RegEmbedding
        self.dev = device

        self.MLP = MLP(MLPInfo, activation, PReLuInit, isUseBN, dropoutRate)  # MLP
        self.output = nn.Linear(MLPInfo[-1], 2)  # output layer
        self.to(self.dev)

    def forward(self, m1,m2,a1,a2):
        #interactive
        movieSequenceEmbedding=m1+m2
        adsEmbedding=a1+a2
        # MLP inputs
        x = torch.cat((movieSequenceEmbedding, adsEmbedding), dim=-1)
        x = self.MLP(x)
        x = F.softmax(self.output(x), dim=1)
        return x  # (B, 2)

    def regLoss(self):
        totalRegLoss = torch.zeros(size=(1,), device=self.dev)
        for name, param in self.named_parameters():
            if 'embedding' in name and 'MovieId' in name and 'weight' in name:
                totalRegLoss += torch.sum(self.l2RegEmbeddding * param*param)
        return totalRegLoss

    def loss(self, m1,m2,a1,a2,label, lossFunc):
        preds = self.forward(m1,m2,a1,a2)
        loss = lossFunc(preds[:, 1], label.float(), reduction='mean') + self.regLoss()
        return loss

    def predict(self, m1,m2,a1,a2):
        preds = self.forward(m1,m2,a1,a2)[:, 1]
        return preds.cpu().detach().numpy()


class SequencePoolingLayer(nn.Module):
    def __init__(self, mod='mean', device=torch.device('cpu')):
        super(SequencePoolingLayer, self).__init__()
        self.mod = mod
        self.dev = device
        self.eps = torch.FloatTensor([1e-8]).to(self.dev)

    def forward(self, x, mask):
        if self.mod == 'mean':
            length = torch.sum(mask.type(torch.float32), dim=-1, keepdim=True)  # (..., dim, 6) -> (...,dim, 1)
            x = torch.sum(x, dim=-2, keepdim=False)  # (..., dim, 6, 8) -> (..., dim, 8)
            x = torch.div(x, length.type(torch.float32) + self.eps)  # (..., dim, 8)
        elif self.mod == 'attention':
            attentionWeights = torch.repeat_interleave(mask, x.shape[-1], dim=-1)  # (..., dim, 1) -> (.... dim, E)
            x = torch.mul(x, attentionWeights)  # (..., dim, E)
            x = torch.sum(x, dim=-2, keepdim=False)  # (..., dim, E) -> (..., E)
        else:
            pass

        return x


class AttentionActivationUnit(nn.Module):
    def __init__(self, attMLPInfo, activation='PReLu', PReLuInit=0.25, initStd=0.0001):
        super(AttentionActivationUnit, self).__init__()

        self.MLP = MLP(attMLPInfo, activation, PReLuInit, isUseBN=False, dropoutRate=0.0, initStd=initStd)
        self.output = nn.Linear(attMLPInfo[-1], 1)

    def forward(self, x, target):
        target = torch.unsqueeze(target, dim=1)  # (B, 1, 24)
        target = torch.repeat_interleave(target, x.shape[-2], dim=1)  # (B, SeqLen, 24)
        product = torch.mul(x, target)  # (B, SeqLen, 24)

        # product = torch.sum(product, dim=-1, keepdim=True)  # (B, SeqLen, 1)

        x = torch.cat((x, target, product), dim=2)  # (B, SeqLen, 72)
        x = self.MLP(x)
        x = self.output(x)
        # product = torch.sum(product, dim=-1, keepdim=True)
        # product = F.softmax(product, dim=1)

        return x  # (B, SeqLen, 1)


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        pass
