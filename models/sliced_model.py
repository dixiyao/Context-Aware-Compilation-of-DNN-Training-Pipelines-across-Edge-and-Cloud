import torch
import torch.nn as nn
from torch.nn import init
import math
import time
#from torchstat import stat


#only for linear models
def getLayers(model):
    layers = []
 
    def unfoldLayer(model):
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            if sublayer_num == 0:
                layers.append(module)
            elif isinstance(module, nn.Module):
                unfoldLayer(module)
            
    unfoldLayer(model)
    return layers

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

    #initiate parameters
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            #m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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

def pretrain(nets,backbone):
    split_id=0
    backbone_iter = backbone.parameters()
    
    for split_id in range(0,len(nets)):
        nets_iter = nets[split_id].parameters()
        try:
            while True:
                nets_weight = next(nets_iter)
                backbone_weight = next(backbone_iter)
                if not nets_weight.shape== backbone_weight.shape:
                    print(backbone_weight.shape)
                    continue
                nets_weight.data=backbone_weight.data
        except StopIteration:
            pass
    
if __name__=="__main__":
    import ResNets
    import DenseNets
    import CHAIR
    import MobileNet
    import VGG
    import torchvision.models as backbones
    import utils
    import matplotlib.pyplot as plt
    
    model=ResNets.resnet50(num_classes=10,start_channel=3)，19
    models=get_sliced_model(model,19)
    inputs=torch.rand(128,3,32,32)

    MAdd=[]
    memory=[]
    feature=[]

    
    x=inputs
    for i in range(19):
        x=models[i](x)
        print(x.shape)
        feature.append(utils.count_tensorsize_in_B(x)/1024/1024)

    print("[",end="")
    for item in MAdd:
        print(item,end=',')
    print("]")
    print("[",end="")
    for item in memory:
        print(item,end=',')
    print("]")
    print("[",end="")
    for item in feature:
        print(item,end=',')
    print("]")

