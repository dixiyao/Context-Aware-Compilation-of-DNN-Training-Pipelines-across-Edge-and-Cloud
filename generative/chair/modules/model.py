import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FirstLayer(nn.Module):
    def __init__(self):
        super(FirstLayer,self).__init__()

        # classes
        self.fc1_c = nn.Linear(809, 512)
        self.fc2_c = nn.Linear(512, 512)

        # views
        self.fc1_v = nn.Linear(4, 512)
        self.fc2_v = nn.Linear(512, 512)

        # transforms
        self.fc1_t = nn.Linear(12, 512)
        self.fc2_t = nn.Linear(512, 512)

    def forward(self, x):
        c,v,t=x
        # process each input separately
        c = F.relu(self.fc1_c(c))
        c = F.relu(self.fc2_c(c))

        v = F.relu(self.fc1_v(v))
        v = F.relu(self.fc2_v(v))

        t = F.relu(self.fc1_t(t))
        t = F.relu(self.fc2_t(t))
        
        # concatenate three tensors
        x = torch.cat((c, v, t), dim=1)

        return x

class UpConv1(nn.Module):
    def __init__(self):
        super(UpConv1,self).__init__()
                                   
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4,stride=2,padding=1)

    def forward(self,x):
        # resize the 1-d tensor into 2-d tensor
        x = x.view(-1, 256, 8, 8)

        # use CNN to process 
        x = F.relu(self.upconv1(x))
        return x
                                   
class UpConv4(nn.Module):
    def __init__(self):
        super(UpConv4,self).__init__()
                                   
        # upconv4 for generating the target color image
        self.upconv4_image = nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1)
        # upconv4 for generating the target segmentation mask
        self.upconv4_mask = nn.ConvTranspose2d(48, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # to get the two ouputs
        image = self.upconv4_image(x)
        # mask = self.upconv4_mask(x) # if use squared Euclidean distance
        # mask = self.softmax(self.upconv4_mask(x))  # if use NLL loss
        mask = self.sigmoid(image)
        return (image, mask)
                                   
class Net(nn.Module):
    # paper used a dataset of 809 cleaned up classes
    def __init__(self, classes = 809):
        super(Net, self).__init__()

        self.layers=[]
        self.layers.append(FirstLayer())                           

        self.layers.append(nn.Sequential(nn.Linear(1536, 1024),nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(1024, 1024),nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(1024, 16384),nn.ReLU()))
        
        # upsample layers
        self.layers.append(UpConv1())
        self.layers.append(nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.ReLU()))
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(256, 92, kernel_size=4, stride=2, padding=1),nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Conv2d(92, 92, kernel_size=3, padding=1),nn.ReLU()))
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(92, 48, kernel_size=4, stride=2, padding=1),nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1),nn.ReLU()))
        self.layers.append(UpConv4())
        
class SlicedBlock(nn.Module):
    def __init__(self, model, layers, splits_id, num_splits):
        super(SlicedBlock,self).__init__()

        self.splits_id = splits_id
        self.num_splits = num_splits
        self.layers = nn.Sequential(*layers)

        #initiate parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')    

    def forward(self, x,v=None,t=None):
        x=self.layers(x)
        return x

def getmodel(num_splits=11):
    backbone=Net()
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
        net = SlicedBlock(backbone, layers[left_idx:right_idx], splits_id, num_splits)
        nets.append(net)

    return nets       


        
