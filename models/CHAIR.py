import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstBlock(nn.Module):
    def __init__(self, classes = 809):
        super(FirstBlock, self).__init__()

        self.classes=classes
        # classes
        self.fc1_c = nn.Linear(classes, 512)
        self.fc2_c = nn.Linear(512, 512)

        # views
        self.fc1_v = nn.Linear(4, 512)
        self.fc2_v = nn.Linear(512, 512)

        # transforms
        self.fc1_t = nn.Linear(12, 512)
        self.fc2_t = nn.Linear(512, 512)

    def forward(self, x):
        c,v,t=torch.split(x,[self.classes,4,12],dim=1)
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

class FC5(nn.Module):
    def __init__(self):
        super(FC5, self).__init__()
        self.fc5 = nn.Sequential(nn.Linear(1024, 16384),nn.ReLU(inplace=True))

    def forward(self,x):
        x=self.fc5(x)
        x = x.view(-1, 256, 8, 8)
        return x

class Chair(nn.Module):
    # paper used a dataset of 809 cleaned up classes
    def __init__(self, classes = 809):
        super(Chair, self).__init__()
        
        self.layers = []

        self.layers.append(FirstBlock(classes))
        self.layers.append(nn.Sequential(nn.Linear(1536, 1024),nn.ReLU(inplace=True)))
        self.layers.append(nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(inplace=True)))
        self.layers.append(FC5())
        
        # upsample layers
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4,stride=2,padding=1),nn.ReLU(inplace=True)))
        self.layers.append(nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.ReLU(inplace=True)))
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(256, 92, kernel_size=4, stride=2, padding=1),nn.ReLU(inplace=True)))
        self.layers.append(nn.Sequential(nn.Conv2d(92, 92, kernel_size=3, padding=1),nn.ReLU(inplace=True)))
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(92, 48, kernel_size=4, stride=2, padding=1),nn.ReLU(inplace=True)))
        self.layers.append(nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1),nn.ReLU(inplace=True)))
        # upconv4 for generating the target color image
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1),nn.ReLU(inplace=True)))
        
