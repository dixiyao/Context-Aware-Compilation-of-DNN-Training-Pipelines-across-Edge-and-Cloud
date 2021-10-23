import torch.nn as nn
import torch.nn.functional as F
import torch
#from torchsummary import summary


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        
        self.classifier=nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
    
class Linear(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Linear, self).__init__()

        self.bn = nn.BatchNorm1d(4096)
        self.relu=nn.ReLU()
        self.classifier=nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class VGG(nn.Module):
    """
    VGG builder
    """

    def __init__(self, arch, num_classes=10,start_channel=3) -> object:
        super(VGG, self).__init__()
        self.in_channels = start_channel
        self.layers = []
        for channels,num in zip([64,128,256,512,512],arch):
            for i in range(num):
                layers=[]
                layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
                layers.append(nn.BatchNorm2d(channels))
                layers.append(nn.ReLU())
                self.in_channels = channels
                if i==num-1:
                    layers.append(nn.MaxPool2d(2))    
                module=nn.Sequential(*layers)
                self.layers.append(module)
        self.layers.append(Linear(1 * 1 * 512, 4096))
        self.layers.append(Linear(4096, 4096))
        self.layers.append(nn.Linear(4096, num_classes))

def VGG_11():
    return VGG([1, 1, 2, 2, 2], num_classes=10,start_channel=3)

def VGG_13():
    return VGG([1, 1, 2, 2, 2], num_classes=10,start_channel=3)

def VGG_16():
    return VGG([2, 2, 3, 3, 3], num_classes=10,start_channel=3)

def VGG_19():
    return VGG([2, 2, 4, 4, 4], num_classes=10,start_channel=3)

def test():
    import torch
    net = VGG_19().cuda()
    x=torch.rand((128,3,32,32)).cuda()
    x=net(x)

if __name__ == '__main__':
    test()
