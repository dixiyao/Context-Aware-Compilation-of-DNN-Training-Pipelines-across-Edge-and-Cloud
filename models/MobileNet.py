import re
import types

import torch.nn as nn
import torch.nn.init

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.sigmoid(x, inplace=True)

class HSwish(torch.nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0

class HSigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0

def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "relu6":
        return torch.nn.ReLU6(inplace=True)
    elif activation == "swish":
        return Swish()
    elif activation == "hswish":
        return HSwish()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid(inplace=True)
    elif activation == "hsigmoid":
        return HSigmoid()
    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))
    
class SEUnit(torch.nn.Module):
    def __init__(self,
                 channels,
                 squeeze_factor=16,
                 squeeze_activation="relu",
                 excite_activation="sigmoid"):
        super().__init__()
        squeeze_channels = channels // squeeze_factor

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels, out_channels=squeeze_channels, bias=True)
        self.activation1 = get_activation(squeeze_activation)
        self.conv2 = conv1x1(in_channels=squeeze_channels, out_channels=channels, bias=True)
        self.activation2 = get_activation(excite_activation)

    def forward(self, x):
        s = self.pool(x)
        s = self.conv1(s)
        s = self.activation1(s)
        s = self.conv2(s)
        s = self.activation2(s)
        return x * s

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=bias)

def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias)

def conv3x3_dw(channels, stride=1):
    return torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False)

def conv5x5_dw(channels, stride=1):
    return torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=channels,
            bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = (activation is not None)

        self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  activation="relu"):
    return ConvBlock(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=stride,
             padding=0,
             bias=bias,
             use_bn=use_bn,
             activation=activation)

def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  activation="relu"):
    return ConvBlock(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=3,
             stride=stride,
             padding=1,
             bias=bias,
             use_bn=use_bn,
             activation=activation)
def conv3x3_dw_block(channels,
                     stride=1,
                     use_bn=True,
                     activation="relu"):
    return ConvBlock(
             in_channels=channels,
             out_channels=channels,
             kernel_size=3,
             stride=stride,
             padding=1,
             groups=channels,
             use_bn=use_bn,
             activation=activation)

def conv5x5_dw_block(channels,
                     stride=1,
                     use_bn=True,
                     activation="relu"):
    return ConvBlock(
             in_channels=channels,
             out_channels=channels,
             kernel_size=5,
             stride=stride,
             padding=2,
             groups=channels,
             use_bn=use_bn,
             activation=activation)

class DepthwiseSeparableConvBlock(torch.nn.Module):
    """
    Depthwise-separable convolution (DSC) block internally used in MobileNets.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()

        self.conv_dw = conv3x3_dw_block(channels=in_channels, stride=stride)
        self.conv_pw = conv1x1_block(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class LinearBottleneck(torch.nn.Module):
    """
    Linear bottleneck block internally used in MobileNets.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 activation="relu6",
                 kernel_size=3,
                 use_se=False):
        super().__init__()
        self.use_res_skip = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se

        self.conv1 = conv1x1_block(in_channels=in_channels, out_channels=mid_channels, activation=activation)
        if kernel_size == 3:
            self.conv2 = conv3x3_dw_block(channels=mid_channels, stride=stride, activation=activation)
        elif kernel_size == 5:
            self.conv2 = conv5x5_dw_block(channels=mid_channels, stride=stride, activation=activation)
        else:
            raise ValueError
        if self.use_se:
            self.se_unit = SEUnit(channels=mid_channels, squeeze_factor=4, squeeze_activation="relu", excite_activation="hsigmoid")
        self.conv3 = conv1x1_block(in_channels=mid_channels, out_channels=out_channels, activation=None)

    def forward(self, x):
        if self.use_res_skip:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se_unit(x)
        x = self.conv3(x)
        if self.use_res_skip:
            x = x + residual
        return x

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        
        self.classifier=nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
    
class MobileNetV3(torch.nn.Module):
    def __init__(self,num_classes=10,type='large',start_channel=3):
        super().__init__()
        self.use_data_batchnorm = True
        self.in_size = (32,32)
        self.dropout_rate = 0.8
        init_conv_stride=1
        init_conv_channels = 16

        if type == "small":
            channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
            mid_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
            strides = [1, 2, 2, 2]
            kernel_sizes = [3, 3, 5, 5]
            activations = ["relu", "relu", "hswish", "hswish"]
            se_units = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
            final_conv_channels = [576,1024]
            final_conv_se = True
        elif type == "large":
            channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
            mid_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
            strides = [1, 1, 2, 2, 2]
            kernel_sizes = [3, 3, 5, 3, 5]
            activations = ["relu", "relu", "relu", "hswish", "hswish"]
            se_units = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
            final_conv_channels = [960,1280]
            final_conv_se = False
        else:
            raise NotImplementedError
    
        self.layers=[]
        self.first_block = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.first_block.add_module("data_bn", torch.nn.BatchNorm2d(num_features=start_channel))

        # init conv
        self.first_block.add_module("init_conv", conv3x3_block(in_channels=start_channel, out_channels=init_conv_channels, stride=init_conv_stride, activation="hswish"))

        self.layers.append(self.first_block)
        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            for unit_id, unit_channels in enumerate(stage_channels):
                stage = torch.nn.Sequential()
                stride = strides[stage_id] if unit_id == 0 else 1
                mid_channel = mid_channels[stage_id][unit_id]
                use_se=se_units[stage_id][unit_id] == 1
                kernel_size = kernel_sizes[stage_id]
                activation = activations[stage_id]
                stage.add_module("unit{}".format(unit_id + 1), LinearBottleneck(in_channels=in_channels, mid_channels=mid_channel, out_channels=unit_channels, stride=stride, activation=activation, use_se=use_se, kernel_size=kernel_size))
                in_channels = unit_channels
                self.layers.append(stage)

        self.final_block = torch.nn.Sequential()
        self.final_block.add_module("final_conv1", conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels[0], activation="hswish"))
        in_channels = final_conv_channels[0]
        if final_conv_se:
            self.final_block.add_module("final_se", SEUnit(channels=in_channels, squeeze_factor=4, squeeze_activation="relu", excite_activation="hsigmoid"))
        self.final_block.add_module("final_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        if len(final_conv_channels) > 1:
            self.final_block.add_module("final_conv2", conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels[1], activation="hswish", use_bn=False))
            in_channels = final_conv_channels[1]
        if  self.dropout_rate != 0.0:
            self.final_block.add_module("final_dropout", torch.nn.Dropout(self.dropout_rate))
        self.layers.append(self.final_block)

        # classifier
        self.layers.append(Classifier(in_channels=final_conv_channels[-1], num_classes=num_classes))

    






