import models.sliced_model as sliced_model
import models.ResNets as ResNets
import models.ResNets_img as ResNets_imgnet
import models.DenseNets as DenseNets
import models.MobileNet as MobileNet
import models.VGG as VGG
import models.BILSTM as BILSTM
import torchvision.models as backbones
import torch

def get_model(model,splits,pretrain,num_classes=10,start_channel=3):
    if model=='resnet50':
        model=ResNets.resnet50(num_classes=num_classes,start_channel=start_channel)
        models=sliced_model.get_sliced_model(model,splits)
        if pretrain==True:
            sliced_model.pretrain(models,backbones.resnet50(pretrained=True))
    elif model=='VGG19':
        model=VGG.VGG_19()
        models=sliced_model.get_sliced_model(model,splits)
        if pretrain==True:
            sliced_model.pretrain(models,backbones.VGG19(pretrained=True))
    elif model=='Mobilenetv3_large':
        model=model=MobileNet.MobileNetV3(num_classes=num_classes,type='large',start_channel=start_channel)
        models=sliced_model.get_sliced_model(model,splits)
    elif model=='resnet50_imgnet':
        model=ResNets_imgnet.resnet50(num_classes=num_classes,start_channel=start_channel)
        models=sliced_model.get_sliced_model(model,splits)
        if pretrain==True:
            sliced_model.pretrain(models,backbones.resnet50(pretrained=True))
    elif model=='BILSTM':
        models=BILSTM.getmodel(10000,128,128,0,torch.device("cuda"if torch.cuda.is_available()else "cpu"))
    else:
        raise ValueError('no such model')
    return models
            
        
        
