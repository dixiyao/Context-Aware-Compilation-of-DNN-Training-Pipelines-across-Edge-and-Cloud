import torch
# import hyper.ErrorFeedbackSGD as efsgd


def get_optim(model, name, learning_rate=0.01):
    if name=='resnet':
        optim=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=0.0001,nesterov=True)
        sch=torch.optim.lr_scheduler.MultiStepLR(optim, [81,122], gamma=0.1, last_epoch=-1)
    elif name=='resnetpre':
        optim=torch.optim.RMSprop(model.parameters(), lr=0.00005,
                              weight_decay=0.0001)
        sch=torch.optim.lr_scheduler.MultiStepLR(optim, [1000], gamma=0.1, last_epoch=-1)
    elif name=='mobilenet':
        optim=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,
                                  weight_decay=0.00001)
        sch=torch.optim.lr_scheduler.MultiStepLR(optim, [100,150,180], gamma=0.1, last_epoch=-1)
    elif name=='tinyimagenet':
        optim = torch.optim.Adam(model.parameters(), weight_decay=0.001, lr=8e-5)
        sch = torch.optim.lr_scheduler.MultiStepLR(optim, [3,20], gamma=0.5, last_epoch=-1)
    elif name=='imdb':
        optim = torch.optim.Adam(model.parameters())
        sch=torch.optim.lr_scheduler.MultiStepLR(optim, [1000], gamma=0.1, last_epoch=-1) 
    else:
        print('No such predefined optimizers')
    return optim, sch
