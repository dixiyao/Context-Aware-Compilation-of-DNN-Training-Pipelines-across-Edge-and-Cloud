import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import time


def cloud_forward(models,input_c,input_v,input_t,use):
    input_c=input_c.cuda(non_blocking=True)
    input_v=input_v.cuda(non_blocking=True)
    input_t=input_t.cuda(non_blocking=True)
    x=(input_c,input_v,input_t)
    for model in models:
        x=model(x)
    return x.detach().cpu(),0

def edge(models,inputs,target_image,target_mask,optimizers,point):
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()
    target_image=target_image.cuda(non_blocking=True)
    target_mask=target_mask.cuda(non_blocking=True)
    if point==0:
        input_c,input_v,input_t=inputs
        input_c=input_c.cuda(non_blocking=True)
        input_v=input_v.cuda(non_blocking=True)
        input_t=input_t.cuda(non_blocking=True)
        x=(input_c,input_v,input_t)
    else:
        outputs=Variable(inputs, requires_grad=True).cuda(non_blocking=True)
        outputs.retain_grad()
        x=outputs
    for model in models:
        x=model(x)
    out_image,out_mask=x
    loss1 = criterion1(out_image, target_image)
    loss2 = criterion2(out_mask, target_mask)
    loss = loss1 + 0.1 * loss2
    for optim in optimizers[point:]:
        if optim=="FREE":
            continue
        else:
            optim.zero_grad()
    loss.backward()
    for optim in optimizers[point:]:
        if optim=="FREE":
            continue
        else:
            optim.step()
    if point==0:
        grad=torch.tensor([-1.1])
    else:
        grad=outputs.grad.detach().cpu()
    return loss.detach().cpu().data.item(),grad

def cloud_backward(models,gradients,Q_history,optimizers,point):
    epoch,i,input_c,input_v,input_t,E=Q_history.get()
    input_c=input_c.cuda(non_blocking=True)
    input_v=input_v.cuda(non_blocking=True)
    input_t=input_t.cuda(non_blocking=True)
    x=(input_c,input_v,input_t)
    gradients=gradients.cuda(non_blocking=True)
    for model in models:
        x=model(x)
    for optim in optimizers[:point+1]:
        if optim=="FREE":
            pass
        else:
            optim.zero_grad()
    x.backward(gradients)
    for optim in optimizers[:point+1]:
        if optim=="FREE":
            continue
        else:
            optim.step()
        
def test_edge(models,client,test_dataloader):
    t=0
    c=0
    for i, (data, target) in enumerate(test_dataloader):
        x=data.cuda(non_blocking=True)
        for model in models:
            x=model(x)
        feature=copy.copy(x)
        feature=feature.detach().cpu()
        client.send_tensor('valid',-2,i,target,feature)
        #download
        _,_,_,result,gradient=client.recieve_tensor()
        _,id=torch.max(result,1)
        correct=torch.sum(id==target.data)
        t+=correct.data.item()
        c+=target.shape[0]
    print("test:{}".format(t/c))     

def summary_iter(result,label):
    _,id=torch.max(result,1)
    correct=torch.sum(id==label.data)
    loss=F.cross_entropy(result,label)
    return correct.data.item(),loss.data.item(),label.shape[0]

def edge_backprocess(head,epoch,iters,result,gradient,Q_history,meter,models,optims,point,report_freq=100):
    item='None'
    topic='None'
    #print(head)
    if head=='Train':
        bepoch,bi,inputs,label=Q_history.get()
        #print(epoch==bepoch and bi==iters)
        edge_backward(models,gradient,inputs,optims,point)
        tt,loss_i,cc=summary_iter(result,label)
        meter.update(tt,loss_i,cc)
        if iters%report_freq==0:
            print('train %03d %e %f'%(iters,meter.losses/meter.cnt,meter.correct/meter.cnt))
    elif head=='Valid':
        inputs,label=Q_history.get()
        tt,loss_i,cc=summary_iter(result,label)
        meter.update(tt,loss_i,cc)
        if iters%report_freq==0:
            print('valid %03d %e %f'%(iters,meter.losses/meter.cnt,meter.correct/meter.cnt))
    elif head=='EndTrain':
        loss=meter.losses/meter.cnt
        acc=meter.correct/meter.cnt
        meter.reset()
        torch.cuda.synchronize()
        end=time.time()
        item='EndTrain'
        topic=(loss,acc,end)
    elif head=='EndValid':
        acc=meter.correct/meter.cnt
        meter.reset()
        item='EndValid'
        topic=acc
    return item,topic    

def dynamic_decision(client,models,global_models,remain_epoch,edge,cloud,feature_size,model_size,K,point):
    #test bandwdith
    upload,download=client.testspeed()
    upload=abs(upload)
    download=abs(download)
    #print("upload speed {} and download speed {}".format(upload,download))
    #change partition
    estimate_latency,new_point=decision_engine.decide_point(edge,cloud,feature_size,upload,download,model_size,point,K,remain_epoch)

    return estimate_latency,new_point,upload,download

def dynamic_change(client,models,global_models,point,new_point):
    for model in models:
        model=model.cpu()
    global_models[:point+1]=models
    if point<new_point:
        #download point+1,...,new_point from cloud
        client.recieve_and_update_model(global_models)
    else:
        #upload new_point+1,...,point to cloud
        index=(new_point+1,point)
        client.send_model(index,global_models[new_point+1:point+1])
    models=global_models[:new_point+1]
    for model in models:
        model=model.cuda()
        model.train()
    return models
        
def cloud_dynamic_change_model(global_models,models,point,epoch,iters,server):
    global_models[point:]=models
    for model in models:
        model=model.cpu()
    if epoch<iters:
        #send param to edge
        index=(epoch+1,iters)
        server.send_model(index,global_models[epoch+1:iters+1])
    else:
        #recv param from edge
        server.recieve_and_update_model(global_models)
    point=iters+1
    models=global_models[point:]
    for model in models:
        model=model.cuda()
        model.train()
    return point, models    

def cloud_heisen(models,inputs,lable,optimizers,point,use_BNN=False):
    lable=lable.cuda()
    outputs=Variable(inputs, requires_grad=True).cuda()
    outputs.retain_grad()
    x=outputs
    for model in models:
        x=model(x)
    loss=F.cross_entropy(x,lable)
    for optim in optimizers[point:]:
        if optim=="FREE":
            pass
        else:
            optim.zero_grad()
    loss.backward()
    if use_BNN:
        for model in models:
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
    for optim in optimizers[point:]:
        if optim=="FREE":
            continue
        else:
            optim.step()
    if use_BNN:
        for model in models:
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
    grad=outputs.grad.detach()
    heisen=grad*grad

    return x.detach().cpu(),grad.cpu(),heisen.cpu()
