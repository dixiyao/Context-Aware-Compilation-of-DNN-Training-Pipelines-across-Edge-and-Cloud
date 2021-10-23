import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import time
import pickle
import lz4.frame

import decision.engine as decision_engine

def edge_forward(models,inputs,use_Q):
    x=inputs.cuda()
    for model in models:
        x=model(x)
    x=x.detach().cpu()
    if not use_Q:
        return x,0
    else:
        min_range = max(torch.min(x).item(),-128)
        max_range = min(torch.max(x).item(),128)
        Q = torch.quantize_per_tensor(x, scale=(max_range-min_range)/(2**8), zero_point=int(min_range), dtype=torch.qint8)
        E=torch.dequantize(Q)-x
        Q = pickle.dumps(Q)
        return Q,E

def edge(iters,models,inputs,label,meter,optimizers):
    label=label.cuda()
    outputs=Variable(inputs)
    x=outputs.cuda()
    for model in models:
        x=model(x)
    loss=F.cross_entropy(x,label)
    for optim in optimizers:
        if optim=="FREE":
            pass
        else:
            optim.zero_grad()
    loss.backward()
    for optim in optimizers:
        if optim=="FREE":
            continue
        else:
            optim.step()
    tt,loss_i,cc=summary_iter(x.detach().cpu(),label.cpu())
    meter.update(tt,loss_i,cc)
    if iters%100==0:
        print('train %03d %e %f'%(iters,meter.losses/meter.cnt,meter.correct/meter.cnt))

def cloud(models,inputs,lable,optimizers,point,useQ):
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
    for optim in optimizers[point:]:
        if optim=="FREE":
            continue
        else:
            optim.step()
    grad=outputs.grad.detach()
    return x.detach().cpu(),grad.cpu()

def edge_backward(models,gradients,inputs,optimizers,point):
    try:
        x=torch.autograd.Variable(inputs,requires_grad=True).cuda()
        for model in models:
            x=model(x)
    except:
        x=models[0](inputs.cuda())
        x=torch.autograd.Variable(x,requires_grad=True).cuda()
        for model in models[1:]:
            x=model(x)
    gradients=gradients.cuda()
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
        x=data.cuda()
        for model in models:
            x=model(x)
        feature=copy.copy(x)
        feature=feature.detach().cpu()
        client.send_tensor('valid',-2,i,target,feature,False)
        #download
        _,_,_,result,gradient,_,_,_,_,_,_=client.recieve_tensor()
        _,id=torch.max(result,1)
        correct=torch.sum(id==target.data)
        t+=correct.data.item()
        c+=target.shape[0]
    print("test:{}".format(t/c))     

def summary_iter(result,label):
    _,id=torch.max(result,1)
    correct=torch.sum(id==label.data)
    loss=F.cross_entropy(result.cuda(),label.cuda())
    return correct.data.item(),loss.data.item(),label.shape[0]

def edge_backprocess(head,epoch,iters,result,gradient,Q_history,meter,models,optims,point,report_freq=100):
    item='None'
    topic='None'
    #print(head)
    if head=='Train':
        s=time.time()
        bepoch,bi,inputs,label,E=Q_history.get()
        #error feedback
        if not isinstance(E,int):
            heisen=gradient*gradient
            gradient=gradient-E*heisen
        #print(epoch==bepoch and bi==iters)
        edge_backward(models,gradient,inputs,optims,point)
        tt,loss_i,cc=summary_iter(result,label)
        meter.update(tt,loss_i,cc)
        if iters%report_freq==0:
            print('train %03d %e %f'%(iters,meter.losses/meter.cnt,meter.correct/meter.cnt))
        #print(bi,"backward",time.time()-s)
        #print(bi,'end',time.time())
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

def dynamic_decision(upload,download,models,global_models,remain_epoch,edge,cloud,feature_size,model_size,K,point,qtime):
    #test bandwdith
    upload=abs(upload)
    download=abs(download)
    #print("upload speed {} and download speed {}".format(upload,download))
    #change partition
    estimate_latency,new_point, use_Q=decision_engine.decide_point(edge,cloud,feature_size,upload,download,model_size,point,K,remain_epoch,qtime)

    return estimate_latency,new_point,use_Q

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
