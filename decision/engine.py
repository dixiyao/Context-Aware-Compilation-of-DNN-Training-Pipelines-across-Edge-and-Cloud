import numpy as np
import math

def time_parellel(T,K,r):
    T_star=max(T)
    T_0=sum(T)
    t=(r-1)*T_star+T_0+(math.ceil(r/(K+1))-1)*max(0,T_0-(K+1)*T_star) 
    return t
    

def time_serial(T):
    return sum(T)

def time_update(modelsize,l,new_l,upload,download):
    if l==new_l:
        return 0
    elif new_l>l:
        #download
        return sum(modelsize[l+1:new_l+1])/download
    else:
        #upload
        return sum(modelsize[new_l+1:l+1])/upload

'''
compbox: calculate T1, replay+backward+forward
commubox, upload: calculate T2 ,T2=commu/upload
commubox, download: calculate T4, T4=commu/download
cloudbox: calculate T3

use some allgorithm to get T1,T2,T3,T4 and the estimated time and give the decicisioni point

K is the limits of stale iterations in feature replay

point: partition point, point at i. from 0 to i on edge, i+1 to final on cloud
'''
def decide_point(compbox,cloudbox,commubox,upload,download, model_size,l,K,remain,qtime):
    length=len(compbox)
    compbox=np.array(compbox)
    cloudbox=np.array(cloudbox)
    commubox=np.array(commubox)
    mintime=100000000
    point=-1
    use_Q=False
    decision_function=time_parellel
    times=[]
    times2=[]
    for i in range(length-1):
        for j in [True,False]:
            T1=compbox[i]
            T2=commubox[i]/upload
            T3=cloudbox[i]
            T4=commubox[i]/download
            #print(T1,T2,T3,T4)
            if j==True:
                T2=T2/4
                T1=T1+qtime
            latency=decision_function([T1,T2,T3,T4],K,remain)
            T_update=time_update(model_size,l,i,upload,download)
            t=latency+T_update    
            if t<mintime:
                mintime=t
                point=i
                use_Q=j
            times.append(latency)
            times2.append(T_update)
        #print(times)
        #print(times2)
    T_update=time_update(model_size,l,length-1,upload,download)
    t=compbox[length-1]*remain+T_update
    if t<mintime:
        mintime=t
        point=length-1
        use_Q=False
        
    return mintime/remain,point, use_Q
    
