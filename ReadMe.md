# Overview
This is the implementation of paper **Context-Aware Compilation of DNN Training Pipelines across Edge and Cloud**
## Requirements
python 3.x, pytorch, torchvision, pandas
## DataSets
default dataset path is **data**. CIFAR10 can be automatically downloaded with *torchvision* tools. IMDB can be automatically downloaded with *keras* tools. Tiny-ImageNet please refer to http://cs231n.stanford.edu/tiny-imagenet-200.zip. Chair downloading guidelies is in **generative/readme.txt**. MovieLens 1M dataset is included in **DIN**.
# Running our system
client.py is deployed on the edge and the server.py is deployed on the cloud
## Discriminative model
To run the discriminative model (ResNet, Vgg, MobilNet, BI-LSTM), the inplementation is under **discriminative**

First start server
```
python3 server.py --ip your-ip --port your-port --hyperport your-port-2
```
and then start client
```
python3 client.py --ip your-ip --port your-port --hyperport your-port-2
```
and

Then you can run our system. For various models and use cases, the setting commands are different, you can refer to them under discriminative folder

## Generative model
To run the generative model (Chair), the inplementation is under **generative**
First start server
```
python3 server.py --ip your-ip --port your-port --hyperport your-port-2
```
and then start client
```
python3 client.py --ip your-ip --port your-port --hyperport your-port-2
```
## Cross-Slio Federated Learning
To run the Cross-Slio Federated Leaningr model such as DIN, the inplementation is under **DIN**

First start server
```
python3 server.py --ip your-ip --portA port-for-ClientA --portB port-for-ClientB
```
Then start clientA
```
python3 clientA.py --ip your-ip --portA port-for-ClientA
```
Finally start cientB
```
python3 clientB.py --ip your-ip --portA port-for-ClientB
```
# Other Useful tools
## Baselines
We also provide codes for checking our baselines in baseline folder. 

base.py is running the original model without any advanced methods, the user command setting for different models and datasets are diffefrent you can refer to them detailedly under **baseline**

JointDNN can be checked by just set client.py with
```
--stale_it 0
```
## Ablation Study
To check the impacts of using feature replay error feedback, we can also use base.py to do the abaltion study.

To check feature replay, set base.py with
```
--stale_it K --split_index le
```
To check error feed back, set base.py with
```
--use_EF 1
```
Also, we can change the compression methods in both baseline and acutal system implementations.
To investigate about weights distribution and track weights, set base.py with
```
--track_weights 1 --track_point 0,1,2,3,....
```
## Profile and Measurements
Energy Consumption and Memory cost are critical for edge devices. So, we provide code to test energy consumption on TX2 in **quantify/energy.py** and memory test in **quantify/memory.py**

We also provide our profiling results under **profile**
### Requirements
psutil, tkinter and matpolib (for visualization)