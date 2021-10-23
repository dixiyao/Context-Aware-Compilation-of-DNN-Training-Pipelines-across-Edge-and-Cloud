# User Command Settings
## ResNet50
```
python3 server.py --model resnet50 --num_clasees 10 --splits 19 
```
and
```
python3 client.py --data_path ../data --dataset cifar10 --splits 19 --epochs 160 --optim resnet --stale_it 5 --model resnet50 --num_classes 10 
```
## MobileNetV3 (large)
```
python3 server.py --model Mobilenetv3_large --num_clasees 10 --splits 18
```
and
```
python3 client.py --data_path ../data --dataset cifar10 --splits 18 --epochs 200 --optim mobilenet --stale_it 5 --model Mobilenetv3_larg e--num_classes 10 
```
## VGG19
```
python3 server.py --model VGG19 --num_clasees 10 --splits 19
```
and
```
python3 client.py --data_path ../data --dataset cifar10 --splits 19 --epochs 100 --optim resnet --stale_it 5 --model VGG19 --num_classes 10 
```
## ResNet50 Tiny ImageNet
```
python3 server.py --model resnet50_imgnet --num_clasees 10 --splits 19 
```
and
```
python3 client.py --data_path ../data --dataset tiny_imagenet --splits 19 --epochs 5 --optim resnet_pre --stale_it 5 --model resnet50_imgnet --pretrained --num_classes 20 
```
## BI-LSTM
```
python3 server.py --model BILSTM --num_clasees 2 --splits 5
```
and
```
python3 client.py --data_path ../data --dataset imdb --splits 5 --epochs 10 --optim imdb --stale_it 4 --model BLSTM --num_classes 2
```