# Check for baselines
## User Command Lines Setting
### ResNet
```
python3 base.py --dataset cifar10 --splits 19  --epochs 160 --optim resnet --model resnet50 
```
### MobileNetV3(large)
```
python3 base.py --dataset cifar10 --splits 18  --epochs 200 --optim mobilenet --model Mobilenetv3_large
```
### VGG
```
python3 base.py --dataset cifar10 --splits 19  --epochs 100 --optim resnet --model VGG19
```
### BI-LSTM
```
python3 base.py --dataset imdb --splits 5  --epochs 10 --optim imdb --model BILSTM
```
### ResNet (transfer to Cifar10)
```
python3 base.py --dataset cifar10 --splits 19  --epochs 5 --optim resnet_pre --model resnet50 --pretrained
```
### ResNet (transfer to Tiny Imagenet, Type1)
```
python3 base.py --dataset tiny_imagenet --splits 19  --epochs 5 --optim tinyimagenet --model resnet50_imgnet --pretrained --num_classes 20
```
### ResNet (transfer to Tiny Imagenet, Type2)
```
python3 base.py --dataset tiny_imagenet --splits 19  --epochs 5 --optim tinyimagenet --model resnet50_imgnet --pretrained --num_classes 20 --freeze 1
```
## DIY model and dataset
### Model
If your model or dataset is out of our scope, you can write pytorch model in the form of split format (reffering to models in Model folder). It is quite simple. In common way, you may write a forward function to align the forwarding route. Now, you just delete forward function and put the forwarding sequence in a member of Model CLASS named layers, and use self.layers.append to adding these forwarding paths sequentially.
### Dataset
Copy Pytorch dataset loading processes into DataLoader

