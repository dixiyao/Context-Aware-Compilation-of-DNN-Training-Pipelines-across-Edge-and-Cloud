import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import pandas as pd

def check_full(Q,E):
    while True:
        if not Q.full():
            break
        else:
            E.wait()

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.losses = 0
        self.cnt = 0

    def update(self, correct,loss,cnt):
        self.correct += correct
        self.cnt += cnt
        self.losses += loss


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_svhn(args):
    SVHN_MEAN = [0.43768206, 0.44376972, 0.47280434]
    SVHN_STD = [0.19803014, 0.20101564, 0.19703615]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
    return train_transform, valid_transform

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_tensorsize_in_B(tensor):
    tensor=tensor.detach().numpy()
    words=b'head'+pickle.dumps('head')+b'epoch'+pickle.dumps(0)+b'iters'+pickle.dumps(0)+b'tensor'+pickle.dumps(tensor)+b'end of feature'
    return len(words)       

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def get_profile(runtimefile,featuresizefile,model,cloud_device,edge_device):
    runtime=pd.read_csv(runtimefile+'_'+model+'.csv')
    cloud=np.array(runtime[cloud_device])
    edge=np.array(runtime[edge_device])
    feature=pd.read_csv(featuresizefile+'.csv')
    feature_size=np.array(feature[model])

    return edge,cloud, feature_size

def count_model_size_in_MB(model):
    Total_params=0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
    return Total_params*4/1024/1024

def count_models_size_in_MB(models):
    return [count_model_size_in_MB(model) for model in models]
    
