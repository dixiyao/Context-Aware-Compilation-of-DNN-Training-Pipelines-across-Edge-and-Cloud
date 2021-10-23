import os
import re

import numpy as np
from numpy import newaxis
import pandas
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch.utils import data
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
from utils.fs_utils import get_all_filenames


class Dataset():
    def __init__(self, folderPath, is_train=True, output_size=128):
        """
         Arguments:
            is_train: whether used for training or validataion
            folderPath: like "../data", is the folder storing chairs.train.csv and chairs.valid.csv 
            output_size: the desired output size of images and masks, in the paper, it is 64, 128, or 256.             
        """
        # get the path of train.csv and valid.csv
        if is_train:
            csv_path = folderPath + '/train.csv'
            print("loading train dataset")
        else:
            csv_path = folderPath + '/valid.csv'
            print("loading validataion dataset")    
        # read the data
        self.data_info = pd.read_csv(csv_path, header=None)
        # get the image path
        # Note to remove the header!!
        self.images =  np.asarray(self.data_info.iloc[:, 0])[1:]
        self.categories = np.asarray(self.data_info.iloc[:, 1])[1:]
        print('loaded ', len(self.images), ' files')
        # print(self.files)
        
        # do the transformation
        self.transformations = transforms.Compose([transforms.Resize(output_size),\
                                                        transforms.ToTensor()])
        self.sigmoid=nn.Sigmoid()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # load images
        image_filename = self.images[index]
        image = Image.open(image_filename)
        image = self.transformations(image)

        # load the category of the image
        category = int(self.categories[index])

        # load masks
        mask=self.sigmoid(image)
        
        # path: like "../data/rendered_chairs/b663f1e4df6c51fe19fb4103277a6b93/renders"
        # filename: like "image_001_p020_t011_r096.png"
        _, filename = os.path.split(image_filename)
        
        # get the pieces of image filename to parse the parameters
        # pieces = filename.split('_')
        
        #parse the parameters from file name and return those too
        pieces = filename.split('_')[1:] #split and throw images away

        # print(pieces)
        # id = pieces[0]
        # remove the first char to get the parameters
        phi = int(pieces[1].strip('p'))
        theta = int(pieces[2].strip('t'))
        #rho = int(pieces[3].strip('r').split('.')[0])
        
        # construct the class vector
        c = np.zeros(809)
        c[category] = 1
        
        # construct the view vector
        v = np.zeros(4)
        v[0] = np.sin(theta/180 * np.pi)
        v[1] = np.cos(theta/180 * np.pi)
        v[2] = np.sin(phi/180 * np.pi)
        v[3] = np.sin(phi/180 * np.pi)
        
        # construct the tranform parameter vector
        t = np.ones(12)
        
        # transform them into tensor and into tench.FloatTensor
        c = torch.from_numpy(c)
        v = torch.from_numpy(v)
        t = torch.from_numpy(t)
        c = c.float()
        v = v.float()
        t = t.float()

        return image, mask, c, v, t


# dataset = Dataset('datasets/lspet_dataset', myTransforms.Compose([myTransforms.TestResized((368, 368))]))
# loader = DataLoader(
#     dataset,
#     batch_size=10,
#     num_workers=0,
#     shuffle=False
# )

def computeStatistics(loader):

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

# if __name__ == '__main__':
#     print('computing std and mean of LSPet')
#     print(computeStatistics(dataset))


if __name__ == '__main__':

    image, mask, phi, theta, rho = Dataset('../data', is_train=False).__getitem__(100)
    print(image.size())
    print(mask.size())
    print(image.numpy().shape)
    print(mask.numpy().shape)
    #plt.imshow(image.numpy().transpose(1, 2, 0))
    plt.imshow(mask.numpy().transpose(1, 2, 0))
    plt.show()
    print("phi: ", phi)
    print("theta: ", theta)
    print("rho: ", rho)
    
