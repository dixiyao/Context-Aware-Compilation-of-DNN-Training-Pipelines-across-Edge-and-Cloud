"""
This script is for getting the mean and std of three channels of the datasets. 
"""

import numpy as np
from skimage import io
import torch
import pandas as pd

# get the .csv paths
train_csv_path = "../data/chairs.train.class.csv"
valid_csv_path = "../data/chairs.valid.class.csv"

# read the dataset and append three channels respectively
train_df = pd.read_csv(train_csv_path, header=None)
train_image = np.asarray(train_df.iloc[:, 0])[1:]
valid_df = pd.read_csv(valid_csv_path, header=None)
valid_image = np.asarray(valid_df.iloc[:, 0])[1:]

mean = 0.
std = 0.
nb_samples = 0.

for image in train_image:
    image = torch.from_numpy(io.imread(image).transpose((2, 0, 1)))
    image = image.view( image.size(0), -1)
    mean += image.float().mean(1)
    std += image.float().std(1)
    nb_samples += 1

for image in valid_image:
    image = torch.from_numpy(io.imread(image).transpose((2, 0, 1)))
    image = image.view( image.size(0), -1)
    mean += image.float().mean(1)
    std += image.float().std(1)
    nb_samples += 1

mean /= nb_samples
std /= nb_samples

print("mean: ", mean)
print("std: ", std)
