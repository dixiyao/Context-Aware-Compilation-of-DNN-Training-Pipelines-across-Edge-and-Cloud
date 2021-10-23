import numpy as np

# single channel
def zeroMean1d(data):
    # center the data
    return data - data.mean(axis=0)

# 3 channels
def zeroMean3d(data):
    for i in range(3):
        means = [m for m in np.mean(data, axis = (3, i))]
    # center the data
    return data - means

def covariance(data):
    # compute the data's covariance matrix
    return np.dot(data.T, data) / data.shape[0]

def normalize1d(data):
    # rescale by the standard deviation, makes little to no difference?
    return data / np.std(data, axis = 0)

def normalize3d(data):
    for i in range(3):
        stds = [s for s in np.std(data, axis = (3, i))]
    return data / stds

def decorrelate(data):
    cov = covariance(data)
    U,S,V = np.linalg.svd(cov)

    # use result of PCA (eigenvectors form a rotation xform) to rotate data s.t. it's aligned with the principal axes
    Xrot = np.dot(data, U)
    # add 1e-6 to avoid divide by 0 errors
    return Xrot / np.sqrt(S + 1e-6)

def PCAWhitening(data):
    data = zeroMean1d(data)
    # unnecessary since the final step involves rescaling data's eigenbasis by the eigenvalue anyway
    # data = normalize(data)

    data = decorrelate(data)
    #double check it's the identity
    # cov = covariance(data)

    return data

def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)