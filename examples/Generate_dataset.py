import os, sys, gzip
import numpy as np
import pandas as pd
import h5py

datapath='/home/peter/Data/CMAP/training.csv'
savepath='/home/peter/Data/CMAP/data.hd5f'

f = h5py.File(savepath, "w")
shuffle_index = np.random.permutation(100000)
data = pd.read_csv(datapath, header=None, dtype=np.float32)
genes = data.as_matrix()
num_landmark = 970
num_nonlandmark = 11350
num_samples = genes.shape[1]
landmark = genes[:970,:]
nonlandmark = genes[970:,:]
del genes

dset = f.create_dataset("landmark", data=landmark)
dset = f.create_dataset("nonlandmark", data=nonlandmark)
del landmark
del nonlandmark
