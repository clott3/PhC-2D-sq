#!/usr/bin/env python3

import torch
from torch.utils import data
import numpy as np
import h5py
import os
import math
from skimage.transform import resize

class PhCdata(data.Dataset):

    def __init__(self, path_to_h5_file, trainsize, validsize, testsize,
        split='train', nbands = 6, input_size = 32):

        """
        :param path_to_h5_file: path to h5 file for dataset
        :param trainsize: size of training set
        :param validsize: size of valid set
        :param testsize: size of test set
        :param split: to retrieve train, validation or test set
        :param nbands: specify number of bands, this code allows max 10 bands (moduleDict is single digit)

        """

        # the following is to make a fix set of train-valid-test split
        totalstart = 1

        if split == 'train':
            indstart = totalstart
            indend = indstart + trainsize
        elif split == 'valid':
            indstart = totalstart + trainsize
            indend = indstart + validsize
        elif split == 'test':
            indstart = totalstart + trainsize + validsize
            indend = indstart + testsize

        self.len = indend - indstart
        self.input_size = input_size

        ## initialize data lists
        self.x_data = []
        self.y_data = []

        with h5py.File(path_to_h5_file, 'r') as f:
            for memb in range(indstart,indend):

                eps = f['shapes/'+str(memb)+'/unitcell/epsilon_comput'][()]
                inputeps = resize(np.array(eps),(self.input_size,self.input_size))

                y = f['shapes/'+str(memb)+'/eigfreqs'][()][:nbands]

                self.x_data.append(inputeps)
                self.y_data.append(y.T) ## data has dim (batchsize, output_dim, nbands)

        # normalize x data
        self.x_data = (np.array(self.x_data).astype('float32')-1) / 9 # normalize
        self.x_data = self.x_data.reshape(self.x_data.shape[0],1,self.input_size,self.input_size) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('float32')


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        :return: random sample
        """
        ## input always first element in tuple and output always second element
        return self.x_data[index], self.y_data[index]
