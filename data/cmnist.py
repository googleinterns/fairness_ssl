"""Waterbirds data loader."""

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets

from data import data_util

from util.utils import DEFAULT_MISSING_CONST as DF_M
from util.utils import flip_bit

import pdb

DATA_DIRECTORY = 'data/datasets/mnist_dataset/'
    
class CMNIST(object):
    """CMNIST data loader."""

    def __init__(self, lab_split = 1.0, reweight = False, shuffle_train = True, seed = 42):
        print('Using CMNIST dataset!')

        self.data_dir = DATA_DIRECTORY
        self.dataseed = seed
        self.lab_split = lab_split
        self.reweight = reweight
        self.shuffle_train = shuffle_train
        
        # Noise attributes
        self.label_noise = 0.25
        self.color_noise = [0.2, 0.1, 0.9]

        # Load data
        mnist = datasets.MNIST(self.data_dir, train=True, download=True)
        images = mnist.data
        images = images.float() / 255.
        images = images.reshape((-1, 28, 28))[:, ::2, ::2] # reduce image dimensions to -1x14x14
        images = torch.stack([images, images], dim=1) # add channels for adding color
        self.sample = images

        # Target values
        self.target = mnist.targets
        self.target = (self.target < 5).float()
        self.n_targets = len(np.unique(self.target))
        
        # Control values
        probs = [0.53, 0.4, 0.07] # Marginal probabilities
        np.random.seed(self.dataseed+1)
        self.control = np.random.choice(len(probs), size=len(self.sample), p=probs)
        self.n_controls = len(np.unique(self.control))
        assert self.n_controls == len(probs) == len(self.color_noise), "Error in control list"
        
        # Split idx
        probs = [0.66, 0.17, 0.17] # train, val and test splits
        np.random.seed(self.dataseed+2)
        self.split_idx = np.random.choice(len(probs), size=len(self.sample), p=probs)

        # Prepare each control environment by adding noise
        self.add_color_label_noise()
        
        # Split train, valid, test
        sm_train, sm_valid, sm_test, \
            y_train, y_valid, y_test, \
            self.c_train, c_valid, c_test = self.generate_splits()

        # Create Torch Custom Datasets
        self.train_set = data_util.ImageFromMemory(sample=sm_train, \
                                                   target=y_train, \
                                                   control=self.c_train)
        
        self.val_set = data_util.ImageFromMemory(sample=sm_valid, \
                                                 target=y_valid, \
                                                 control=c_valid)
        
        self.test_set = data_util.ImageFromMemory(sample=sm_test, \
                                                  target=y_test, \
                                                  control=c_test)
        return


    def add_color_label_noise(self):
        for cid in range(self.n_controls):
            images = self.sample[self.control==cid]
            labels = self.target[self.control==cid]

            label_noise = (torch.rand(len(labels)) < self.label_noise).float()
            labels = flip_bit(labels, label_noise)

            color_noise = (torch.rand(len(labels)) < self.color_noise[cid]).float()
            colors = flip_bit(labels, color_noise)

            # Add color to the samples
            images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0

            self.sample[self.control==cid] = images
            self.target[self.control==cid] = labels

    def generate_splits(self):
        """Create the splits in sample, targets and controls
        """

        sm_train = self.sample[self.split_idx == 0] # 0 for train
        sm_valid = self.sample[self.split_idx == 1] # 1 for valid
        sm_test = self.sample[self.split_idx == 2] # 2 for test

        y_train = self.target[self.split_idx == 0] # 0 for train
        y_valid = self.target[self.split_idx == 1] # 1 for valid
        y_test = self.target[self.split_idx == 2] # 2 for test

        c_train = self.control[self.split_idx == 0] # 0 for train
        c_valid = self.control[self.split_idx == 1] # 1 for valid
        c_test = self.control[self.split_idx == 2] # 2 for test    
    
        # SSL Setting
        if self.lab_split < 1.0:
            #TODO: check uniform sampling
            np.random.seed(self.dataseed)
            select = np.random.choice([False, True], size=len(c_train),\
            replace=True, p = [self.lab_split, 1-self.lab_split])
            c_train[select] = DF_M # DF_M denotes that the label is not available      
    
        return (sm_train, sm_valid, sm_test,\
                y_train, y_valid, y_test,\
                c_train, c_valid, c_test)

    def load_dataset(self,
                     batch_size=64,
                     num_batch_per_epoch=None,
                     num_workers=4,
                     pin_memory=False,
                     **kwargs):
        """Loads dataset.
    
        Args:
        batch_size: integer for batch size.
        num_batch_per_epoch: integer for number of batch per epoch.
        num_workers: number of workers for data loader.
        **kwargs: for backward compatibility.

        Returns:
        list of data loaders.
        """

        del kwargs

        # Generate DataLoaders
        if self.reweight:
            self.c_train = torch.LongTensor(self.c_train)
            c_counts = (torch.arange(self.n_controls).unsqueeze(1)==self.c_train).sum(1).float()
            c_invprobs = len(self.train_set) / c_counts
            invprobs = c_invprobs[self.c_train]

            sampler_train = WeightedRandomSampler(invprobs, len(self.train_set), replacement=True)
            shuffle_train = False
        else:
            sampler_train = None
            shuffle_train = self.shuffle_train

        # Full-batch training, adjust the hyper-param in the bin file accordingly
        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=shuffle_train,
                                                   sampler=sampler_train, drop_last=False)
        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                 batch_size=len(self.val_set),
                                                 num_workers=num_workers,
                                                 shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(self.test_set,
                                                  batch_size=len(self.test_set),
                                                  num_workers=num_workers,
                                                  shuffle=False, drop_last=False)
        
        return [train_loader, val_loader, test_loader]
    


