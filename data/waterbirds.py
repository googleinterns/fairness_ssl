"""Waterbirds data loader."""

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms

from data import data_util

from util.utils import DEFAULT_MISSING_CONST as DF_M

import pdb

DATA_DIRECTORY = 'data/datasets/waterbirds_dataset/'
    
class Waterbirds(object):
    """Waterbirds data loader."""

    def __init__(self, lab_split=1.0, reweight=False, shuffle_train=True, seed=42, get_dataset_from_lmdb=False):
        print('Using Waterbirds dataset!')

        self.data_dir = DATA_DIRECTORY
        self.dataseed = seed
        self.lab_split = lab_split
        self.reweight = reweight
        self.shuffle_train = shuffle_train
        self.get_dataset_from_lmdb = get_dataset_from_lmdb
      
        if not os.path.exists(self.data_dir):
            raise ValueError(f'{self.data_dir} does not exist yet.')

        # Read metadata files
        self.metadata = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))

        # Target values
        self.target = self.metadata['y'].values
        self.n_targets = len(np.unique(self.target))

        # Control values
        self.environment = self.metadata['place'].values
        self.n_envs = len(np.unique(self.environment))
    
        # Generate control groups
        # Change the formula for semi-sup setting
        self.n_controls = self.n_targets * self.n_envs # Each target x env counts as one group
        self.control = (self.target*(self.n_controls/2) + self.environment).astype('int')
        assert self.n_controls == len(np.unique(self.control)), "Error in control list"

        # Marginal count from data=0.53,0.25,0.07,0.15
        
        # Extract filenames and splits
        self.filename = self.metadata['img_filename'].values
        self.split_idx = self.metadata['split'].values

        # Split train, valid, test
        (fn_train, fn_valid, fn_test,
            y_train, y_valid, y_test,
            self.c_train, c_valid, c_test) = self.generate_splits()

        # Get custom transforms
        train_transform, eval_transform = self.get_transforms()

        # Create Torch Custom Datasets
        if self.get_dataset_from_lmdb:
            self.alldata_set = data_util.ImageFromLMDB(filename=self.filename,
                                                       target=self.target,
                                                       control=self.control,
                                                       data_dir=self.data_dir,
                                                       transform=eval_transform)
        else:
            self.alldata_set = data_util.ImageFromDisk(filename=self.filename,
                                                       target=self.target,
                                                       control=self.control,
                                                       data_dir=self.data_dir,
                                                       transform=eval_transform)
        self.train_set = torch.utils.data.Subset(self.alldata_set,
                                                 np.where(self.split_idx==0)[0])
        self.train_set.dataset.transform = train_transform
        self.val_set = torch.utils.data.Subset(self.alldata_set,
                                               np.where(self.split_idx==1)[0])
        self.test_set = torch.utils.data.Subset(self.alldata_set,
                                                np.where(self.split_idx==2)[0])

    def generate_splits(self):
        """Create the splits in filename, targets and controls."""

        fn_train = self.filename[self.split_idx == 0] # 0 for train
        fn_valid = self.filename[self.split_idx == 1] # 1 for valid
        fn_test = self.filename[self.split_idx == 2] # 2 for test

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
            select = np.random.choice([False, True],
                                      size=len(c_train),
                                      replace=True,
                                      p=[self.lab_split, 1-self.lab_split])
            # DF_M denotes that the label is not available for train data.
            c_train[select] = DF_M
            self.control[np.where(self.split_idx == 0)[0][select]] = DF_M

        return (fn_train, fn_valid, fn_test,\
                y_train, y_valid, y_test,\
                c_train, c_valid, c_test)

    def get_transforms(self):
        """Gets transformation."""
        scale = 256.0/224.0
        target_resolution = (224, 224)

        train_transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, eval_transform    

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
    
        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=shuffle_train,
                                                   sampler=sampler_train, drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(self.test_set,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  shuffle=False, drop_last=False)
        
        return [train_loader, val_loader, test_loader]
    

if __name__ == '__main__':
    # Compare LMDB-based loader and Disk-based loader
    db = Waterbirds(lab_split=0.3, reweight=False, seed=42, get_dataset_from_lmdb=False)
    loaders_disk = db.load_dataset(batch_size=64)
    db = Waterbirds(lab_split=0.3, reweight=False, seed=42, get_dataset_from_lmdb=True)
    loaders_lmdb = db.load_dataset(batch_size=64)
    for batch_disk, batch_lmdb in zip(loaders_disk[1], loaders_lmdb[1]):
        assert torch.norm(batch_disk[0] - batch_lmdb[0]) == 0, 'data are aligned'
