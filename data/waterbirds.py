"""Waterbirds data loader."""

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data import data_util

from util.utils import DEFAULT_MISSING_CONST as DF_M

import pdb

DATA_DIRECTORY = 'data/datasets/waterbirds_dataset/'
    
class Waterbirds(object):
  """Waterbirds data loader."""

  def __init__(self, lab_split = 1.0, seed = 42):
    print('Using Waterbirds dataset!')

    self.data_dir = DATA_DIRECTORY
    self.dataseed = seed
    self.lab_split = lab_split
      
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

    # Extract filenames and splits
    self.filename = self.metadata['img_filename'].values
    self.split_idx = self.metadata['split'].values

    # Split train, valid, test
    fn_train, fn_valid, fn_test, \
        y_train, y_valid, y_test, \
        c_train, c_valid, c_test = self.generate_splits()

    # Create Torch Custom Datasets
    self.train_set = data_util.ImageFromMemory(filename=fn_train, \
                                               target=y_train, \
                                               control=c_train,
                                               data_dir=self.data_dir)
    self.val_set = data_util.ImageFromMemory(filename=fn_valid, \
                                             target=y_valid, \
                                             control=c_valid,
                                             data_dir=self.data_dir)                  
    self.test_set = data_util.ImageFromMemory(filename=fn_test, \
                                              target=y_test, \
                                              control=c_test,
                                              data_dir=self.data_dir)
    pdb.set_trace()
    
  def generate_splits(self):
    """Create the splits in filename, targets and controls
    """

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
        # TODO: Write logic for SSL setting
        '''
        np.random.seed(self.dataseed)
        select = np.random.choice([False, True], size=len(self.c_train),\
                       replace=True, p = [self.lab_split, 1-self.lab_split])
        self.c_train[select] = DF_M # DF_M denotes that the label is not available      
        '''
        pass
    
    return fn_train, fn_valid, fn_test, y_train, y_valid, y_test, c_train, c_valid, c_test
        
        
    

    
    
  def load_dataset(self,
                   batch_size=64,
                   num_batch_per_epoch=None,
                   num_workers=4,
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
    train_loader = torch.utils.data.DataLoader(self.train_set,
                                               batch_size=batch_size,
                                               shuffle= True)
    valid_loader = torch.utils.data.DataLoader(self.val_set,
                                               batch_size=batch_size,
                                               shuffle= False)
    test_loader = torch.utils.data.DataLoader(self.test_set,
                                               batch_size=batch_size,
                                               shuffle= False)

    return [train_loader, valid_loader, test_loader]
    


