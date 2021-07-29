"""Tabular data loader."""

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data import data_util

from util.utils import DEFAULT_MISSING_CONST as DF_M

import pdb
    
class Tabular(object):
  """Tabular data loader."""

  def __init__(self, dataset_name='Adult', lab_split = 1.0, seed = 42):
    print('Using ', dataset_name)
    self.dataset_name = dataset_name
    self.dataseed = seed
    self.lab_split = lab_split
    
    self.load_raw_data(dataset_name=dataset_name)

    # Create Torch Custom Datasets
    self.train_set = data_util.ArrayFromMemory(data=self.x_train, \
                                          target=self.y_train, \
                                          control=self.c_train)
    self.val_set = data_util.ArrayFromMemory(data=self.x_valid, \
                                          target=self.y_valid, \
                                          control=self.c_valid)
    self.test_set = data_util.ArrayFromMemory(data=self.x_test, \
                                          target=self.y_test, \
                                          control=self.c_test)

    # Class counts
    self.n_targets = len(np.unique(self.y_train))
    self.n_controls = len(np.unique(self.c_train[self.c_train != DF_M])) # ignoring unavailable labels
    
  def load_raw_data(self, dataset_name='Adult'):
    """Load raw tabular data.
    """

    # Download data if unavailable
    data_util.maybe_download(adult_flag=True, german_flag=True)

    # Load the dataset
    #TODO: Single function for process tabular data
    
    if dataset_name == 'Adult':
      
      train_data, train_target, train_control,\
        valid_data, valid_target, valid_control,\
        test_data, test_target, test_control = \
          data_util.process_adult_data()
      
    elif dataset_name == 'German':
      
      train_data, train_target, train_control,\
        valid_data, valid_target, valid_control,\
        test_data, test_target, test_control = \
          data_util.process_german_data()
      
    elif dataset_name == 'AdultConfounded':
      train_data, train_target, train_control,\
        valid_data, valid_target, valid_control,\
        test_data, test_target, test_control = \
          data_util.process_adultconf_data()

      # 'race' \times 'sex' is control group
      # Each of 'race' and 'sex' is binary
      # 3:Black-Female, 2:Black-Male, 1:NonBlack-Female, 0:NonBlack-Male
      n_controls = 4
      train_control = train_control[:, 0]*2  + train_control[:, 1]
      valid_control = valid_control[:, 0]*2  + valid_control[:, 1]
      test_control = test_control[:, 0]*2 + test_control[:, 1]      

      # Stats prior to sampling
      print('Pr(y=1/g) before resampling')
      for cid in range(n_controls):
        for prefix in ['train', 'valid', 'test']:
          print(f'{prefix}-control{cid}',
                eval(f'sum(({prefix}_target.squeeze(-1) == 1) & ({prefix}_control == {cid})) / sum({prefix}_control == {cid})' ))
          
      train_data, train_target, train_control = data_util.resample(train_data, train_target, train_control,
                                                         n_controls=4, seed=self.dataseed,
                                                         probs=[0.94, 0.06, 0.94, 0.06])
      
          
    else:
      raise NotImplementedError

    self.x_train = train_data
    self.y_train = train_target
    self.c_train = train_control

    self.x_valid = valid_data
    self.y_valid = valid_target 
    self.c_valid = valid_control

    self.x_test = test_data
    self.y_test = test_target
    self.c_test = test_control
    
    # SSL Setting
    if self.lab_split < 1.0:
      np.random.seed(self.dataseed)
      select = np.random.choice([False, True], size=len(self.c_train),\
                       replace=True, p = [self.lab_split, 1-self.lab_split])
      self.c_train[select] = DF_M # DF_M denotes that the label is not available      
      
    
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

    # Semi-supervised Setting
    # TODO(lokhandevishnu:)

    return [train_loader, valid_loader, test_loader]
    
