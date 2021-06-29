"""Tabular data loader."""

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data import data_util
    
class Tabular(object):
  """Tabular data loader."""

  def __init__(self, data_name='Adult'):
    self.data_name = data_name
    self.load_raw_data(data_name=data_name)

  def load_raw_data(self, data_name='Adult'):
    """Load raw tabular data.
    """

    # Download data if unavailable
    data_util.maybe_download(adult_flag = True, german_flag = True)

    # Load the dataset
    if data_name == 'Adult':
      train_data, train_target, train_control,\
        valid_data, valid_target, valid_control,\
        test_data, test_target, test_control = \
          data_util.process_adult_data()
      
    elif data_name == 'German':
      train_data, train_target, train_control,\
        valid_data, valid_target, valid_control,\
        test_data, test_target, test_control = \
          data_util.process_german_data()
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

    # Create Torch Custom Datasets
    train_set = data_util.ArrayFromMemory(data=self.x_train, \
                                          target=self.y_train, \
                                          control=self.c_train)
    valid_set = data_util.ArrayFromMemory(data=self.x_valid, \
                                          target=self.y_valid, \
                                          control=self.c_valid)
    test_set = data_util.ArrayFromMemory(data=self.x_test, \
                                          target=self.y_test, \
                                          control=self.c_test)

    # Generate DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle= True)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=batch_size,
                                               shuffle= False)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=batch_size,
                                               shuffle= False)

    # Semi-supervised Setting
    # TODO(lokhandevishnu:)

    return [train_loader, valid_loader, test_loader]
    
