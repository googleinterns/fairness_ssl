from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

from PIL import Image
import numpy as np
import pandas as pd
import joblib
import lmdb
from tqdm import trange

import torch
from torchvision.datasets.folder import is_image_file

import six
from six.moves import urllib
import pdb


ADULT_SOURCE_URL =\
  'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
GERMAN_SOURCE_URL =\
  'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/'

ADULT_VALIDATION_SPLIT = 0.5
GERMAN_VAL_SPLIT = 0.2
GERMAN_TEST_SPLIT = 0.2

DATA_DIRECTORY = 'data/datasets/'

def maybe_download(adult_flag=False, german_flag=False):
  if not adult_flag and not german_flag:
    raise("Neither flag specified, aborting.")

  if adult_flag:
    maybe_download_adult("adult.data")
    maybe_download_adult("adult.test")

  if german_flag:
    maybe_download_german("german.data")

# Download Adult dataset
def maybe_download_adult(filename):

  if not os.path.exists(DATA_DIRECTORY + "raw"):
    os.makedirs(DATA_DIRECTORY + "raw")

  filepath = os.path.join(DATA_DIRECTORY + "raw", filename)

  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(ADULT_SOURCE_URL + filename, filepath)
    size = os.path.getsize(filepath)
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

# Download German Dataset
def maybe_download_german(filename):

  if not os.path.exists(DATA_DIRECTORY + "raw"):
    os.makedirs(DATA_DIRECTORY + "raw")

  filepath = os.path.join(DATA_DIRECTORY + "raw", filename)

  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(GERMAN_SOURCE_URL + filename, filepath)
    size = os.path.getsize(filepath)
    print('Successfully downloaded', filename, size, 'bytes.')

  return filepath


def process_adult_data():
  """Process the entries of Adult dataset. Creates the training, 
  validation and test splits. 

  Target Variable: income
  Control Variable: sex
  """

  # Load the dataset
  ADULT_ALL_COL_NAMES =  ["age", "workclass", "fnlwgt", "education", \
                          "education-num", "marital-status", "occupation",\
                          "relationship", "race", "sex", "capital-gain",\
                           "capital-loss", "hours-per-week", "native-country",\
                           "income" ]
  train_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw", "adult.data"),\
    delimiter=", ", header=None, names=ADULT_ALL_COL_NAMES,
    na_values="?",keep_default_na=False)
  test_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw", "adult.test"),\
    delimiter=", ", header=None, names=ADULT_ALL_COL_NAMES,
    na_values="?",keep_default_na=False, skiprows=1)

  # Drop empty entries
  train_data.dropna(inplace=True)
  test_data.dropna(inplace=True)

  # Binarize the attributes
  ADULT_SELECT_COL_INDEX =  [1,3,5,6,7,8,13]
  all_data = pd.concat([train_data,test_data])
  all_data = pd.get_dummies(all_data,\
    columns=[ADULT_ALL_COL_NAMES[i] for i in ADULT_SELECT_COL_INDEX])
  all_data.loc[all_data.income == ">50K","income"] = 1
  all_data.loc[all_data.income == ">50K.","income"] = 1
  all_data.loc[all_data.income == "<=50K","income"] = 0
  all_data.loc[all_data.income == "<=50K.","income"] = 0

  all_data.loc[all_data.sex == "Female","sex"] = 1
  all_data.loc[all_data.sex == "Male","sex"] = 0

  # Create Training and Test Splits
  cutoff = train_data.shape[0]
  train_data = all_data.loc[:cutoff,\
    (all_data.columns != "income") & (all_data.columns != "sex")]
  train_control = all_data.loc[:cutoff,all_data.columns == "sex"]
  train_target = all_data.loc[:cutoff,all_data.columns == "income"]

  test_data = all_data.loc[cutoff:,\
    (all_data.columns != "income") & (all_data.columns != "sex")]
  test_control = all_data.loc[cutoff:,all_data.columns == "sex"]
  test_target = all_data.loc[cutoff:,all_data.columns == "income"]

  # Filter invalid columns
  col_valid_in_train_data =\
     [len(train_data.loc[:,x].unique()) > 1 for x in train_data.columns]
  col_valid_in_test_data =\
     [len(test_data.loc[:,x].unique()) > 1 for x in test_data.columns]
  col_valid = list(map(lambda x,y: x and y, col_valid_in_train_data, col_valid_in_test_data))
  train_data = train_data.loc[:,col_valid]
  test_data = test_data.loc[:,col_valid]

  # Sample Validation dataset
  cutoff = int((1.0 - ADULT_VALIDATION_SPLIT) * train_data.shape[0])
  val_data = train_data.loc[cutoff:,:]
  train_data = train_data.loc[:cutoff,:]

  val_target = train_target.loc[cutoff:,:]
  train_target = train_target.loc[:cutoff,:]

  val_control = train_control.loc[cutoff:,:]
  train_control = train_control.loc[:cutoff,:]

  # Normalize the Training dataset
  maxes = train_data.max(axis=0)

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  return train_data.values, train_target.values, train_control.values,\
    val_data.values, val_target.values, val_control.values,\
    test_data.values, test_target.values, test_control.values


def process_adult_sex_race_data():
  """Process the entries of Adult dataset. Creates the training, 
  validation and test splits. 

  Target Variable: income
  Control Variable: sex
  """

  # Load the dataset
  ADULT_ALL_COL_NAMES =  {0: "age", 1: "workclass", 2: "fnlwgt", 3: "education", \
                          4: "education-num", 5: "marital-status", 6: "occupation",\
                          7: "relationship", 8: "race", 9: "sex", 10: "capital-gain",\
                          11:  "capital-loss", 12: "hours-per-week", 13: "native-country",\
                           14: "income" }
  train_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw", "adult.data"),\
    delimiter=", ", header=None, names=ADULT_ALL_COL_NAMES.values(),
    na_values="?",keep_default_na=False)
  test_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw", "adult.test"),\
    delimiter=", ", header=None, names=ADULT_ALL_COL_NAMES.values(),
    na_values="?",keep_default_na=False, skiprows=1)

  # Drop empty entries
  train_data.dropna(inplace=True)
  test_data.dropna(inplace=True)

  # Binarize the attributes
  ADULT_SELECT_COL_INDEX =  [1,3,5,6,7,13]
  all_data = pd.concat([train_data,test_data])
  all_data = pd.get_dummies(all_data,\
    columns=[ADULT_ALL_COL_NAMES[i] for i in ADULT_SELECT_COL_INDEX])
  
  all_data.loc[all_data.income == ">50K","income"] = 1
  all_data.loc[all_data.income == ">50K.","income"] = 1
  all_data.loc[all_data.income == "<=50K","income"] = 0
  all_data.loc[all_data.income == "<=50K.","income"] = 0

  all_data.loc[all_data.sex == "Female","sex"] = 1
  all_data.loc[all_data.sex == "Male","sex"] = 0

  all_data.loc[all_data.race != "Black","race"] = 0  
  all_data.loc[all_data.race == "Black","race"] = 1

  # Create Training and Test Splits
  cutoff = train_data.shape[0]
  train_data = all_data.loc[:cutoff, (all_data.columns != "income") &\
                            (all_data.columns != "sex") &\
                            (all_data.columns != "race")]
  train_control = all_data.loc[:cutoff, (all_data.columns == "sex") |\
                               (all_data.columns == "race")]
  train_target = all_data.loc[:cutoff,all_data.columns == "income"]

  test_data = all_data.loc[cutoff:, (all_data.columns != "income") &\
                           (all_data.columns != "sex") &\
                           (all_data.columns != "race")]
  test_control = all_data.loc[cutoff:, (all_data.columns == "sex") |\
                               (all_data.columns == "race")]
  test_target = all_data.loc[cutoff:,all_data.columns == "income"]
  
  # Filter invalid columns
  col_valid_in_train_data =\
     [len(train_data.loc[:,x].unique()) > 1 for x in train_data.columns]
  col_valid_in_test_data =\
     [len(test_data.loc[:,x].unique()) > 1 for x in test_data.columns]
  col_valid = list(map(lambda x,y: x and y, col_valid_in_train_data, col_valid_in_test_data))
  train_data = train_data.loc[:,col_valid]
  test_data = test_data.loc[:,col_valid]

  # Sample Validation dataset
  cutoff = int((1.0 - ADULT_VALIDATION_SPLIT) * train_data.shape[0])
  val_data = train_data.loc[cutoff:,:]
  train_data = train_data.loc[:cutoff,:]

  val_target = train_target.loc[cutoff:,:]
  train_target = train_target.loc[:cutoff,:]

  val_control = train_control.loc[cutoff:,:]
  train_control = train_control.loc[:cutoff,:]

  # Normalize the Training dataset
  maxes = train_data.max(axis=0)

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  return train_data.values, train_target.values, train_control.values,\
    val_data.values, val_target.values, val_control.values,\
    test_data.values, test_target.values, test_control.values

def process_german_data():
  """Process the entries of German dataset. Creates the training, 
  validation and test splits. 

  Target Variable: cr_good_bad
  Control Variable: sex
  """

  # Load the dataset
  GERMAN_ALL_COL_NAMES = ["checking_acc", "duration", "credit_hist", \
                          "purpose", "credit_amount", "savings", \
                          "employment_status", "install_rate", \
                          "relationship_and_sex", "debtors", "res_interval",\
                          "property", "age", "other_plans", "housing",\
                          "credits_at_bank", "job", "liable_persons", \
                          "phone", "foreign", "cr_good_bad"]
  all_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw", "german.data"),\
    delimiter=" ", header=None, names=GERMAN_ALL_COL_NAMES,
    na_values="?",keep_default_na=False)

  # Drop empty entries
  all_data.dropna(inplace=True)

  # Binarize Entries
  GERMAN_SELECT_COL_INDEX = [0,2,3,5,6,8,9,11,13,14,16,18,19]
  
  all_data = all_data.assign(sex=(all_data.relationship_and_sex == "A92").astype(int) | \
    (all_data.relationship_and_sex == "A95").astype(int))
  col_names = GERMAN_ALL_COL_NAMES +["sex"] 
  all_data.loc[:,all_data.columns == "cr_good_bad"] =\
    all_data.loc[:,all_data.columns == "cr_good_bad"] - 1
  all_data = pd.get_dummies(all_data,\
    columns=[col_names[i] for i in GERMAN_SELECT_COL_INDEX])

  # Create Train and Test Splits
  cutoff = int(all_data.shape[0] * (1.0 - GERMAN_TEST_SPLIT))
  train_data = all_data.loc[:cutoff,\
    (all_data.columns != "cr_good_bad") & (all_data.columns != "sex")]
  train_control = all_data.loc[:cutoff,all_data.columns == "sex"]
  train_target = all_data.loc[:cutoff,all_data.columns == "cr_good_bad"]

  test_data = all_data.loc[cutoff:,\
    (all_data.columns != "cr_good_bad") & (all_data.columns != "sex")]
  test_control = all_data.loc[cutoff:,all_data.columns == "sex"]
  test_target = all_data.loc[cutoff:,all_data.columns == "cr_good_bad"]

  # Filter Invalid Columns
  col_valid_in_train_data =\
     [len(train_data.loc[:,x].unique()) > 1 for x in train_data.columns]
  col_valid_in_test_data =\
     [len(test_data.loc[:,x].unique()) > 1 for x in test_data.columns]
  col_valid = list(map(lambda x,y: x and y, col_valid_in_train_data, col_valid_in_test_data))

  train_data = train_data.loc[:,col_valid]
  test_data = test_data.loc[:,col_valid]

  # Prepare  validation data
  cutoff = int((1.0 - GERMAN_VAL_SPLIT) * train_data.shape[0])
  val_data = train_data.loc[cutoff:,:]
  train_data = train_data.loc[:cutoff,:]

  val_target = train_target.loc[cutoff:,:]
  train_target = train_target.loc[:cutoff,:]

  val_control = train_control.loc[cutoff:,:]
  train_control = train_control.loc[:cutoff,:]

  #data normalization
  maxes = np.maximum(np.maximum(\
    train_data.max(axis=0),\
    val_data.max(axis=0)),\
    test_data.max(axis=0))

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  return train_data.values, train_target.values, train_control.values, \
    val_data.values, val_target.values, val_control.values,\
    test_data.values, test_target.values, test_control.values


class ArrayFromMemory(torch.utils.data.Dataset):
  """Creates a custom dataset for tabular data
  """

  def __init__(self, data, target, control):
    super(ArrayFromMemory, self).__init__()
    self.data = data
    self.target = target
    self.control = control
    
  def __len__(self):
    """torch.utils.Dataset function
       Returns:
       Size of the dataset
    """
    return self.data.shape[0]

  def __getitem__(self, index):
    """torch.utils.Dataset function
       Returns:
       Item at a specific index
    """
    x = self.data[index, :].astype(float)
    y = int(self.target[index])
    c = int(self.control[index])
    return torch.tensor(x).float(), torch.tensor(y).long(), torch.tensor(c).long()

class ImageFromDisk(torch.utils.data.Dataset):
  """Creates a custom dataset for imaging data
  """
  
  def __init__(self, filename, target, control, data_dir, transform=None):
    self.filename = filename
    self.target = target
    self.control = control
    self.data_dir = data_dir
    self.transform = transform
    
  def __len__(self):
    return self.filename.shape[0]

  def __getitem__(self, idx):
    y = self.target[idx]
    c = self.control[idx]

    img_filename = os.path.join(self.data_dir, self.filename[idx])
    img = Image.open(img_filename).convert('RGB')

    if self.transform is not None:
      img = self.transform(img)
    
    x = img

    return x, torch.tensor(y).long(), torch.tensor(c).long()

class ImageFromMemory(torch.utils.data.Dataset):
  """Creates a custom dataset for imaging data
  """
  
  def __init__(self, sample, target, control, transform=None):
    self.sample = sample
    self.target = target
    self.control = control
    self.transform = transform
    
  def __len__(self):
    return self.sample.shape[0]

  def __getitem__(self, idx):
    x = self.sample[idx]
    y = self.target[idx]
    c = self.control[idx]
    
    if self.transform is not None:
      x = self.transform(x)

    return x, y.long(), torch.tensor(c).long()
  
def resample(data, target, control, n_controls = 4, seed=42, probs = [0.94, 0.06, 0.94, 0.06]):
  for cid in range(n_controls):
    indices_sub = control == cid
    data_sub = data[indices_sub]
    target_sub = target[indices_sub]
    control_sub = control[indices_sub]

    count_cid_old = np.bincount(target_sub.squeeze(-1).astype(int)).astype(float)
    probs_cid_old = count_cid_old / count_cid_old.sum()
    probs_cid_new = np.array([1-probs[cid], probs[cid]])

    sampling_prob = probs_cid_new[target_sub.squeeze(-1).astype(int)].astype(float) \
      / probs_cid_old[target_sub.squeeze(-1).astype(int)].astype(float)
    sampling_prob = sampling_prob / sampling_prob.sum()
    sampling_indices = np.random.choice(np.arange(len(data_sub)), size=len(data_sub), p=sampling_prob)
    data[indices_sub] = data_sub[sampling_indices]
    target[indices_sub] = target_sub[sampling_indices]
    control[indices_sub] = control_sub[sampling_indices]    

  return data, target, control
      
  
class ImageFromLMDB(torch.utils.data.Dataset):
  """Creates a custom dataset for image data by reading from LMDB format."""

  def __init__(self, filename, target, control, data_dir, mdb_path=None, transform=None):
    super(ImageFromLMDB, self).__init__()
    mdb_name, txt_name = self.create_lmdb_from_filename(data_dir, filename, mdb_path=mdb_path)
    self.env = lmdb.open(mdb_name,
                         max_readers=1,
                         readonly=True,
                         lock=False,
                         readahead=False,
                         meminit=False)
    with self.env.begin(write=False) as txn:
      self.length = txn.stat()['entries']
    self.data_dir = data_dir
    self.sample_key = [line.split()[0].encode() for line in open(txt_name, 'r').readlines()]
    self.target = target
    self.control = control
    self.transform = transform

  def is_valid(self, data_dir, line):
    return True if is_image_file(os.path.join(data_dir, line.split()[0])) else False

  def read_image_list(self, data_dir, filename):
    return [(os.path.join(data_dir, f), f) for f in filename if self.is_valid(data_dir, f)]

  def create_lmdb(self, imlist, mdb_name):
    env = lmdb.open(mdb_name, map_size=1000*2**30)
    with env.begin(write=True) as txn:
      for i in trange(len(imlist)):
        path, key = imlist[i]
        with open(path, 'rb') as fd:
          rawBytes = fd.read()
          txn.put(key.encode(), rawBytes)

  def create_lmdb_from_filename(self, data_dir, filename, mdb_path=None):
    if mdb_path is None:
      mdb_path = os.path.join(data_dir, 'lmdb')
    mdb_name = os.path.join(mdb_path, 'data.mdb')
    txt_name = os.path.join(mdb_path, 'data.txt')

    if not os.path.exists(mdb_path):
      os.makedirs(mdb_path)

    if os.path.exists(mdb_name):
      return mdb_name, txt_name

    # Generate (fpath, label) pairs by reading from txt file.
    imlist = self.read_image_list(data_dir, filename)
    self.create_lmdb(imlist, mdb_name)

    # Write new text file that includes only files used for creating LMDB.
    with open(txt_name, 'w') as g:
      g.writelines([f'{imlist[i][1]}\n' for i in range(len(imlist))])
    return mdb_name, txt_name

  def __len__(self):
    return len(self.sample_key)

  def __getitem__(self, idx):
    """Gets items.

    Get image from LMDB and target and control variables from input.
    """
    env = self.env
    key = self.sample_key[idx]
    with env.begin(write=False) as txn:
      imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)

    x = Image.open(buf).convert('RGB')
    y = self.target[idx]
    c = self.control[idx]

    if self.transform is not None:
      x = self.transform(x)

    return x, torch.tensor(y).long(), torch.tensor(c).long()