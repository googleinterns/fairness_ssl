from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np
import pandas as pd
import joblib
  
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
  """ Process the entries of Adult dataset. Creates the training, 
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
    os.path.join(DATA_DIRECTORY + "raw","adult.data"),\
    delimiter=", ", header=None, names=ADULT_ALL_COL_NAMES,
    na_values="?",keep_default_na=False)
  test_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw","adult.test"),\
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

  split_numbers = (train_data.shape[1],train_data.shape[1] + 1)
  train_size = train_data.shape[0]

  # Normalize the Training dataset
  maxes = np.maximum(np.maximum(\
    train_data.max(axis=0),\
    val_data.max(axis=0)),\
    test_data.max(axis=0))

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  train_data = np.concatenate(\
    (train_data.values, train_target.values, train_control.values),\
    axis=1\
  )

  return train_data, split_numbers, train_size,\
    val_data.values, val_target.values, val_control.values,\
    test_data.values, test_target.values, test_control.values


def process_german_data():
  """ Process the entries of German dataset. Creates the training, 
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
    os.path.join(DATA_DIRECTORY + "raw","german.data"),\
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

  split_numbers = (train_data.shape[1],train_data.shape[1] + 1)
  train_size = train_data.shape[0]

  #data normalization
  maxes = np.maximum(np.maximum(\
    train_data.max(axis=0),\
    val_data.max(axis=0)),\
    test_data.max(axis=0))

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  train_data = np.concatenate(\
    (train_data.values, train_target.values, train_control.values),\
    axis=1\
  )

  return train_data, split_numbers, train_size,\
    val_data.values, val_target.values, val_control.values,\
    test_data.values, test_target.values, test_control.values


if __name__ == "__main__":
  maybe_download(adult_flag = True, german_flag = True)

  # Prepare Save Directory
  if not os.path.exists(DATA_DIRECTORY + "processed"):
    os.makedirs(DATA_DIRECTORY + "processed")
   
  # Save processed Adult Dataset
  train_data, split_numbers, train_size,\
    validation_data, validation_target, validation_control,\
    test_data, test_target, test_control = \
      process_adult_data()


  joblib.dump(
    value = ( train_data, split_numbers, train_size,\
              validation_data, validation_target, validation_control,\
              test_data, test_target, test_control ),\
    filename = os.path.join(DATA_DIRECTORY + "processed", "adult_processed.z"), \
    compress = ("zlib",1)\
  )
    
  print("adult test control split %f" % (np.sum(test_control)/test_control.shape[0]))
  print("adult test target split %f" % (np.sum(test_target)/test_target.shape[0]))

  # Save processed German Dataset
  train_data, split_numbers, train_size,\
    validation_data, validation_target, validation_control,\
    test_data, test_target, test_control = \
    process_german_data()

  joblib.dump(
    value = ( train_data, split_numbers, train_size,\
              validation_data, validation_target, validation_control,\
              test_data, test_target, test_control ),\
    filename = os.path.join(DATA_DIRECTORY + "processed", "german_processed.z"), \
    compress = ("zlib",1)\
  )

  print("german test control split %f" % (np.sum(test_control)/test_control.shape[0]))
  print("german test target split %f" % (np.sum(test_target)/test_target.shape[0]))
