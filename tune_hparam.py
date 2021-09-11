"""Tuning Hyper-parameters from a given directory"""

from absl import app
from absl import flags
from numpy import genfromtxt
import numpy as np

from util.utils import HParams

import os
import pdb
import csv

flags.DEFINE_string(name='directory', default='results_all/Adult2_GroupDRO/', help='The directory to traverse')
flags.DEFINE_integer(name='min_c', default=2, help='index of the minority group')
flags.DEFINE_integer(name='nvp_th', default=10, help='chooses best worstoff accuracy from top nvp_th accuracies')
flags.DEFINE_integer(name='nvp_percent', default=95, help='chooses best worstoff accuracy from top nvp_th accuracies')

FLAGS = flags.FLAGS

def main(unused_argv):
    
    # Set parameters.
    hp = HParams({
            flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')
        })

    # Go through each sub-directory. Collect the last epoch val acc and test acc. Collect -1 and min_c accs.
    val_array = np.zeros((1, 2))
    test_array = np.zeros((1, 2))
    subdir_array = []
    for subdir in os.listdir(hp.directory):
        if subdir.endswith(".txt"):
            continue
        csv_path = os.path.join(hp.directory, subdir, 'run_44', 'stats', 'stats.csv')
        data = np.genfromtxt(csv_path, dtype=None, delimiter=',', names=True, deletechars="") 

        if len(data) < 1:
            print(f'{subdir} did not finish run')
            continue
        # Place the inside a numpy nd array and keep appending the numpy nd array.
        val_array = np.append(val_array, values=[[data[f'val.acc.-1'][-1], data[f'val.acc.{hp.min_c}'][-1]]], axis=0)
        test_array = np.append(test_array, values=[[data[f'test.acc.-1'][-1], data[f'test.acc.{hp.min_c}'][-1]]], axis=0)
        subdir_array.append(subdir)
    val_array = np.delete(val_array, (0), axis=0)
    test_array = np.delete(test_array, (0), axis=0)    
    print(val_array.shape, test_array.shape)

    # Assign index
    acc_idx = 0
    min_c_idx = 1

    # Perform a group-sort as per the validation accuracy
    sort_idx = val_array[:, acc_idx].argsort()[::-1]
    val_array_sorted = val_array[sort_idx]
    test_array_sorted = test_array[sort_idx]
    subdir_array_sorted = [subdir_array[idx] for idx in sort_idx]


    # Among top hp.nvp_th accuracies, pick the highest min_c accuracy. Pick corresponding test and test_min_c accuracy    
    select_idx = val_array_sorted[:hp.nvp_th, min_c_idx].argmax()
    select_val = val_array_sorted[select_idx]
    select_test = test_array_sorted[select_idx]
    select_subdir = subdir_array_sorted[select_idx]


    '''
    # Among accuracies>hp.nvp_percent  of best accuracy, pick the highest min_c accuracy.
    # Pick corresponding test and test_min_c accuracy
    nvp_acc = 0.01*hp.nvp_percent*val_array_sorted[0, acc_idx]
    nvp_choose = val_array_sorted[:, acc_idx] > nvp_acc
    print('selects:', sum(nvp_choose))
    select_idx = val_array_sorted[nvp_choose, min_c_idx].argmax() # since sorted array, index not corrupted
    select_val = val_array_sorted[select_idx]
    select_test = test_array_sorted[select_idx]
    select_subdir = subdir_array_sorted[select_idx]
    '''
    
    # Print the result
    string_to_print = f'File: {select_subdir}\n'
    string_to_print += f'VALIDATION-- Accuracy: {select_val[acc_idx]:.4f} |||| MinAccuracy: {select_val[min_c_idx]:.4f}\n'
    string_to_print += f'TESTING-- Accuracy: {select_test[acc_idx]:.4f} |||| MinAccuracy: {select_test[min_c_idx]:.4f}\n'
    print(string_to_print)
    
    # log the result in tune.txt
    with open(os.path.join(hp.directory, 'tune.txt'), 'w') as f:
        f.write(string_to_print)

if __name__ == '__main__':
    app.run(main)

