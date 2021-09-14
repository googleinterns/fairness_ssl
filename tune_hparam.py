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
flags.DEFINE_list(name='seed_list', default=['42', '43', '44'],
                  help='Marginal probabilities for each group.')
flags.DEFINE_integer(name='min_c', default=2, help='index of the minority group')
flags.DEFINE_integer(name='nvp_th', default=5, help='chooses best worstoff accuracy from top nvp_th accuracies')
flags.DEFINE_integer(name='nvp_percent', default=98, help='chooses best worstoff accuracy from top nvp_th accuracies')

FLAGS = flags.FLAGS

def main(unused_argv):
    
    # Set parameters.
    hp = HParams({
            flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')
        })

    # Go through each sub-directory. Collect the last epoch val acc and test acc. Collect -1 and min_c accs.
    val_mean = np.zeros((1, 2)); val_std = np.zeros((1, 2));
    test_mean = np.zeros((1, 2)); test_std = np.zeros((1, 2));
    subdir_array = []
    for subdir in os.listdir(hp.directory):
        if subdir.endswith(".txt"):
            continue

        val_item_all = []; val_item_min = [];
        test_item_all = []; test_item_min = [];
        for seed in hp.seed_list:
            csv_path = os.path.join(hp.directory, subdir, f'run_{seed}', 'stats', 'stats.csv')
            
            try:
                data = np.genfromtxt(csv_path, dtype=None, delimiter=',', names=True, deletechars="") 
            except OSError:
                #print(f'{csv_path} not found')
                continue

            if len(data) < 1:
                print(f'{subdir} did not finish run')
                continue

            # Picking the last epoch measurements
            val_item_all.append(data[f'val.acc.-1'][-1]); val_item_min.append(data[f'val.acc.{hp.min_c}'][-1]);
            test_item_all.append(data[f'test.acc.-1'][-1]); test_item_min.append(data[f'test.acc.{hp.min_c}'][-1]);            

        # Place inside a numpy nd array and keep appending the numpy nd array.
        val_mean = np.append(val_mean, values=[[np.mean(val_item_all), np.mean(val_item_min)]], axis=0);
        val_std = np.append(val_std, values=[[np.std(val_item_all), np.std(val_item_min)]], axis=0);         
        test_mean = np.append(test_mean, values=[[np.mean(test_item_all), np.mean(test_item_min)]], axis=0);
        test_std = np.append(test_std, values=[[np.std(test_item_all), np.std(test_item_min)]], axis=0);         
        subdir_array.append(subdir)

    val_mean = np.delete(val_mean, (0), axis=0); val_std = np.delete(val_std, (0), axis=0);
    test_mean = np.delete(test_mean, (0), axis=0); test_std = np.delete(test_std, (0), axis=0);
    print(val_mean.shape, val_std.shape,  test_mean.shape, test_std.shape)
    
    # Assign index
    acc_idx = 0
    min_c_idx = 1

    # Perform a group-sort as per the validation accuracy
    sort_idx = val_mean[:, acc_idx].argsort()[::-1]
    val_mean_sorted = val_mean[sort_idx]
    val_std_sorted = val_std[sort_idx]    
    test_mean_sorted = test_mean[sort_idx]
    test_std_sorted = test_std[sort_idx]    
    subdir_array_sorted = [subdir_array[idx] for idx in sort_idx]
    
    # Among top hp.nvp_th accuracies, pick the highest min_c accuracy. Pick corresponding test and test_min_c accuracy    
    select_idx = val_mean_sorted[:hp.nvp_th, min_c_idx].argmax()
    select_val_mean = val_mean_sorted[select_idx]
    select_val_std = val_std_sorted[select_idx]    
    select_test_mean = test_mean_sorted[select_idx]
    select_test_std = test_std_sorted[select_idx]    
    select_subdir = subdir_array_sorted[select_idx]

    '''
    # Among accuracies>hp.nvp_percent  of best accuracy, pick the highest min_c accuracy.
    # Pick corresponding test and test_min_c accuracy
    nvp_acc = 0.01*hp.nvp_percent*val_mean_sorted[0, acc_idx]
    nvp_choose = val_mean_sorted[:, acc_idx] > nvp_acc
    print('selects:', sum(nvp_choose))
    select_idx = val_mean_sorted[nvp_choose, min_c_idx].argmax() # since sorted array, index not corrupted
    select_val_mean = val_mean_sorted[select_idx]
    select_val_std = val_std_sorted[select_idx]    
    select_test_mean = test_mean_sorted[select_idx]
    select_test_std = test_std_sorted[select_idx]    
    select_subdir = subdir_array_sorted[select_idx]
    '''
    
    # Print the result
    string_to_print = f'File: {select_subdir}\n'
    string_to_print += f'VALIDATION-- Accuracy: {select_val_mean[acc_idx]:.4f} +- {select_val_std[acc_idx]:.4f} ||||          MinAccuracy: {select_val_mean[min_c_idx]:.4f} +- {select_val_std[min_c_idx]:.4f}\n'
    string_to_print += f'TESTING-- Accuracy: {select_test_mean[acc_idx]:.4f} +- {select_test_std[acc_idx]:.4f} ||||          MinAccuracy: {select_test_mean[min_c_idx]:.4f} +- {select_test_std[min_c_idx]:.4f}\n'
    print(string_to_print)
    
    # log the result in tune.txt
    with open(os.path.join(hp.directory, 'tune.txt'), 'w') as f:
        f.write(string_to_print)

if __name__ == '__main__':
    app.run(main)

