# Non-torch methods

import math
import pdb
import csv
import os

from absl import logging
from google.cloud import storage

DEFAULT_MISSING_CONST = -42

def flip_bit(a, b):
    return (a-b).abs()


def remove_first_string_from_string(s, d):
    return d.join(x for x in s.split(d) if x)


class HParams(dict):
    """Custom dictionary that allows to access dict values as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.valsum = 0.0
        self.count = 0.0

    def update(self, val, num=1):
        # nan implies division by zero
        # we skip nans here
        if math.isnan(val):
            return
        self.val = val
        self.valsum += val * num
        self.count += num
        self.avg = self.valsum / self.count

    def get_avg(self):
        return self.avg

    def get_current_val(self):
        return self.val


class CSVLogger(object):
    def __init__(self, log_dir, mode='w'):
        self.csv_path = os.path.join(log_dir, 'stats.csv')
        self.mode = mode
        self.csv_file = open(self.csv_path, self.mode)
        self.columns = []
        self.stats_dict = {}
        self.writer = None
        
    def set_header(self, n_controls=4):
        self.columns.append('epoch')

        for p in ['train', 'val', 'test']:
            for cid in range(-1, n_controls):
                m = 'loss'
                self.columns.append(f'{p}.{m}.{cid}')
                
                m = 'acc'
                self.columns.append(f'{p}.{m}.{cid}')
                
                m = 'auc'
                self.columns.append(f'{p}.{m}.{cid}')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.columns)
        if self.mode=='w':
            self.writer.writeheader()

    def init_stats_dict(self, epoch):
        self.stats_dict = {}
        self.stats_dict['epoch'] = epoch

    def add_item_dict(self, key, value):
        self.stats_dict[key] = value

    def write_stats_dict(self):
        self.writer.writerow(self.stats_dict)

    def flush(self):
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def upload(upload_dir, gcs_bucket, output_dir):
    """Upload files from a directory to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket)
    for dirpath, _, filenames in os.walk(upload_dir):
        for name in filenames:
            filename = os.path.join(dirpath, name)
            blob = storage.Blob(os.path.join(output_dir, name), bucket)
            with open(filename, 'rb') as f:
                blob.upload_from_file(f)
            logging.info('blob path: %s', blob.path)
            logging.info('bucket path: gs://%s/%s', gcs_bucket, output_dir)
