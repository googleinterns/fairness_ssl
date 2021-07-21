"""Base trainer."""

import os
import time
from tqdm import tqdm
import json
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

import pdb

from data.tabular import Tabular
from data.waterbirds import Waterbirds

from model.fullyconn import FullyConnected
from torchvision.models import resnet50 as ResNet50

from util.utils import HParams, AverageMeter

from util.metrics_store import MetricsEval

import tensorflow as tf

class BaseTrain(object):
    """Base model trainer."""

    def __init__(self, hparams):
        self.hp = hparams
        self.dataset_name = hparams.dataset

        # Set file path to save.
        self.get_ckpt_path_suffix() # sets self.file_suffix        
        self.get_ckpt_path() # 
        self.set_ckpt_path() # sets self.ckpt_path, self.tb_path, self.stat_path

        # Set the logpaths
        self.logf = open(self.log_path, 'w')
        self.logf.write(self.params_str + '\n')
        self.logf.close()
        
        # Set tensorboards.
        self.writer = SummaryWriter(log_dir=self.tb_path)
        

    def get_dataset(self):
        """Gets dataset."""

        if self.dataset_name in ['German', 'Adult']:
            return Tabular(self.dataset_name, lab_split=self.hp.lab_split)
        elif self.dataset_name in ['Waterbirds']:
            return Waterbirds(lab_split=self.hp.lab_split)
        else:
            raise ValueError('Dataset not supported.')

    def get_dataloader(self):
        """Gets data loader."""
        dset = self.get_dataset()
        train_loader, val_loader,\
            test_loader = dset.load_dataset(batch_size=self.hp.batch_size)

        return train_loader, val_loader, test_loader, dset

    def get_model(self, input_dim, n_targets):
        """Gets model."""

        if self.hp.model_type == 'fullyconn':
            model = FullyConnected(input_dim=input_dim, latent_dim=self.hp.latent_dim)
        elif self.hp.model_type == 'resnet50':
            model = ResNet50(pretrained=True) # default model is pretrained
            last_dim = model.fc.in_features
            model.fc = nn.Linear(last_dim, n_targets)
            
        # Print model summary.
        # print(summary(model, input_dim, show_input=False))

        # Cast to CUDA if GPUs are available.
        if self.hp.flag_usegpu and torch.cuda.is_available():
            print('cuda device count: ', torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model = model.cuda()

        return model

    def get_optimizer(self, model_params):
        """Gets optimizer."""

        if self.hp.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model_params,
                                      lr=self.hp.learning_rate,
                                      weight_decay=self.hp.weight_decay,
                                      momentum=0.9)
        elif self.hp.optimizer == 'Adam':
            optimizer = optim.Adam(model_params,
                                   lr=self.hp.learning_rate)
        elif self.hp.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model_params,
                                        lr=self.hp.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.hp.weight_decay)
        else:
            raise ValueError(f'{self.hp.optimizer} not supported')

        return optimizer

    def get_scheduler(self, optimizer):
        """Gets scheduler."""

        if self.hp.scheduler == '':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda step: 1)
        elif self.hp.scheduler == 'linear':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda step: linearDecay(step,
                                                                                       1,
                                                                                       0.01,
                                                                                       self.hp.num_epoch)
                                                    )
        elif self.hp.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=(self.hp.num_epoch // 3),
                                                  gamma=0.1)
        elif self.hp.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=self.hp.num_epoch,
                                                             eta_min=0.01*self.hp.learning_rate)
        elif self.hp.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             'min',
                                                             factor=0.1,
                                                             patience=5,
                                                             threshold=0.0001,
                                                             min_lr=0,
                                                             eps=1e-08)
        else:
            raise ValueError(f'{self.hp.scheduler} not supported')

        return scheduler

    def get_ckpt(self):
        """Loads from checkpoint when exists."""
        """Note. Function cannot be called independently"""

        # TODO: Save and load best val accuracy
        
        ckpt_name = os.path.join(self.ckpt_path, 'ckpt.pth')
        if self.hp.resume is True and os.path.exists(ckpt_name):
            state = torch.load(ckpt_name)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.epoch = state['epoch']
        else:
            self.epoch = 0
        return self.epoch

    def save_checkpoint(self, suffix='ckpt', updatelog=True):
        """Saves model checkpoint."""
        """Note. Function cannot be called independently"""
        if not self.hp.flag_saveckpt  or not updatelog :
            return
        
        ckpt_name = os.path.join(self.ckpt_path, f'{suffix}.pth')
        state = {
            'hparams': self.hp,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(state, ckpt_name)
        print(f'checkpoint saved at {ckpt_name}')
        del state

    def get_ckpt_path_suffix(self):
        """Gets file suffix."""

        self.file_suffix = ''

    def get_ckpt_path(self):
        """Gets file paths to save model, tensorboard, etc."""

        if self.hp.ckpt_path:
            ckpt_path = self.hp.ckpt_path
        elif self.hp.flag_debug:
            ckpt_path = f'{self.dataset_name}_{self.__class__.__name__}_debug'
        else:
            runTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ckpt_path = f'{self.dataset_name}_{self.__class__.__name__}_{runTime}'            
        self.ckpt_path = ckpt_path
        params = ['dataset', self.dataset_name,
                  'method', self.__class__.__name__,
                  'optimizer', self.hp.optimizer,
                  'learning_rate', self.hp.learning_rate,
                  'batch_size', self.hp.batch_size,
                  'seed', self.hp.seed,
                  'latent_dim', self.hp.latent_dim,
                  'lab_split', self.hp.lab_split]
        self.params_str = '_'.join([str(x) for x in params])
        

    def set_ckpt_path(self):
        """Sets file paths to save model, tensorboard, etc."""

        if self.file_suffix:
            self.ckpt_path = f'{self.ckpt_path}_{self.file_suffix}'
        self.ckpt_path = self.ckpt_path.replace('__', '_')

        # Set paths for saved model, tensorboard and eval stats.
        self.ckpt_path = os.path.join(self.hp.ckpt_prefix, self.ckpt_path)
        self.log_path = os.path.join(self.ckpt_path, 'log.txt')
        self.tb_path = os.path.join(self.ckpt_path, 'tb')
        self.stat_path = os.path.join(self.ckpt_path, 'stat')
        print(self.ckpt_path)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.exists(self.tb_path):
            os.makedirs(self.tb_path)
        if not os.path.exists(self.stat_path):
            os.makedirs(self.stat_path)

    def get_config(self):
        """Gets config."""

        # Set random seed.
        np.random.seed(self.hp.seed)
        torch.manual_seed(self.hp.seed)
        torch.backends_cudnn_deterministic = True if self.hp.seed == 42 else False

        # Data loader.
        self.train_loader, self.val_loader, \
            self.test_loader, self.dset = self.get_dataloader()

        # Model architecture (using DataParallel).
        dummy_x, _, _ = self.dset.train_set.__getitem__(0)
        input_dim = dummy_x.size(0)
        n_targets = self.dset.n_targets
        self.model = self.get_model(input_dim, n_targets)

        # Optimizer and Scheduler.
        self.optimizer = self.get_optimizer(self.model.parameters())
        self.scheduler = self.get_scheduler(self.optimizer)

    def create_metrics_dict(self):
        """Gets metrics.

        Each key in metrics_dict is of the kind f"{prefix}{measure}{control}". Here, prefix stands train/val/test. Measure is either loss/acc/y_score/y_true. Control stands for the sensitive group ID for which the measure is computed. When control is set to "-1" then the measure across all the samples is computed.
        """
        n_controls = self.dset.n_controls
        self.metrics_dict = {}
        
        for prefix in ['train', 'val', 'test']:
            for control in range(-1, n_controls):
                for measure in ['loss', 'acc']:
                    self.metrics_dict[f'{prefix}.{measure}.{control}'] = AverageMeter()
                for measure in ['y_score', 'y_true']:
                    self.metrics_dict[f'{prefix}.{measure}.{control}'] = np.array([])

                    
    def reset_metrics_dict(self, prefix=''):
        """Note. Function cannot be called independently"""
        
        for key in self.metrics_dict:
            if key.startswith(prefix):
                if isinstance(self.metrics_dict[key], AverageMeter):
                    self.metrics_dict[key].reset()
                elif isinstance(self.metrics_dict[key], np.ndarray):
                    self.metrics_dict[key] = []

    def monitor(self, updatelog=True):
        """Prints monitoring variables."""
        """Note. Function cannot be called independently"""
        
        # Command line outputs.
        t = 'time'
        message = f'[{self.epoch}/{self.hp.num_epoch}] {t: <7} (train) {self.train_time: .2f} (min) (eval) {self.eval_time: .2f} (min)'
        print(message)
        if updatelog: self.logf.write(message+'\n')
        for p in ['train', 'val', 'test']:
            string_to_print = f'[{self.epoch}/{self.hp.num_epoch}] {p: <7} \n'
            min_acc, max_acc = float('inf'), -float('inf')
            for cid in range(-1, self.dset.n_controls):
                m = 'acc'
                score = self.metrics_dict[f'{p}.{m}.{cid}'].get_avg()
                string_to_print += f' {m} group{cid} {score:.4f}'
                min_acc, max_acc = min(min_acc, score), max(max_acc, score)
                
                m = 'auc'
                score = MetricsEval().roc_auc(self.metrics_dict[f'{p}.y_score.{cid}'],\
                                              self.metrics_dict[f'{p}.y_true.{cid}'])
                string_to_print += f' {m} group{cid} {score:.4f} \n'

            print(string_to_print)
            if updatelog: self.logf.write(string_to_print+'\n')
                
            # TODO update the assertion for multiple groups
            assert ((min_acc <= self.metrics_dict[f'{p}.acc.-1'].get_avg() <= max_acc) \
                    or (min_acc <= self.metrics_dict[f'{p}.acc.-1'].get_avg() <= max_acc)), "Accuracy Trend incorrect"
            
        # Tensorboards.
        for p in ['train', 'val', 'test']:
            for cid in range(-1, self.dset.n_controls):
                m = 'acc'
                score = self.metrics_dict[f'{p}.{m}.{cid}'].get_avg()
                self.writer.add_scalar(f'{p}/{m}.{cid}', score, self.epoch)
                
                m = 'auc'
                score = MetricsEval().roc_auc(self.metrics_dict[f'{p}.y_score.{cid}'],\
                                              self.metrics_dict[f'{p}.y_true.{cid}'])
                self.writer.add_scalar(f'{p}/{m}.{cid}', score, self.epoch)                
                
    def dump_stats_to_json(self):
        """Dumps metrics to json.
        todo: write json dump file
        """
        pass
    
        
    def train(self):
        """Trains a model."""

        start_epoch = self.train_begin() # calls create_metrics_dict(), get_ckpt
        for epoch in range(start_epoch, self.hp.num_epoch):
            self.train_epoch_begin() # calls reset_metrics_dict(), train_time
            self.train_epoch() # calls train_step
            self.train_epoch_end() # calls eval_epoch, scheduler, monitor, save_checkpoint
        self.train_end() # calls save_checkpoint, dump_stats_to_json

    def train_begin(self):
        """Calls at the beginning of the training."""

        # Get metrics.
        self.create_metrics_dict()

        # Load from checkpoint when exists.
        start_epoch = self.get_ckpt()
        self.best_val_acc = 0.
        
        return start_epoch

    def train_end(self):
        """Calls at the end of the training."""

        self.writer.close()
        self.save_checkpoint('last')
        self.dump_stats_to_json()

    def train_epoch(self):
        """Trains a model for one epoch."""
        """Note: Should be called only after get_config"""

        self.model.train()
        for batch in self.train_loader:
            self.train_step(batch)
            if self.hp.flag_debug: break

    def train_epoch_begin(self):
        """Calls at the beginning of the epoch."""
        """Note: Should be called only after get_config"""
        """Requires self.metrics to be defined"""
        
        self.reset_metrics_dict(prefix='train')
        self.logf = open(self.log_path, 'a+')
        self.train_time = time.time()

    def train_epoch_end(self):
        """Calls at the end of the epoch."""
        """Note: Should be called only after get_config"""                

        self.train_time = (time.time() - self.train_time) / 60.0
        self.eval_time = time.time()

        val_acc = self.eval_epoch(self.val_loader, prefix='val')
        test_acc = self.eval_epoch(self.test_loader, prefix='test')

        self.eval_time = (time.time() - self.eval_time) / 60.0
        self.scheduler.step()
        self.epoch += 1

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            updatelog = True
        else:
            updatelog = False
        
        self.monitor(updatelog)
        self.save_checkpoint('ckpt', updatelog)
        self.logf.close()

    def train_step(self, batch):
        """Trains a model for one step."""
        """This will be model specific """
        pass
    

    def eval_epoch(self, data_loader, prefix='test'):
        """Evaluates a model for one epoch."""
        """Note: Should be called only after get_config"""
        """Requires self.metrics to be defined"""
        
        self.model.eval()
        self.reset_metrics_dict(prefix=prefix)
        for batch in data_loader:
            self.eval_step(batch, prefix=prefix)
            if self.hp.flag_debug: break
            
        return self.metrics_dict[f'{prefix}.acc.-1'].get_avg()

    def eval_step(self, batch, prefix='test'):
        """Evaluates a model for one step."""
        """Note: Should be called only after get_config"""
        pass
