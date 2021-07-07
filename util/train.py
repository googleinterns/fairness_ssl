"""Base trainer."""

import os
import time
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

import pdb

from data.tabular import Tabular

from model.fullyconn import FullyConnected

from util.utils import HParams

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

        # Set tensorboards.
        self.writer = SummaryWriter(log_dir=self.tb_path)
        

    def get_dataset(self):
        """Gets dataset."""

        if self.dataset_name in ['German', 'Adult']:
            return Tabular(self.dataset_name)
        else:
            raise ValueError('Dataset not supported.')

    def get_dataloader(self):
        """Gets data loader."""
        dset = self.get_dataset()
        train_loader, val_loader,\
            test_loader = dset.load_dataset(batch_size=self.hp.batch_size)

        return train_loader, val_loader, test_loader, dset

    def get_model(self, input_dim):
        """Gets model."""

        if self.hp.model_type == 'fullyconn':
            model = FullyConnected(input_dim=input_dim, latent_dim=self.hp.latent_dim)

        # Print model summary.
        print(summary(model, input_dim, show_input=False))

        # Cast to CUDA if GPUs are available.
        if self.hp.use_gpu == 'True' and torch.cuda.is_available():
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
        else:
            raise ValueError(f'{self.hp.scheduler} not supported')

        return scheduler

    def get_ckpt(self):
        """Loads from checkpoint when exists."""
        """Note. Function cannot be called independently"""
        
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

    def save_checkpoint(self, suffix='ckpt'):
        """Saves model checkpoint."""
        """Note. Function cannot be called independently"""
        if self.hp.save_checkpoint == 'False' :
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
        else:
            ckpt_path = os.path.join(f'{self.dataset_name}_fold{self.hp.dataseed}',
                                    self.hp.model_type,
                                    f'{self.__class__.__name__}_{self.hp.optimizer}_{self.hp.scheduler}' + \
                                    f'_lr{self.hp.learning_rate}_wd{self.hp.weight_decay}_seed{self.hp.seed}' + \
                                    f'_bs{self.hp.batch_size}_ep{self.hp.num_epoch}')
        self.ckpt_path = ckpt_path

    def set_ckpt_path(self):
        """Sets file paths to save model, tensorboard, etc."""

        if self.file_suffix:
            self.ckpt_path = f'{self.ckpt_path}_{self.file_suffix}'
        self.ckpt_path = self.ckpt_path.replace('__', '_')

        # Set paths for saved model, tensorboard and eval stats.
        self.ckpt_path = os.path.join(self.hp.ckpt_prefix, self.ckpt_path)
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
        self.model = self.get_model(input_dim)

        # Optimizer and Scheduler.
        self.optimizer = self.get_optimizer(self.model.parameters())
        self.scheduler = self.get_scheduler(self.optimizer)

    def create_metrics_dict(self):
        """Gets metrics.
        Note: Each metric computes sum total over the samples in the batch
        """
        n_controls = self.dset.n_controls
        self.metrics_dict = {}

        for prefix in ['train', 'val', 'test']:
            for control in range(-1, n_controls):
                for measure in ['loss', 'size', 'acc']:
                    self.metrics_dict[f'{prefix}.{measure}.{control}'] = 0.0
                for measure in ['y_score', 'y_true']:
                    self.metrics_dict[f'{prefix}.{measure}.{control}'] = []

                    
    def reset_metrics_dict(self, prefix=''):
        """Note. Function cannot be called independently"""
        
        for key in self.metrics_dict:
            if key.startswith(prefix):
                if isinstance(self.metrics_dict[key], float):
                    self.metrics_dict[key] = 0.0
                elif isinstance(self.metrics_dict[key], list):
                    self.metrics_dict[key] = []

    def monitor(self):
        """Prints monitoring variables."""
        """Note. Function cannot be called independently"""
        
        # Command line outputs.
        t = 'time'
        print(f'[{self.epoch}/{self.hp.num_epoch}] {t: <7} (train) {self.train_time: .2f} (min) (eval) {self.eval_time: .2f} (min)')
        for p in ['train', 'val', 'test']:
            string_to_print = f'[{self.epoch}/{self.hp.num_epoch}] {p: <7}'
            for cid in range(-1, self.dset.n_controls):
                m = 'acc'
                score = self.metrics_dict[f'{p}.{m}.{cid}']
                string_to_print += f' {m} group{cid} {score:.4f}'

                m = 'auc'
                score = MetricsEval().roc_auc(self.metrics_dict[f'{p}.y_score.{cid}'],\
                                              self.metrics_dict[f'{p}.y_true.{cid}'])
                string_to_print += f' {m} group{cid} {score:.4f}'
                
            print(string_to_print)

        # Tensorboards.
        for p in ['train', 'val', 'test']:
            for cid in range(-1, self.dset.n_controls):
                m = 'acc'
                score = self.metrics_dict[f'{p}.{m}.{cid}'] 
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

    def train_epoch_begin(self):
        """Calls at the beginning of the epoch."""
        """Note: Should be called only after get_config"""
        """Requires self.metrics to be defined"""
        
        self.reset_metrics_dict(prefix='train')
        self.train_time = time.time()

    def train_epoch_end(self):
        """Calls at the end of the epoch."""
        """Note: Should be called only after get_config"""                

        self.train_time = (time.time() - self.train_time) / 60.0
        self.eval_time = time.time()
        self.eval_epoch(self.val_loader, prefix='val')
        self.eval_epoch(self.test_loader, prefix='test')
        self.eval_time = (time.time() - self.eval_time) / 60.0
        self.scheduler.step()
        self.epoch += 1
        self.monitor()
        self.save_checkpoint('ckpt')

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

    def eval_step(self, batch, prefix='test'):
        """Evaluates a model for one step."""
        """Note: Should be called only after get_config"""
        pass
