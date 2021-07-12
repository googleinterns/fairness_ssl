"""Fully-Supervised model."""

import torch
import torch.nn.functional as F

from model.fullyconn import FullyConnected

from util.train import BaseTrain
from util.utils import HParams, DEFAULT_MISSING_CONST as DF_M
from util.metrics_store import MetricsEval

from pytorch_model_summary import summary

import numpy as np

import pdb

class ERM(BaseTrain):
    """Fully-supervised model trainer."""

    def __init__(self, hparams):
        super(ERM, self).__init__(hparams)
        
    def train_step(self, batch):
        """Trains a model for one step."""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].long()
        c = batch[2].long()
        if self.hp.flag_usegpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()

        # Check for missing values
        if DF_M in c:
            raise ValueError('Missing values not supported')
        
        # Compute loss 
        y_logit = self.model(x)
        y_pred = torch.argmax(y_logit, 1)
        
        loss = F.cross_entropy(y_logit, y)

        # Compute gradient.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update metrics.
        # Maintains running average over all the metrics
        prefix = 'train'
        for cid in range(-1, self.dset.n_controls):
            select = c >= 0 if cid == -1 else c == cid

            self.metrics_dict[f'{prefix}.loss.{cid}'].update(
                MetricsEval().cross_entropy(y_logit[select], y[select]))
            
            self.metrics_dict[f'{prefix}.acc.{cid}'].update(
                MetricsEval().accuracy(y_pred[select], y[select]))
            
            self.metrics_dict[f'{prefix}.y_score.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_score.{cid}'],
                                MetricsEval().logit2prob(y_logit[select]).cpu().numpy()))
            
            self.metrics_dict[f'{prefix}.y_true.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_true.{cid}'],
                                y[select].cpu().numpy()))
            

        

    def eval_step(self, batch, prefix='test'):
        """Trains a model for one step."""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].long()
        c = batch[2].long()
        if self.hp.flag_usegpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()

        # Check for missing values
        if DF_M in c:
            raise ValueError('Missing values not supported')

        # Compute loss
        with torch.no_grad():
            y_logit = self.model(x)
            y_pred = torch.argmax(y_logit, 1)
            
        for cid in range(-1, self.dset.n_controls):
            select = c >= 0 if cid == -1 else c == cid

            self.metrics_dict[f'{prefix}.loss.{cid}'].update(
                MetricsEval().cross_entropy(y_logit[select], y[select]))
            
            self.metrics_dict[f'{prefix}.acc.{cid}'].update(
                MetricsEval().accuracy(y_pred[select], y[select]))
            
            self.metrics_dict[f'{prefix}.y_score.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_score.{cid}'],
                                MetricsEval().logit2prob(y_logit[select]).cpu().numpy()))
            
            self.metrics_dict[f'{prefix}.y_true.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_true.{cid}'],
                                y[select].cpu().numpy()))
            
if __name__ == '__main__':
    trainer = ERM(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'fullyconn',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
