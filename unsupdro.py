"""Fully-Supervised model."""

import torch
import torch.nn.functional as F

from model.fullyconn import FullyConnected

from util.train import BaseTrain
from util.utils import HParams
from util.metrics_store import MetricsEval

from pytorch_model_summary import summary

from util.utils import HParams

import numpy as np

import pdb

class UnsupDRO(BaseTrain):
    """Fully Unsupervised DRO measure.
    https://arxiv.org/pdf/1806.08010.pdf
    """

    def __init__(self, hparams):
        super(UnsupDRO, self).__init__(hparams)
        print('UnsupDRO!')

    def get_ckpt_path(self):
        super(UnsupDRO, self).get_ckpt_path()
        new_params = ['_unsupdro_eta', self.hp.unsupdro_eta]
        self.params_str += '_'.join([str(x) for x in new_params])
        
    def get_config(self):
        super(UnsupDRO, self).get_config()

        # Additional hyperparameters.
        self.unsupdro_eta = self.hp.unsupdro_eta
        self.relu = torch.nn.ReLU()
            
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
            
        # Compute loss 
        y_logit = self.model(x)
        y_pred = torch.argmax(y_logit, 1)

        # Calculating unsupervised dro
        loss = F.cross_entropy(y_logit, y, reduction='none')
        loss_unsupdro = self.relu(loss - self.unsupdro_eta).mean()
        
        # Compute gradient.
        self.optimizer.zero_grad()
        loss_unsupdro.backward()
        self.optimizer.step()

        # Update metrics.
        # Maintains running average over all the metrics
        # TODO: eliminate the need for for loops
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

        # Compute loss
        with torch.no_grad():
            y_logit = self.model(x)
            y_pred = torch.argmax(y_logit, 1)

        for cid in range(-1, self.dset.n_controls):
            select = c != DF_M if cid == -1 else c == cid 

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
    trainer = UnsupDRO(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'fullyconn',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
