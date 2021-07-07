"""Fully-Supervised model."""

import torch
import torch.nn.functional as F

from model.fullyconn import FullyConnected

from util.train import BaseTrain
from util.utils import HParams
from util.metrics_store import MetricsEval

from pytorch_model_summary import summary

import numpy as np

import pdb

class Supervised(BaseTrain):
    """Fully-supervised model trainer."""

    def __init__(self, hparams):
        super(Supervised, self).__init__(hparams)

    def get_model(self, input_dim):
        """Gets model."""

        if self.hp.model_type == 'fullyconn':
            model = FullyConnected(input_dim=input_dim, latent_dim=self.hp.latent_dim)

        # Print model summary.
        #print(summary(model, input_dim, show_input=False))

        # Cast to CUDA if GPUs are available.
        if self.hp.flag_usegpu and torch.cuda.is_available():
            print('cuda device count: ', torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model = model.cuda()

        return model
        
    def train_step(self, batch):
        """Trains a model for one step."""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].long()
        c = batch[2].long()
        if self.hp.flag_usegpu == 'True' and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()

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
        # TODO: using average meter 
        prefix = 'train'
        for cid in range(-1, self.dset.n_controls):
            bsize = len(c) if cid == -1 else len(c == cid)
            select = c > -1 if cid == -1 else c == cid

            self.metrics_dict[f'{prefix}.loss.{cid}'] = \
                (MetricsEval().cross_entropy(y_logit, y, select) * bsize + self.metrics_dict[f'{prefix}.loss.{cid}'] * self.metrics_dict[f'{prefix}.size.{cid}']) / (bsize + self.metrics_dict[f'{prefix}.size.{cid}'])
            
            self.metrics_dict[f'{prefix}.acc.{cid}'] = \
                (MetricsEval().accuracy(y_pred, y, select) * bsize + self.metrics_dict[f'{prefix}.acc.{cid}'] * self.metrics_dict[f'{prefix}.size.{cid}']) / (bsize + self.metrics_dict[f'{prefix}.size.{cid}'])

            self.metrics_dict[f'{prefix}.y_score.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_score.{cid}'], MetricsEval().logit2prob(y_logit[select]).cpu().numpy()))
            self.metrics_dict[f'{prefix}.y_true.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_true.{cid}'], y[select].cpu().numpy()))

            self.metrics_dict[f'{prefix}.size.{cid}'] += bsize
                               

    def eval_step(self, batch, prefix='test'):
        """Trains a model for one step."""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].long()
        c = batch[2].long()
        if self.hp.flag_usegpu is 'True' and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()

        # Compute loss
        with torch.no_grad():
            y_logit = self.model(x)
            y_pred = torch.argmax(y_logit, 1)
            

        for cid in range(-1, self.dset.n_controls):
            bsize = len(c) if cid == -1 else len(c == cid)
            select = c > -1 if cid == -1 else c == cid

            self.metrics_dict[f'{prefix}.loss.{cid}'] = \
                (MetricsEval().cross_entropy(y_logit, y, select) * bsize + self.metrics_dict[f'{prefix}.loss.{cid}'] * self.metrics_dict[f'{prefix}.size.{cid}']) / (bsize + self.metrics_dict[f'{prefix}.size.{cid}'])

            self.metrics_dict[f'{prefix}.acc.{cid}'] = \
                (MetricsEval().accuracy(y_pred, y, select) * bsize + self.metrics_dict[f'{prefix}.acc.{cid}'] * self.metrics_dict[f'{prefix}.size.{cid}']) / (bsize + self.metrics_dict[f'{prefix}.size.{cid}'])

            self.metrics_dict[f'{prefix}.y_score.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_score.{cid}'], MetricsEval().logit2prob(y_logit[select]).cpu().numpy()))
            self.metrics_dict[f'{prefix}.y_true.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_true.{cid}'], y[select].cpu().numpy()))

            self.metrics_dict[f'{prefix}.size.{cid}'] += bsize


            
if __name__ == '__main__':
    trainer = Supervised(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'fullyconn',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
