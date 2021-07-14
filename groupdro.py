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

class GroupDRO(BaseTrain):
    """Fully-supervised DRO measure.
    https://arxiv.org/pdf/1911.08731.pdf
    """

    def __init__(self, hparams):
        super(GroupDRO, self).__init__(hparams)

    def get_ckpt_path(self):
        super(GroupDRO, self).get_ckpt_path()
        new_params = ['_groupdro_stepsize', self.hp.groupdro_stepsize]
        self.params_str += '_'.join([str(x) for x in new_params])
    
    def get_config(self):
        super(GroupDRO, self).get_config()

        # Additional parameters
        self.weights = torch.ones(self.dset.n_controls)/self.dset.n_controls
        
        # Additional hyperparameters.
        self.groupdro_stepsize = self.hp.groupdro_stepsize
        
    def stats_per_control(self, sample_losses, cid):
        control_map = (cid == torch.arange(self.dset.n_controls).unsqueeze(1).long()).float()
        control_count = control_map.sum(1)
        divide = control_count + (control_count==0).float() # handling None
        control_loss = (control_map @ sample_losses.view(-1))/divide
        return control_loss, control_count

    def calculate_groupdro_loss(self, control_loss, control_count):
        self.weights = self.weights * torch.exp(self.groupdro_stepsize*control_loss.data)
        self.weights = self.weights/(self.weights.sum())

        loss_groupdro = control_loss @ self.weights

        self.writer.add_scalar(f'train/weights.0', self.weights[0], self.epoch)                
        self.writer.add_scalar(f'train/weights.1', self.weights[1], self.epoch)                        
        return loss_groupdro
    
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

        loss = F.cross_entropy(y_logit, y, reduction='none')
        control_loss, control_count = self.stats_per_control(loss, c)
        loss_groupdro = self.calculate_groupdro_loss(control_loss, control_count)
        
        # Compute gradient.
        self.optimizer.zero_grad()
        loss_groupdro.backward()
        self.optimizer.step()

        # Update metrics.
        # Maintains running average over all the metrics
        # TODO: eliminate the need for for loops
        prefix = 'train'
        for cid in range(-1, self.dset.n_controls):
            # Unlabelled samples are denoted by DF_M
            # -1 indicates metrics computed over all the samples
            select = c != DF_M if cid == -1 else c == cid
            size = sum(select)
            
            self.metrics_dict[f'{prefix}.loss.{cid}'].update(
                val=MetricsEval().cross_entropy(y_logit[select], y[select]),
                num=size)
            
            self.metrics_dict[f'{prefix}.acc.{cid}'].update(
                val=MetricsEval().accuracy(y_pred[select], y[select]),
                num=size)
            
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
            size = sum(select)
            
            self.metrics_dict[f'{prefix}.loss.{cid}'].update(
                val=MetricsEval().cross_entropy(y_logit[select], y[select]),
                num=size)
            
            self.metrics_dict[f'{prefix}.acc.{cid}'].update(
                val=MetricsEval().accuracy(y_pred[select], y[select]),
                num=size)
            
            self.metrics_dict[f'{prefix}.y_score.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_score.{cid}'],
                                MetricsEval().logit2prob(y_logit[select]).cpu().numpy()))
            
            self.metrics_dict[f'{prefix}.y_true.{cid}'] = \
                np.concatenate((self.metrics_dict[f'{prefix}.y_true.{cid}'],
                                y[select].cpu().numpy()))
            
if __name__ == '__main__':
    trainer = GroupDRO(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'fullyconn',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
