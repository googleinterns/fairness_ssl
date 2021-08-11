"""Semi-Supervised model."""

import torch
import torch.nn.functional as F

from model.fullyconn import FullyConnected

from util.train import BaseTrain
from util.solver import Solver
from util.utils import HParams, DEFAULT_MISSING_CONST as DF_M
from util.metrics_store import MetricsEval

from pytorch_model_summary import summary

import numpy as np

import pdb

class WorstoffDRO(BaseTrain):
    """Partially labelled DRO measure.
    """

    def __init__(self, hparams):
        super(WorstoffDRO, self).__init__(hparams)

    def get_ckpt_path(self):
        super(WorstoffDRO, self).get_ckpt_path()
        new_params = ['_worstoffdro_stepsize', self.hp.worstoffdro_stepsize,
                      '_worstoffdro_lambda', self.hp.worstoffdro_lambda,
                      '_worstoffdro_latestart', self.hp.worstoffdro_latestart]
        self.params_str += '_'.join([str(x) for x in new_params])
    
    def get_config(self):
        super(WorstoffDRO, self).get_config()

        # Additional parameters
        self.weights = torch.ones(self.dset.n_controls)/self.dset.n_controls
        self.map_vector = torch.arange(self.dset.n_controls).unsqueeze(0).long()
        
        # Additional hyperparameters.
        self.worstoffdro_stepsize = self.hp.worstoffdro_stepsize
        self.worstoffdro_lambda = self.hp.worstoffdro_lambda
        self.worstoffdro_latestart = self.hp.worstoffdro_latestart

        # Initialize the solver
        self.solver = Solver(n_controls=self.dset.n_controls, \
                             bsize=self.hp.batch_size,\
                             marginals=torch.ones(self.dset.n_controls)/self.dset.n_controls)

        # params to gpu
        if self.hp.flag_usegpu and torch.cuda.is_available():
            self.weights = self.weights.cuda()
            self.map_vector = self.map_vector.cuda()
        
    def compute_loss(self, sample_groups, sample_losses):
        sample_counts = sample_groups.sum(0)
        denom = sample_counts + (sample_counts==0).float()
        loss_gp = (sample_losses.unsqueeze(0) @ sample_groups) / denom
        loss = loss_gp.squeeze(0) @ self.weights
        return loss, loss_gp
    
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

        # Get labelled loss
        g_lab = (c.unsqueeze(1) == self.map_vector).float() # 128 X 1, 1 X 4 -> 128 X 4
        loss_lab, loss_lab_gp = self.compute_loss(g_lab[c!=DF_M], loss[c!=DF_M])
        
        # Get unlabelled loss
        Gamma_g = self.solver.eval_nearestnbhs(x)
        g_hat = self.solver.cvxsolve(losses=loss,
                                     weights=self.weights,
                                     Gamma_g=Gamma_g)
        if self.hp.flag_usegpu and torch.cuda.is_available():
            g_hat = g_hat.cuda()
        loss_unlab, loss_unlab_gp = self.compute_loss(g_hat[c==DF_M], loss[c==DF_M])
        
        # Total loss
        if self.epoch >= self.worstoffdro_latestart:
            loss_worstoffdro = loss_lab + self.worstoffdro_lambda * loss_unlab
        else:
            loss_worstoffdro = loss_lab
            
        # Update Neural network parameters
        self.optimizer.zero_grad()
        loss_worstoffdro.backward()
        self.optimizer.step()

        # Update Weights
        if self.epoch >= self.worstoffdro_latestart:
            loss_worstoffdro_gp = (loss_lab_gp + self.worstoffdro_lambda * loss_unlab_gp).view(-1)
        else:
            loss_worstoffdro_gp = loss_lab_gp.view(-1)
            
        self.weights = self.weights * torch.exp(self.worstoffdro_stepsize*loss_worstoffdro_gp.data)
        self.weights = self.weights/(self.weights.sum())
        
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
    trainer = WorstoffDRO(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'fullyconn',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
