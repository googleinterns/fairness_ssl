"""Environment Inference Model
ref - https://arxiv.org/pdf/2010.07249.pdf
"""

import torch
import torch.nn.functional as F

from util.train import BaseTrain
from util.solver import Solver
from util.utils import HParams, DEFAULT_MISSING_CONST as DF_M
from util.metrics_store import MetricsEval

from pytorch_model_summary import summary

import numpy as np

import pdb

TOL = 1e-4

class EIIL(BaseTrain):
    """
    A two phase approach. In the first phase, group assignments are estimated that violate the
    invariance principle. In the second phase, the estimated group assignments are used to train
    an invariant risk minimizer. 
    """

    def __init__(self, hparams):
        super(EIIL, self).__init__(hparams)

    def get_ckpt_path(self):
        super(EIIL, self).get_ckpt_path()
        new_params = ['_worstoffdro_stepsize', self.hp.worstoffdro_stepsize,
                      '_worstoffdro_marginals', '-'.join(self.hp.worstoffdro_marginals),
                      '_epsilon', self.hp.epsilon]
        self.params_str += '_'.join([str(x) for x in new_params])
        print(self.params_str)
        
    def get_config(self):
        super(EIIL, self).get_config()

        # Additional Parameters
        self.est_groups = torch.randn(len(self.dset.train_set), self.dset.n_controls).requires_grad_()
        self.scale = torch.tensor(1.).requires_grad_()
        self.phase2_batch_idx = 0
        self.phase2_max_batch_idx = int(len(self.dset.train_set) / self.hp.batch_size)
        
        # Additional hyperparameters
        self.eiil_phase1_done = False
        self.eiil_refmodel_epochs = self.hp.eiil_refmodel_epochs
        self.eiil_phase1_steps = self.hp.eiil_phase1_steps
        self.eiil_phase1_lr = self.hp.eiil_phase1_lr
        self.eiil_phase2_penalwt = self.hp.eiil_phase2_penalwt
        self.eiil_phase2_annliter = self.hp.eiil_phase2_annliter
        
        # params to gpu
        if self.hp.flag_usegpu and torch.cuda.is_available():
            self.est_groups = self.est_groups.cuda()
            self.scale = self.scale.cuda()
        
    def train_reference_model(self):
        # Model architecture for reference model (using DataParallel).
        dummy_x, _, _ = self.dset.train_set.__getitem__(0)
        input_dim = dummy_x.size(0)
        n_targets = self.dset.n_targets
        self.model_ref = self.get_model(input_dim, n_targets)

        # Optimizer and Scheduler for the reference model
        self.optimizer_ref = self.get_optimizer(self.model_ref.parameters())
        self.scheduler_ref = self.get_scheduler(self.optimizer_ref)

        # Monitor ref model train accuracy
        self.metrics_ref = AverageMeter()
        
        # Train the reference model
        self.model_ref.train()
        for epoch in range(self.eiil_refmodel_epochs):
            for batch in self.train_loader:
                x = batch[0].float()
                y = batch[1].long()
                c = batch[2].long()
                if self.hp.flag_usegpu and torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    c = c.cuda()

                # Compute loss 
                y_logit = self.model_ref(x)
                y_pred = torch.argmax(y_logit, 1)

                loss = F.cross_entropy(y_logit, y)
                
                # Compute gradient.
                self.optimizer_ref.zero_grad()
                loss.backward()
                self.optimizer_ref.step()

                # Compute Accuracy
                self.metrics_ref.update(val=MetricsEval().accuracy(y_pred, y),
                                        size=len(y))

            # End of one epoch
            self.scheduler_ref.step()
            message=f'Phase 1: {epoch}/{self.eiil_refmodel_epochs} train accuracy-{self.metrics_ref.get_avg()}'
            print(message)

    def infer_groups(self):
        self.model_ref.eval()

        # optmizer for groups
        optimizer_groups = optim.Adam([self.est_groups], lr=self.eiil_phase1_lr)
            
        # optimize over the group labels
        for epoch in range(self.eiil_phase1_steps):
            for batch_idx, batch in enumerate(self.train_loader):
                x = batch[0].float()
                y = batch[1].long()
                #c = batch[2].long()
                if self.hp.flag_usegpu and torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    #c = c.cuda()

                est_groups_batch = self.est_groups[batch_idx * self.hp.batch_size:(batch_idx + 1) * self.hp.batch_size]

                # Compute loss 
                y_logit = self.model_ref(x)
                loss = F.cross_entropy(y_logit * self.scale, y, reduction='none')
                loss_groups = (loss.squeeze() * F.softmax(est_groups_batch, dim=1)).mean()
                grad_groups = autograd.grad(loss_groups, [self.scale], create_graph=True)[0]
                penalty = (grad_groups ** 2).mean()
                npenalty = - penalty

                # Compute gradient.
                optimizer_groups.zero_grad()
                npenalty.backward(retain_graph=True)
                optimizer_groups.step()

        # Compute the control groups
        est_control =  torch.argmax(self.est_groups, 1).detach()
        return est_control
        
    def train_step(self, batch):

        if self.epoch == 0 and not self.eiil_phase1_done:
            """Train a reference model"""
            self.train_reference_model()

            """Run Phase 1 to infer the groups"""
            est_control = self.infer_groups()

            """Update Phase 1 flag"""
            self.eiil_phase1_done = True

        
        """Trains a model for one step in Phase 2"""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].long()
        #c = batch[2].long()
        if self.hp.flag_usegpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            #c = c.cuda()

        # Compute XE loss
        y_logit = self.model(x)
        y_pred = torch.argmax(y_logit, 1)
        loss = F.cross_entropy(y_logit, y)
        
        # Compute penalty loss
        est_control_batch  = est_control[self.phase2_batch_idx * self.hp.batch_size:(self.phase2_batch_idx + 1) * self.hp.batch_size]
        self.phase2_batch_idx = (self.phase2_batch_idx + 1) % (self.phase2_max_batch_idx)
        
        scale = torch.tensor(1.).requires_grad_()
        if self.hp.flag_usegpu and torch.cuda.is_available():
            scale = scale.cuda()
        loss_scale = F.cross_entropy(y_logit*scale, y)
        grad = autograd.grad(loss_scale, [scale], create_graph=True)[0]
        loss_penalty = torch.sum(grad**2)
            
        # Penalty regularization
        penalty_weight = (self.eiil_phase2_penalwt
                          if self.epoch >= self.eiil_phase2_annliter else 1.0)
        loss += penalty_weight * loss_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
    trainer = EIIL(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'fullyconn',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
