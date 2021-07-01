"""Fully-Supervised model."""

import torch
import torch.nn.functional as F

from util.train import BaseTrain
from util.utils import HParams
from util.metrics_store import MetricsEval


class Supervised(BaseTrain):
    """Fully-supervised model trainer."""

    def __init__(self, hparams):
        super(Supervised, self).__init__(hparams)

    def get_model(self, input_dim):
        """Gets model."""

        if self.hp.model_type == 'fullyconn':
            model = FullyConnected(input_dim=input_dim, latent_dim=self.hp.latent_dim)

        # Print model summary.
        print(summary(model, input_dim, show_input=False))

        # Cast to CUDA if GPUs are available.
        if torch.cuda.is_available():
            print('cuda device count: ', torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model = model.cuda()

        return model
        
    def train_step(self, batch):
        """Trains a model for one step."""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].float()
        c = batch[2].float()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()

        # Compute loss 
        pred_logit = self.model(x)
        loss = F.cross_entropy(pred_logit, y)
        pdb.set_trace()
        #loss = loss.mean()

        # Compute per-group accuracy, area under the curve
        acc, acc_c0, acc_c1, auc, auc_c0, auc_c1 = MetricsEval(pred_logit, y, c).float()

        # Compute gradient.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update metrics.
        #TODO: Write the case for more than two sensitive attribtues
        self.metrics['train.loss'] += (loss.item() * len(x))
        self.metrics['train.batch'] += len(x)
        self.metrics['train.acc'] += (acc.item() * len(x))
        self.metrics['train.acc_c0'] += (acc_c0.item() * len(x))
        self.metrics['train.acc_c1'] += (acc_c1.item() * len(x))        
        self.metrics['train.auc'] += (auc.item() * len(x))
        self.metrics['train.auc_c0'] += (auc_c0.item() * len(x))
        self.metrics['train.auc_c1'] += (auc_c1.item() * len(x))        

    def eval_step(self, batch, prefix='test'):
        """Trains a model for one step."""
        # Prepare data.
        x = batch[0].float()
        y = batch[1].float()
        c = batch[2].float()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()

        # Compute loss
        with torch.no_grad():
            pred_logit = self.model(x)

        # Compute per-group accuracy, area under the curve
        acc, acc_c0, acc_c1, auc, auc_c0, auc_c1 = MetricsEval(pred_logit, y, c).float()

        # Update metrics.
        #TODO: Write the case for more than two sensitive attribtues
        self.metrics['train.loss'] += (loss.item() * len(x))
        self.metrics['train.batch'] += len(x)
        self.metrics['train.acc'] += (acc.item() * len(x))
        self.metrics['train.acc_c0'] += (acc_c0.item() * len(x))
        self.metrics['train.acc_c1'] += (acc_c1.item() * len(x))        
        self.metrics['train.auc'] += (auc.item() * len(x))
        self.metrics['train.auc_c0'] += (auc_c0.item() * len(x))
        self.metrics['train.auc_c1'] += (auc_c1.item() * len(x))        

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
