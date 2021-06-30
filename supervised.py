"""Fully-Supervised model."""

import torch
import torch.nn.functional as F

from util.train import BaseTrain
from util.utils import HParams
#, get_accuracy


class Supervised(BaseTrain):
    """Fully-supervised model trainer."""

    def __init__(self, hparams):
        super(Supervised, self).__init__(hparams)

    def train_step(self, batch):
        pass
        '''
        """Trains a model for one step."""

        # Prepare data.
        img = batch['image'].float()
        mask = batch['mask'].float()
        labeled = batch['labeled'].float()
        if torch.cuda.is_available():
            img = img.cuda()
            mask = mask.cuda()
            labeled = labeled.cuda()

        # Compute loss for labeled portions only.
        logit = self.model(img)
        loss = F.binary_cross_entropy_with_logits(logit, mask, reduction='none')
        loss = loss * labeled
        loss = loss.mean()

        # Compute per-pixel accuracy, dice, iou.
        dice, iou, acc = get_accuracy(mask, (torch.sigmoid(logit) > 0.5).float())

        # Compute gradient.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update metrics.
        self.metrics['train.loss'] += (loss.item() * len(img))
        self.metrics['train.batch'] += len(img)
        self.metrics['train.dice'] += (dice.item() * len(img))
        self.metrics['train.iou'] += (iou.item() * len(img))
        self.metrics['train.acc'] += (acc.item() * len(img))
        '''

    def eval_step(self, batch, prefix='test'):
        pass

if __name__ == '__main__':
    trainer = Supervised(hparams=HParams({'dataset': 'Adult',
                                         'batch_size': 64,
                                         'model_type': 'nnet',
                                         'learning_rate': 0.0001,
                                         'weight_decay': 0.00001,
                                         'num_epoch': 100,
                                         }))
    trainer.get_config()
    trainer.train()
