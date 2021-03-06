import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

import pdb

class MetricsEval(object):
    """A class to evaluate different metrics."""
    
    def __init__(self):
        pass

    def get_allmeasures(self):
        return ['loss', 'acc', 'y_score', 'y_true']
        
    def get_reduction(self, m, reduction='mean'):
        if reduction == 'mean':
            return torch.mean(m)
        elif reduction == 'sum':
            return torch.sum(m)
        elif reduction == 'none':
            return m

    def cross_entropy(self, y_logit, y_true):
        loss =  F.cross_entropy(y_logit, y_true)
        return loss.item()

    def accuracy(self, y_pred, y_true):
        acc = torch.sum(y_pred == y_true) / y_pred.size(0)
        return acc.item()

    def roc_auc(self, y_score, y_true):
        # TODO: Temporary fix
        if len(np.unique(y_true)) < 2:
            return 0
        return roc_auc_score(y_true, y_score)

    def logit2prob(self, logits):
        logits = logits - torch.max(logits, 1)[0].unsqueeze(1)
        logits = torch.exp(logits)
        prob = logits / logits.sum(dim=1).unsqueeze(1)
        return prob[:, 1].detach()

if __name__ == '__main__':
    pass

    
