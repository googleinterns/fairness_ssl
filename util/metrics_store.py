import torch

import pdb

EPS = 0.0001

class MetricsEval(object):
    """A class to evaluate different metrics."""
    
    def __init__(self, y_pred_logits, y_true, c):

        y_pred = torch.argmax(y_pred_logits, 1)
        
        return (self.get_accuracy(y_pred, y_true, 'mean'),
                self.get_accuracy_control(y_pred, y_true, c == 0, 'mean'),
                self.get_accuracy_control(y_pred, y_true, c == 1, 'mean'),            
                self.get_auc(y_pred_logits, y_true, 'mean'),
                self.get_auc_control(y_pred_logits, y_true, c == 0, 'mean'),
                self.get_auc_control(y_pred_logits, y_true, c == 1, 'mean'))

    def get_reduction(self, m, reduction='mean'):
        if reduction == 'mean':
            return torch.mean(m)
        elif reduction == 'sum':
            return torch.sum(m)
        elif reduction == 'none':
            return m

    def get_accuracy(self, y_pred, y_true, reduction='mean'):
        """Computes accuracy.
        
        Args:
        y_pred: 4D Tensor of 
        y_true: 4D Tensor of 
        
        Returns:
        Overall Accuracy
        """

        '''
        pdb.set_trace()
        torch.sum(y_pred == y_true)
        y_pred = y_pred.flatten(start_dim=1)
        y_true = y_true.flatten(start_dim=1)
        corrects = (y_pred == y_true).float()
        return get_reduction(corrects, reduction)
        '''
        return 0.0
        
    def get_accuracy_control(self, y_pred, y_true, choose, reduction='mean'):
        """Dummy function
        
        Args:
        y_pred: 4D Tensor of 
        y_true: 4D Tensor of 
        
        Returns:
        Overall Accuracy
        """

        return 0.0

    def get_auc(self, y_pred, y_true, reduction='mean'):
        """Dummy function
        
        Args:
        y_pred: 4D Tensor of 
        y_true: 4D Tensor of 
        
        Returns:
        Overall AUC
        """
        return 0.0

    def get_auc_control(self, y_pred, y_true, choose, reduction='mean'):
        """Dummy function
        
        Args:
        y_pred: 4D Tensor of 
        y_true: 4D Tensor of 
        
        Returns:
        Overall Accuracy
        """
        return 0.0

if __name__ == '__main__':
    pass

    
