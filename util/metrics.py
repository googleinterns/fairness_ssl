import torch

EPS = 0.0001

class MetricsEval(object):
    """ A class to evaluate different metrics.
    TODO: complete metrics evaluation
    """
    
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
        y_pred = y_pred.flatten(start_dim=1)
        y_true = y_true.flatten(start_dim=1)
        corrects = (y_pred == y_true).float()
        return get_reduction(corrects, reduction)


    def get_auc(self, y_pred, y_true, reduction='mean'):
        """Computes accuracy.
        
        Args:
        y_pred: 4D Tensor of 
        y_true: 4D Tensor of 
        
        Returns:
        Overall Accuracy
        """
        y_pred = y_pred.flatten(start_dim=1)
        y_true = y_true.flatten(start_dim=1)
        corrects = (y_pred == y_true).float()
        return get_reduction(corrects, reduction)
    


if __name__ == '__main__':
    y_pred = (torch.rand(3, 1, 16, 16) > 0.5).float()
    y_true = (torch.rand(3, 1, 16, 16) > 0.5).float()
    for reduction in ['mean', 'sum', 'none']:
        print(dice_score(y_pred, y_true, reduction))
        print(iou_score(y_pred, y_true, reduction))
        print(accuracy(y_pred, y_true, reduction))
