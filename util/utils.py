import torch

class HParams(dict):
    """Custom dictionary that allows to access dict values as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.valsum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.valsum += val * n
        self.count += n
        self.avg = self.valsum / self.count

    def get_avg(self):
        return self.avg

    def get_current_val(self):
        return self.val
