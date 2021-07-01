import torch

class HParams(dict):
    """Custom dictionary that allows to access dict values as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
