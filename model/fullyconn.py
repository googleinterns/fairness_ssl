"""NNet architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import model
import model.model_utils as util

class FullyConnected(nn.Module):
    """
    A simple four layered neural network predicting the target attributes
    """
    def __init__(self, input_dim=121, latent_dim=64, n_targets=2):
        super(FullyConnected, self).__init__()
        
        self.latent_dim = latent_dim

        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.latent_dim) 
        self.fc3 = nn.Linear(self.latent_dim, n_targets)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)        
        return out

if __name__ == '__main__':
    fullyconn = FullyConnected()
