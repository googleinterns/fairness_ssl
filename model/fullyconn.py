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
        
        # Encoder layers
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.latent_dim) 

        # y-predictor
        self.fc3 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_targets)          
        self.relu = nn.ReLU()

    def encode(self, x):
        fc1 = self.relu(self.fc1(x))
        fc2 = self.fc2(fc1)
        return fc2
    
    def predict(self, x):
        fc3 = self.relu(self.fc3(x))
        fc4 = self.fc4(fc3)
        return fc4
        
    def forward(self, x):
        z = self.encode(x)
        pred = self.predict(z)
        
        return pred

if __name__ == '__main__':
    fullyconn = FullyConnected()
