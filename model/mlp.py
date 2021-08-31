import torch
import torch.nn as nn


class MLP(nn.Module):
  def __init__(self, input_dim, latent_dim=64, n_targets=2):
    super(MLP, self).__init__()

    self.latent_dim = latent_dim
    
    self.fc1 = nn.Linear(2 * 14 * 14, self.latent_dim)
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.zeros_(self.fc1.bias)
    
    self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)
    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.zeros_(self.fc2.bias)
    
    self.fc3 = nn.Linear(self.latent_dim, n_targets)
    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)

    self.relu = nn.ReLU()
    
  def forward(self, x):
    x = x.view(x.shape[0], 2 * 14 * 14)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    out = self.fc3(x)
    return out

