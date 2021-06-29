"""Utilities to run model training and evaluation."""

from data.tabular import Tabular

def get_tabular_data():
  dl = Tabular(data_name='Adult')
  train_loader, valid_loader, test_loader = dl.load_dataset(batch_size=64)
  
  import pdb
  pdb.set_trace()
  # Extract a single batch of 64 examples
  x, y, s, index = train_step(iterator)
  import pdb
  pdb.set_trace()


def train_step(iterator):
  def step_fn(data):
    x = data[0]
    y = data[1]
    s = data[2]
    index = data[3]
    return x, y, s, index

  return step_fn(next(iterator))
