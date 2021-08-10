import cvxpy as cp
import numpy as np
import torch

import pdb

class Solver(object):
    def __init__(self, n_controls, bsize, marginals):

        marginals = marginals.cpu().numpy()[..., np.newaxis]
        self.X = cp.Variable((bsize, n_controls))
        self.l = cp.Parameter((bsize, 1))
        self.p = cp.Parameter((n_controls, 1), value=marginals)
        self.q = cp.Parameter(n_controls)
        
        counts = cp.sum(self.X, axis=0, keepdims=True)
        #denom = counts + (counts==0)
        obj = ((self.l.T @ self.X) / self.p.T) @ self.q # can replace counts with self.p.T
        constraints = [self.X >= 0,
                       cp.sum(self.X, axis=1, keepdims=True) == np.ones((bsize, 1)),
                       cp.sum(self.X, axis=0, keepdims=True) == self.p.T]

        self.prob = cp.Problem(cp.Maximize(obj), constraints)

    def cvxsolve(self, losses, weights, Gamma_g=None):
        self.l.value = losses.cpu().numpy()[..., np.newaxis]
        self.q.value = weights.cpu().numpy()

        self.prob.solve()

        return self.X.value
    
        
    def eval_nearestnbhs(self, data):
        # TODO: nearest neighbours here
        return None
        

if __name__ == '__main__':
    n_controls = 4
    bsize = 128
    marginals = torch.tensor([.25, .25, .25, .25])
    sv = Solver(n_controls, bsize, marginals)

    losses = torch.randn(bsize)
    weights = torch.tensor([0.2500, 0.2500, 0.2500, 0.2500])

    g_hat = sv.cvxsolve(losses, weights)
    print('g_hat', g_hat)
