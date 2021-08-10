import cvxpy as cp
import numpy as np
import torch

import pdb

TOL = 1e-6

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
                       cp.sum(self.X, axis=0, keepdims=True) / bsize == self.p.T]

        self.prob = cp.Problem(cp.Maximize(obj), constraints)

    def cvxsolve(self, losses, weights, Gamma_g=None):
        self.l.value = losses.data.cpu().numpy()[..., np.newaxis]
        self.q.value = weights.data.cpu().numpy()

        self.prob.solve()

        X = self.X.value
        X[abs(X) < TOL] = 0. # set very low value to zero

        # Numpy to Torch
        X = torch.from_numpy(X).float()
        
        return X
    
        
    def eval_nearestnbhs(self, data):
        # TODO: nearest neighbours here
        return None
        

if __name__ == '__main__':
    n_controls = 4
    bsize = 128
    marginals = torch.tensor([.1, .4, .35, .15])
    sv = Solver(n_controls, bsize, marginals)

    losses = abs(torch.randn(bsize))
    weights = torch.tensor([0.25, 0.2, 0.4, 0.25])

    g_hat = sv.cvxsolve(losses, weights)
    print('g_hat', g_hat)
    print('rmean', g_hat.mean(0))
    
    assert (g_hat >= 0.).all()
    #assert (g_hat.sum(1) == 1.).all()
