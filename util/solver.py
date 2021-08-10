import cvxpy as cp
import numpy as np
import torch

import pdb

class Solver(object):
    def __init__(self, n_controls, bsize, marginals):
        self.X = cp.Variable((bsize, n_controls))
        self.l = cp.Parameter((bsize, 1))
        self.p = cp.Parameter((n_controls, 1), value=marginals)
        self.q = cp.Parameter(n_controls)
        
        counts = self.X.sum(0)
        denom = counts + (counts==0).float()
        obj = (self.l.T @ self.X) @ self.q
        constraints = [self.X >= 0,
                       cv.sum(self.X, axis=1) == torch.ones(bsize, 1).cuda(),
                       cv.sum(self.X, axis=0) == marginals.T]

        self.prob = cp.Problem(obj, constraints)

    def cvxsolve(self, losses, weights, Gamma_g):
        self.l.value = losses
        self.q.value = weights

        self.prob.solve()

        return self.X.value
    
        
    def eval_nearestnbhs(self, data):
        # TODO: nearest neighbours here
        return None
        

if __name__ == '__main__':
    t = Solver()
