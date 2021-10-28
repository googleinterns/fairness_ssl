import cvxpy as cp
import numpy as np
import torch

import pdb

TOL = 1e-6

import cvxpy as cp
import numpy as np

class Solver(object):
    def __init__(self, n_controls, bsize, marginals, epsilon):
        """Group assignment solver.
        
        Arguments:
        n_controls: An integer for the number of groups.
        bsize: An integer for the batch size.
        marginals: A 1D array for the marginal distribution.
        epsilon: A float for the variance.
        """
        marginals = np.array(marginals)[..., np.newaxis]
        
        self.X = cp.Variable((bsize, n_controls))
        self.l = cp.Parameter((bsize, 1))
        self.p = cp.Parameter((n_controls, 1), value=marginals)
        self.q = cp.Parameter(n_controls)

        counts = cp.sum(self.X, axis=0, keepdims=True)

        obj = ((self.l.T @ self.X) / self.p.T) @ self.q
        constraints = [self.X >= 0,
                       cp.sum(self.X, axis=1, keepdims=True) == np.ones((bsize, 1)),
                       cp.abs(cp.sum(self.X, axis=0, keepdims=True) / bsize - self.p.T) <= epsilon]

        self.prob = cp.Problem(cp.Maximize(obj), constraints)

    def cvxsolve(self, losses, weights):
        """Solver.
        
        Arguments:
        losses: A 1D array for loss values.
        weights: A 1D array for group weights q.
        
        Returns:
        A 2D array for soft group assignments.
        """
        self.l.value = losses.data.cpu().numpy()[..., np.newaxis]
        self.q.value = weights.data.cpu().numpy()
        self.prob.solve()
        X = self.X.value

        X[abs(X) < TOL] = 0. # set very low value to zero

        # Numpy to Torch
        X = torch.from_numpy(X).float()
        return X
        
if __name__ == '__main__':
    n_controls = 4
    bsize = 128
    marginals = np.array([.1, .4, .35, .15])
    losses = np.random.rand(bsize, 1)
    weights = np.array([0.25, 0.21, 0.4, 0.24])

    sv = Solver(n_controls, bsize, marginals, epsilon=0.01)
    g_hat = sv.cvxsolve(losses, weights)

    print('g_hat', g_hat)
    print('rmean', g_hat.mean(0))
    print('g_hat.sum(1)', g_hat.sum(1))
    
    assert (g_hat >= 0.).all()
    assert np.isclose(g_hat.sum(1), 1.).all()

