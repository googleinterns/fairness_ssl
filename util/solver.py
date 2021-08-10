import cvxpy as cp
import numpy as np
import torch

import pdb

class Solver(object):
    def eval_nearestnbhs(self, data):
        # TODO: nearest neighbours here
        return None

    def cvxsolve(self, losses, weights, Gamma_g):
        return torch.randn(losses.shape[0], 4).cuda()
        
        '''
        m = 15
        n = 10
        np.random.seed(1)
        s0 = np.random.randn(m)
        lamb0 = np.maximum(-s0, 0)
        s0 = np.maximum(s0, 0)
        x0 = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A @ x0 + s0
        c = -A.T @ lamb0
        
        # Define and solve the CVXPY problem.
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c.T@x),
                          [A @ x <= b])
        prob.solve()
        
        return x.value
        '''

if __name__ == '__main__':
    t = Solver()
