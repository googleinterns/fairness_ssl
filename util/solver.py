import cvxpy as cp
import numpy as np

import pdb

class Solver(object):
    def __init__(self):
        pass

    def cvxsolve(self, losses, weights):
        
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


if __name__ == '__main__':
    t = Solver()
