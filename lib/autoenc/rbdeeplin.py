import numpy as np

class RaoBallardDeepLinear:
    def __init__(self, p):
        self.nLayer = len(p['nNode'])

        self.etaH = p['dt'] / p['tauX']
        self.etaU = p['dt'] / p['tauU']

        self.h = []   # Values of hidden layers
        self.u = []   # Values of matrices
        for i in range(self.nLayer):
            self.h += [np.random.uniform(0.9, 1, p['nNode'][i])]

        for i in range(1, self.nLayer):
            uThis = np.random.uniform(0, 1, (p['nNode'][i], p['nNode'][i - 1]))
            self.u += [uThis / np.linalg.norm(uThis)]

        # Storage for temporarily logged variables
        self.log = []


    def err(self, x, h, u, i):
        inpThis = x if i == 0 else h[i - 1]         # Efffective input
        return inpThis - u[i].dot(h[i])


    def rhsX(self, err, h, u, i):
        # rhs due to feedforward propagation
        rhsX = u[i].T.dot(err)

        # rhs due to feedback propagation
        if i != self.nLayer - 1:
            rhsX += -h[i] + u[i + 1].dot(h[i + 1])

        return rhsX


    # Update system by presenting input x for time nt
    def step(self, x, nt, withSP, withLog):
        for it in range(nt):

            # Construct right hand sides
            err = [self.err(x, self.h, self.u, i) for i in range(self.nLayer)]
            rhsX = [self.rhsX(err, self.h, self.u, i) for i in range(self.nLayer)]

            # Synaptic Plasticity
            if withSP:
                rhsU = [np.outer(err[i], self.h[i]) for i in range(self.nLayer)]

            # Update
            for i in range(self.nLayer):
                self.h[i] += self.etaH * rhsX
                if withSP:
                    self.u[i] += self.etaU * rhsU

            # Log variables if requested
            if withLog:
                log = (self.h, err)
                if withSP:
                    log += ([np.linalg.norm(u) for u in self.u], )
                self.log += [log]
