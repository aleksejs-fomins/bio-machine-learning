import numpy as np

class EchoRate:
    def __init__(self, p):
        # Parameters
        self.etaX = p['dt'] / p['tauX']

        # Variables
        self.nNode = p['nNode']
        self.x = np.random.uniform(0, 1, p['nNode'])
        self.W = np.random.uniform(-1, 1, (p['nNode'], p['nNode']))
        self.W *= np.random.uniform(0, 1, (p['nNode'], p['nNode'])) < p['freqConn']
        self.W[np.eye(p['nNode'], dtype=bool)] = 0

        self.logX = []
        self.logW = []

    def rhsX(self, x, W):
        return -x + W.dot(x)

    def step(self, nStep, log=False):
        for iStep in range(nStep):
            self.x += self.etaX * self.rhsX(self.x, self.W)

            if log:
                M = (1 - self.etaX) * np.eye(self.nNode) + self.etaX * self.W

                self.logX += [np.copy(self.x)]
                self.logW += [np.linalg.eigvals(M)]
