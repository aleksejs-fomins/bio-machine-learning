import numpy as np

class ShallowNet:
    def __init__(self, p):
        nIn, nOut = p['nNode']
        self.func = p['nonlinFunc']
        self.fprim = p['nonlinPrimFunc']
        self.b = np.random.normal(0, p['bSTD'], nOut)
        self.W = np.random.normal(0, p['wSTD'], (nOut, nIn))

    # Linear part of the predictor
    def lin(self, x, W, b):
        return x.dot(W.T) + b  # Flipped order to do vector broadcasting

    # Dimensions of x are [nTrial, nFeature]
    def predict(self, x):
        lin = self.lin(x, self.W, self.b)
        return self.func(lin)

    def step(self, x, y, eta):
        lin = self.lin(x, self.W, self.b)
        f = self.func(lin)  # Nonlinear function
        g = self.fprim(lin)  # It's derivative

        err = y - f
        errg = err * g
        self.b += eta * np.sum(errg, axis=0)
        self.W += eta * errg.T.dot(x)
        return np.mean(np.linalg.norm(err, axis=1))