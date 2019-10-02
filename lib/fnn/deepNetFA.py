import numpy as np

class DeepNetFeedbackAlignment:
    def __init__(self, p):
        self.nLayer = len(p['nNode'])

        self.b = []
        self.W = []
        self.R = []
        for i in range(1, self.nLayer):
            self.b += [p['bSTD'] * np.random.normal(0, 1, p['nNode'][i])]
            self.W += [p['wSTD'] * np.random.normal(0, 1, (p['nNode'][i], p['nNode'][i - 1]))]
            self.R += [p['wSTD'] * np.random.normal(0, 1, (p['nNode'][i], p['nNode'][i - 1]))]
        self.func = p['nonlinFunc']
        self.fprim = p['nonlinPrimFunc']

    # Linear part of the feedforward step
    def lin(self, x, W, b):
        return x.dot(W.T) + b  # Flipped order to do vector broadcasting

    # Dimensions of x are [nTrial, nFeature]
    def predict(self, x):
        hThis = x
        for b, W, func in zip(self.b, self.W, self.func):
            lin = self.lin(hThis, W, b)
            hThis = func(lin)
        return hThis

    # Make single gradient descent step, given a bunch of data and associated labels
    def step(self, x, y, eta):
        # Forwards pass
        f = [x]  # Function values for different layers
        g = [1]  # Derivative function values for different layers
        for b, W, func, fprim in zip(self.b, self.W, self.func, self.fprim):
            lin = self.lin(f[-1], W, b)
            f += [func(lin)]
            g += [fprim(lin)]

        # Backwards pass
        err = [1] * (self.nLayer - 2) + [y - f[-1]]
        for iLayer in range(self.nLayer - 2, 0, -1):
            # print(iLayer, err[iLayer].shape, f[iLayer].shape, g[iLayer].shape)
            errg = err[iLayer] * g[iLayer + 1]
            err[iLayer - 1] = errg.dot(self.R[iLayer])
            self.b[iLayer] += eta * np.sum(errg, axis=0)
            self.W[iLayer] += eta * errg.T.dot(f[iLayer])

        return np.mean(np.linalg.norm(err[-1], axis=1))