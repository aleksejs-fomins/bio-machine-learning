import numpy as np
from lib.nonlinFunctions import func_step

class echoSimpleHp:
    def __init__(self, p):
        # Parameters
        self.p = {
            'nNode'     : p['nNode'],
            'xMax'      : p['xMax'],
            'etaX'      : p['dt'] / p['tauX'],
            'etaHpBeta' : p['dt'] / p['tauHpBeta'],
            'hpMax'     : p['hpMax'],
            'hpXMean'   : p['hpXMean']
        }

        # Variables
        self.hpBeta = np.random.uniform(0, 1, p['nNode'])
        #self.x = np.random.uniform(0, 1, p['nNode'])
        self.x = np.zeros(p['nNode'])
        self.W = np.random.uniform(0, 1, (p['nNode'], p['nNode']))
        self.W *= np.random.uniform(0, 1, (p['nNode'], p['nNode'])) < p['freqConn']
        self.W[np.eye(p['nNode'], dtype=bool)] = 0

        self.logX = []
        self.logW = []

    def rhs(self, x, W, hpBeta, inp):
        # Threshold:
        #   linear in range [0,1], clipped outside
        hpAlpha = self.p['hpMax'] * np.clip(hpBeta, 0, 1)
        WEff = W * np.outer(hpAlpha, np.ones(self.p['nNode']))
        rhsX = -x + WEff.dot(x)

        if inp is not None:
            rhsX += inp

        # Threshold Homeostatic Plasticity:
        #   If activity below fixed mean, grow beta at constant rate until reaches 1, then set derivative to 0
        #   If activity above fixed mean, shrink beta at constant rate until reaches 0, then set derivative to 0
        hpGamma = 1 - x / self.p['hpXMean']
        rhsBeta = func_step(hpGamma) * func_step(1 - hpBeta) - func_step(-hpGamma) * func_step(hpBeta)

        return rhsX, rhsBeta, WEff

    def step(self, nStep, inp=None, log=False):
        for iStep in range(nStep):
            rhsX, rhsHpBeta, WEff = self.rhs(self.x, self.W, self.hpBeta, inp)

            self.x      += self.p['etaX'] * rhsX
            self.hpBeta += self.p['etaHpBeta'] * rhsHpBeta

            # Truncate x to available range
            self.x[self.x > self.p['xMax']] = self.p['xMax']

            if log:
                M = (1 - self.p['etaX']) * np.eye(self.p['nNode']) + self.p['etaX'] * WEff

                self.logX += [np.copy(self.x)]
                self.logW += [np.linalg.eigvals(M)]
