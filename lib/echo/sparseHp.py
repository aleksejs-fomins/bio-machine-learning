import numpy as np
import lib.plasticity.hp as hplib

class echoSimpleHp:
    def __init__(self, p):
        # Parameters
        self.p = {
            'nNode'     : p['nNode'],
            'etaX'      : p['dt'] / p['tauX'],
            'etaHpBeta' : p['dt'] / p['tauHpBeta'],
            'hpMax'     : p['hpMax'],
            'hpXMean'   : p['hpXMean']
        }

        # Variables
        self.hpBeta = np.random.uniform(0, 1, p['nNode'])
        self.x = np.random.uniform(0, 1, p['nNode'])
        self.W = np.random.uniform(0, 1, (p['nNode'], p['nNode']))
        self.W *= np.random.uniform(0, 1, (p['nNode'], p['nNode'])) < p['freqConn']
        self.W[np.eye(p['nNode'], dtype=bool)] = 0

        self.logX = []
        self.logW = []

    def rhs(self, x, W, beta):
        hpAlpha = hplib.sig_relu_asymm(beta, self.p['hpMax'])
        WEff = W * np.outer(hpAlpha, np.ones(self.p['nNode']))

        rhsX = -x + WEff.dot(x)
        rhsBeta = hplib.rhs_sig_relu_asymm(beta, x, self.p['hpXMean'])
        return rhsX, rhsBeta, WEff

    def step(self, nStep, log=False):
        for iStep in range(nStep):
            rhsX, rhsHpBeta, WEff = self.rhs(self.x, self.W, self.hpBeta)

            self.x += self.p['etaX'] * rhsX
            self.hpBeta += self.p['etaHpBeta'] * rhsHpBeta

            if log:
                M = (1 - self.p['etaX']) * np.eye(self.p['nNode']) + self.p['etaX'] * WEff

                self.logX += [np.copy(self.x)]
                self.logW += [np.linalg.eigvals(M)]
