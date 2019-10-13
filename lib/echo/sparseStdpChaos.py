import numpy as np
from lib.nonlinFunctions import func_step, outer_same

class echoSimpleHp:
    def __init__(self, p):
        # Parameters
        self.p = {
            'nNode'         : p['nNode'],
            'etaX'          : p['dt'] / p['tauX'],
            'etaXmu'        : p['dt'] / p['tauXmu'],
            'etaXStd'       : p['dt'] / p['tauXStd'],
            'etaHP'         : p['dt'] / p['tauHP'],
            'etaSTDP'       : p['dt'] / p['tauSTDP'],
            'hpMax'         : p['hpMax'],
            'hpXMean'       : p['hpXMean'],
            'wMax'          : p['wMax'],
            'stdpCorrMean'  : p['stdpCorrMean'],

        }

        # Variables
        #self.x = np.random.uniform(0, 1, p['nNode'])
        self.x = np.zeros(p['nNode'])
        self.thetaW = np.random.uniform(0, 1, (p['nNode'], p['nNode']))
        self.thetaW *= np.random.uniform(0, 1, (p['nNode'], p['nNode'])) < p['freqConn']
        self.thetaW[np.eye(p['nNode'], dtype=bool)] = 0

        # HP
        self.thetaThr = np.random.uniform(0, 1, p['nNode'])

        # STDP
        self.xMu = np.zeros(p['nNode'])
        self.xStd  = np.ones(p['nNode'])

        self.logX = []
        self.logW = []

    def rhs(self, x, xMu, xStd, thetaThr, thetaW, inp):
        # Threshold:
        #   linear in range [0,1], clipped outside
        thr = self.p['hpMax'] * np.clip(thetaThr, 0, 1)
        W = self.p['wMax'] * np.clip(thetaW, 0, 1)

        WEff = W * np.outer(thr, np.ones(self.p['nNode']))
        rhsX = -x + WEff.dot(x)

        if inp is not None:
            rhsX += inp

        # Threshold Homeostatic Plasticity:
        #   If activity below fixed mean, grow beta at constant rate until reaches 1, then set derivative to 0
        #   If activity above fixed mean, shrink beta at constant rate until reaches 0, then set derivative to 0
        gammaHp = 1 - x / self.p['hpXMean']
        rhsThetaThr = func_step(gammaHp) * func_step(1 - thetaThr) - func_step(-gammaHp) * func_step(thetaThr)

        # Correlation STDP
        rhsXMu = - xMu + x
        rhsXStd = - xStd + (x-xMu)**2
        zetaStdp = (x - rhsXMu) / rhsXStd
        corrStdp = np.outer(zetaStdp, zetaStdp)
        gammaStdp = 1 - corrStdp / self.p['stdpCorrMean']
        rhsThetaW = func_step(gammaStdp) * func_step(1 - thetaW) - func_step(-gammaStdp) * func_step(thetaW)

        return rhsX, rhsXMu, rhsXStd, rhsThetaThr, rhsThetaW, WEff

    def step(self, nStep, inp=None, log=False):
        for iStep in range(nStep):
            rhsX, rhsXMu, rhsXStd, rhsThetaThr, rhsThetaW, WEff = self.rhs(self.x, self.xMu, self.xStd, self.thetaThr, self.thetaW, inp)

            self.x         += self.p['etaX'] * rhsX
            self.xMu       += self.p['etaXMu'] * rhsXMu
            self.xStd      += self.p['etaXStd'] * rhsXStd
            self.thetaThr  += self.p['etaHP'] * rhsThetaThr
            self.thetaW    += self.p['etaSTDP'] * rhsThetaW

            if log:
                M = (1 - self.p['etaX']) * np.eye(self.p['nNode']) + self.p['etaX'] * WEff

                self.logX += [np.copy(self.x)]
                self.logW += [np.linalg.eigvals(M)]
