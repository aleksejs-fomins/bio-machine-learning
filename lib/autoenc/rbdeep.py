import numpy as np

class RaoBallardDeep:
    def __init__(self, p):
        self.nLayer = len(p['nNode'])

        self.etaNeuro = p['dt'] / p['tauX']
        self.etaSP    = p['dt'] / p['tauU']

        self.func = p['nonlinFunc']
        self.fprim = p['nonlinPrimFunc']

        # Values of hidden layers
        self.r = []
        for i in range(self.nLayer):
            self.r += [np.random.uniform(0.9, 1, p['nNode'][i])]

        # Values of matrices
        self.u = []
        for i in range(1, self.nLayer):
            uThis = np.random.uniform(0, 1, (p['nNode'][i], p['nNode'][i - 1]))
            self.u += [uThis / np.linalg.norm(uThis)]

        # Storage for temporarily logged variables
        self.log = []


    def err(self, i, x, r, u, func, fprim):
        rEff = x if i == 0 else r[i - 1]
        lin = u[i].dot(r[i])
        f = func[i](lin)
        g = fprim[i](lin)
        err = rEff - f
        errg = err * g
        return err, errg

    def rhsR(self, err, errgLst, u, i):
        # rhs due to feedforward propagation
        rhsR = u[i].T.dot(errgLst[i])

        # rhs due to feedback propagation
        if i != self.nLayer - 1:
            rhsR -= err[i+1]

        return rhsR


    # Update system by presenting input x for time nt
    def step(self, x, nt, withSP, withLog):
        for it in range(nt):
            # Compute linear and nonlinear prediction errors
            errLst = []
            errgLst = []
            for i, r, u, func, fprim in enumerate(self.r, self.u, self.func, self.fprim):
                err, errg = self.err(i, x, r, u, func, fprim)
                errLst += [err]
                errgLst += [errg]

            # Construct right hand sides
            rhsR = [self.rhsR(errLst, errgLst, self.u, i) for i in range(self.nLayer)]

            # Synaptic Plasticity
            if withSP:
                rhsU = [np.outer(errgLst[i], self.r[i]) for i in range(self.nLayer)]

            # Update
            for i in range(self.nLayer):
                self.r[i] += self.etaNeuro * rhsR[i]
                if withSP:
                    self.u[i] += self.etaSP * rhsU[i]

            # Log variables if requested
            if withLog:
                log = (self.r, errLst)
                if withSP:
                    log += ([np.linalg.norm(u) for u in self.u], )
                self.log += [log]
