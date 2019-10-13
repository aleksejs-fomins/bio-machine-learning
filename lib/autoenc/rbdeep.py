import numpy as np

class RaoBallardDeep:
    def __init__(self, p):
        self.nLayer = len(p['nNode']) - 1

        self.etaNeuro = p['dt'] / p['tauX']
        self.etaSP    = p['dt'] / p['tauU']
        self.nu       = p['inputNoise']

        self.func = p['nonlinFunc']
        self.fprim = p['nonlinPrimFunc']

        # Initialize hidden layers and matrices
        self.r = []
        self.u = []
        for i in range(self.nLayer):
            self.r += [np.random.uniform(0, 1, p['nNode'][i + 1])]
            uThis  = np.random.uniform(0, 1, (p['nNode'][i], p['nNode'][i + 1]))  # Note in RB matrices are feedback-oriented
            self.u += [p['uNorm'] * uThis / np.linalg.norm(uThis)]

        # Storage for temporarily logged variables
        self.log = {"r" : [], "err" : [], "u" : []}


    def err(self, rEff, r, u, func, fprim):
        lin = u.dot(r)
        err = rEff - func(lin)
        errg = err * fprim(lin)
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
            # Add a bit of noise to input at every time step
            inpThis = x + self.nu * np.random.normal(0, 1, x.shape[0])

            # Compute linear and nonlinear prediction errors
            errLst = []
            errgLst = []
            rEffLst = [inpThis] + self.r[:-1]

            for (rEff, r, u, func, fprim) in zip(rEffLst, self.r, self.u, self.func, self.fprim):
                err, errg = self.err(rEff, r, u, func, fprim)
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
                    self.u[i] = np.clip(self.u[i], 0, 100)   # synapses can not be negative

            # Log variables if requested
            if withLog:
                self.log['r'] += [[np.copy(r) for r in self.r]]
                self.log['err'] += [errLst]
                if withSP:
                    self.log['u'] += [[np.linalg.norm(u) for u in self.u]]
