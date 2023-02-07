import numpy as np
import matplotlib.pyplot as plt

from lib.echo.rateSTDPChaos import EchoRateSTDPChaos  # Echo state network
import lib.vis.movie1d as movie1d           # Plot movies of activity

param = {
    'nNode'        : 100,
    'freqConn'     : 0.3,
    'xMax'         : 400,  # Hz, max firing rate
    'dt'           : 0.1,
    'tauX'         : 2,
    'tauXMu'       : 20,
    'tauHP'        : 100,
    'tauSTDP'      : 20,
    'hpMax'        : 1.0,
    'hpXMean'      : 10.0,
    'wMax'         : 10.0,
    'stdpCorrMean' : 0.2,
}

inp = np.random.uniform(0, 1, param['nNode'])

# Run model
nStep = 10000
model = EchoRateSTDPChaos(param)
model.step(nStep, inp=inp, log=True)


muS2 = [np.mean(xS2) for xS2 in model.logXS2]
muZeta = [np.mean(zeta) for zeta in model.logZetaStdp]

window = 100
corrLst = [np.outer(zeta, zeta)[~np.eye(param['nNode'], dtype=bool)] for zeta in model.logZetaStdp]
muCorr = [np.mean(corrLst[i:i+window]) for i in range(nStep - window)]

fig, ax = plt.subplots(ncols = 3)
ax[0].plot(np.arange(nStep), muS2)
ax[0].set_yscale('log')
ax[1].plot(np.arange(nStep), muZeta)
ax[1].set_yscale('log')
ax[2].plot(np.arange(nStep-window), muCorr)
ax[2].set_yscale('log')

nStepThis = 100
model.step(nStepThis, inp=inp, log=True)

# Plot activities
movie1d.time_bars(np.array(model.logX[-nStepThis:]), "test_bars.avi", logy=True)

# Plot matrix values
eigsReal = np.array([np.real(e) for e in model.logW[-nStepThis:]])
eigsImag = np.array([np.imag(e) for e in model.logW[-nStepThis:]])
movie1d.time_scatter(eigsReal, eigsImag, "test_eigs.avi")

plt.show()