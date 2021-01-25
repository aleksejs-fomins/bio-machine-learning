import numpy as np

from lib.echo.rateHP import EchoRateHp  # Echo state network
import lib.vis.movie1d as movie1d       # Plot movies of activity

param = {
    'nNode'        : 100,
    'freqConn'     : 0.3,
    'xMax'         : 400,  # Hz, max firing rate
    'dt'           : 0.1,
    'tauX'         : 2,
    'tauHpBeta'    : 20,
    'hpMax'        : 1.0,
    'hpXMean'      : 10.0
}

inp = np.random.uniform(0, 1, param['nNode'])

# Run model
model = EchoRateHp(param)
model.step(500, inp=inp, log=True)

# Plot activities
movie1d.time_bars(np.array(model.logX), "test_bars.avi", logy=True)

# Plot matrix values
eigsReal = np.array([np.real(e) for e in model.logW])
eigsImag = np.array([np.imag(e) for e in model.logW])
movie1d.time_scatter(eigsReal, eigsImag, "test_eigs.avi")