import numpy as np

from lib.echo.sparseHp import echoSimpleHp  # Echo state network
import lib.vis.movie1d as movie1d           # Plot movies of activity

param = {
    'nNode'        : 100,
    'freqConn'     : 0.3,
    'dt'           : 0.1,
    'tauX'         : 2,
    'tauHpBeta'    : 10,
    'hpMax'        : 1.0,
    'hpXMean'      : 1.0
}

inp = np.random.uniform(0, 1, param['nNode'])

# Run model
model = echoSimpleHp(param)
model.step(500, inp=inp, log=True)

# Plot activities
movie1d.time_bars(np.array(model.logX), "test_bars.avi", logy=True)

# Plot matrix values
eigsReal = np.array([np.real(e) for e in model.logW])
eigsImag = np.array([np.imag(e) for e in model.logW])
movie1d.time_scatter(eigsReal, eigsImag, "test_eigs.avi")