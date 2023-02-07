import numpy as np

from lib.echo.rate import EchoRate  # Echo state network
import lib.vis.movie1d as movie1d   # Plot movies of activity

param = {
    'nNode'    : 100,
    'freqConn' : 0.3,
    'dt'       : 0.1,
    'tauX'     : 2
}

# Run model
model = EchoRate(param)
model.step(20, log=True)

# Plot activities
movie1d.time_bars(np.array(model.logX), "test_bars.avi")

# Plot matrix values
eigsReal = np.array([np.real(e) for e in model.logW])
eigsImag = np.array([np.imag(e) for e in model.logW])
movie1d.time_scatter(eigsReal, eigsImag, "test_eigs.avi")