import numpy as np
import matplotlib.pyplot as plt
from lib.data import mnist
from lib.autoenc.rbdeep import RaoBallardDeep
import lib.nonlinFunctions as nonlinFunctions

# Load data
dataMNIST = mnist.load("/home/alyosha/Downloads/mnist_data/")
nData = 5
dataIdxs = np.arange(nData)
xarr = dataMNIST['train_images'][dataIdxs].astype(float)
nPix = xarr.shape[1] * xarr.shape[2]
for i in range(nData):
    xarr[i] /= np.linalg.norm(xarr[i])

sig = nonlinFunctions.func_id
sigp = nonlinFunctions.fprim_id

# Specify parameters
param = {
    'nNode'             : [nPix, 5],
    'dt'                : 0.1,   # ms,  timestep
    'tauX'              : 1.0,   # ms,  neuronal timescale
    'tauU'              : 100.0, # ms,  plasticity timescale
    'inputNoise'        : 0.0,
    'uNorm'             : 1.0,
    'nonlinFunc'        : [sig, sig],
    'nonlinPrimFunc'    : [sigp, sigp]
}

# Run
rbThis = RaoBallardDeep(param)
noImage = np.zeros(nPix)
for iEpoch in range(10):
    print("Epoch Number", iEpoch)
    for iData in range(nData):
        rbThis.step(xarr[iData].flatten(), 2000, withSP=True, withLog=True)
        rbThis.step(noImage, 2000, withSP=False, withLog=True)

for iEpoch in range(10):
    print("Epoch Number", iEpoch)
    for iData in range(nData):
        rbThis.step(xarr[iData].flatten(), 2000, withSP=False, withLog=True)
        rbThis.step(noImage, 2000, withSP=False, withLog=True)

# Plot results
fig, ax = plt.subplots(ncols=3)
for iLayer in range(1):
    rThis = [np.linalg.norm(r[iLayer]) for r in rbThis.log['r']]
    errThis = [np.linalg.norm(err[iLayer]) for err in rbThis.log['err']]
    uNormThis = [uNorm[iLayer] for uNorm in rbThis.log['u']]

    ax[0].plot(rThis, label=str(iLayer))
    ax[1].plot(errThis, label=str(iLayer))
    ax[2].plot(uNormThis, label=str(iLayer))
ax[0].set_title("r")
ax[1].set_title("err")
ax[2].set_title("u")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()