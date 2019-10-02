from lib.data import mnist
from lib.fnn.deepNet import DeepNet
from lib.fnn.performance import trainTestNetwork
import lib.nonlinFunctions as nonlinFunctions

# Load data
dataMNIST = mnist.load("/home/aleksejs/Downloads/mnist_data/")

sig  = nonlinFunctions.func_sig
sigp = nonlinFunctions.fprim_sig

# Set parameters
param = {
    # Network parameters
    'netClass'       : DeepNet,
    'ny'             : 10,                          # Number of possible outputs (for binarization)
    'nHid'           : [50, 30],                    # no hidden layers
    'nonlinFunc'     : [sig, sig, sig],
    'nonlinPrimFunc' : [sigp, sigp, sigp],
    'bSTD'           : 1.0,                         # variance of initial thresholds
    'wSTD'           : 1.0,                         # variance of initial weights

    # Testing parameters
    'etaPref'        : 0.001,                       # Prefactor for learning rate
    'nEpoch'         : 100,                         # Number of times to sweep the entire data
    'nMini'          : 32,                          # Number of datapoints per minibatch
}

trainTestNetwork(*dataMNIST.values(), param)