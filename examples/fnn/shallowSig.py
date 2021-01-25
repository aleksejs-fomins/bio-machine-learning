from lib.data import mnist
from lib.fnn.shallowNet import ShallowNet
from lib.fnn.performance import trainTestNetwork
import lib.nonlinFunctions as nonlinFunctions

# Load data
dataMNIST = mnist.load("/home/aleksejs/Downloads/mnist_data/")

# Set parameters
param = {
    # Network parameters
    'netClass'       : ShallowNet,
    'ny'             : 10,                          # Number of possible outputs (for binarization)
    'nHid'           : [],                          # no hidden layers
    'nonlinFunc'     : nonlinFunctions.func_sig,
    'nonlinPrimFunc' : nonlinFunctions.fprim_sig,
    'bSTD'           : 1.0,                         # variance of initial thresholds
    'wSTD'           : 1.0,                         # variance of initial weights

    # Testing parameters
    'etaPref'        : 0.001,                       # Prefactor for learning rate
    'nEpoch'         : 100,                         # Number of times to sweep the entire data
    'nMini'          : 32,                          # Number of datapoints per minibatch
}

trainTestNetwork(*dataMNIST.values(), param)