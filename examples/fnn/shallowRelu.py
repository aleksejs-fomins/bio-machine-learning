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
    'nonlinFunc'     : nonlinFunctions.func_relu,
    'nonlinPrimFunc' : nonlinFunctions.fprim_relu,
    'bSTD'           : 10**-4,                     # variance of initial thresholds
    'wSTD'           : 10**-4,                     # variance of initial weights

    # Testing parameters
    'etaPref'        : 10**-8,                      # Prefactor for learning rate
    'nEpoch'         : 100,                         # Number of times to sweep the entire data
    'nMini'          : 32,                          # Number of datapoints per minibatch
}

trainTestNetwork(*dataMNIST.values(), param)