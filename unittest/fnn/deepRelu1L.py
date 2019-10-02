from lib.data import mnist
from lib.fnn.deepNet import DeepNet
from lib.fnn.performance import trainTestNetwork
import lib.nonlinFunctions as nonlinFunctions

# Load data
dataMNIST = mnist.load("/home/aleksejs/Downloads/mnist_data/")

relu = nonlinFunctions.func_relu
relup = nonlinFunctions.fprim_relu

# Set parameters
param = {
    # Network parameters
    'netClass'       : DeepNet,
    'ny'             : 10,                          # Number of possible outputs (for binarization)
    'nHid'           : [30],                        # no hidden layers
    'nonlinFunc'     : [relu, relu],
    'nonlinPrimFunc' : [relup, relup],
    'bSTD'           : 0, #10**-4,                     # variance of initial thresholds
    'wSTD'           : 10**-4,                     # variance of initial weights

    # Testing parameters
    'etaPref'        : 10**-2,                      # Prefactor for learning rate
    'nEpoch'         : 100,                         # Number of times to sweep the entire data
    'nMini'          : 32,                          # Number of datapoints per minibatch
}

trainTestNetwork(*dataMNIST.values(), param)