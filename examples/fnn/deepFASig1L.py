from lib.data import mnist
from lib.fnn.deepNetFA import DeepNetFeedbackAlignment
from lib.fnn.performance import trainTestNetwork
import lib.nonlinFunctions as nonlinFunctions

# Load data
dataMNIST = mnist.load("/home/aleksejs/Downloads/mnist_data/")

sig  = nonlinFunctions.func_sig
sigp = nonlinFunctions.fprim_sig

# Set parameters
param = {
    # Network parameters
    'netClass'       : DeepNetFeedbackAlignment,
    'ny'             : 10,                          # Number of possible outputs (for binarization)
    'nHid'           : [30],                        # no hidden layers
    'nonlinFunc'     : [sig, sig],
    'nonlinPrimFunc' : [sigp, sigp],
    'bSTD'           : 10**-4,                         # variance of initial thresholds
    'wSTD'           : 10**-4,                         # variance of initial weights

    # Testing parameters
    'etaPref'        : 0.1,                         # Prefactor for learning rate
    'nEpoch'         : 100,                         # Number of times to sweep the entire data
    'nMini'          : 32,                          # Number of datapoints per minibatch
}

trainTestNetwork(*dataMNIST.values(), param)