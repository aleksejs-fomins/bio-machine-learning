import numpy as np
import matplotlib.pyplot as plt

# Convert labels to array of ones and zeos
def label2arr(x, N):
    arr = np.zeros(N)
    arr[x] = 1
    return arr

# Compute accuracy - how frequently the largest predicted number coincides with the true label
def computeAccuracy(yTrue1D, yPredicted2D):
    yPredicted1D = np.array([np.argmax(out) for out in yPredicted2D])
    return np.sum(yPredicted1D == yTrue1D) / yTrue1D.shape[0]

# Minibatch training algorithm
def trainTestNetwork(xTrain, yTrain, xTest, yTest, param):
    nTrialsTrain = xTrain.shape[0]
    nTrialsTest  = xTest.shape[0]
    nFeatures    = np.prod(xTrain.shape[1:])   # If data features have more than 1D (e.g. images), need to flatten them
    ny           = param['ny']                 # Number of possible outcomes

    # Convert data to format compatible with network [Trials x Features]
    xTrain2D  = xTrain.reshape(nTrialsTrain, nFeatures)
    xTest2D   = xTest.reshape(nTrialsTest, nFeatures)
    yTrain2D  = np.array([label2arr(y, ny) for y in yTrain])  # Binarize labels

    # Initialize network
    netParam = {
        'nNode'             : [nFeatures] + param['nHid'] + [ny],
        'nonlinFunc'        : param['nonlinFunc'],
        'nonlinPrimFunc'    : param['nonlinPrimFunc'],
        'bSTD'              : param['bSTD'],
        'wSTD'              : param['wSTD']
    }
    net = param['netClass'](netParam)

    # Train network, trackign accuracy
    lossProgress = []
    trainAccuracy = np.zeros(param['nEpoch'])
    testAccuracy  = np.zeros(param['nEpoch'])
    nMiniBatches  = int(np.ceil(nTrialsTrain / param['nMini']))

    for iEpoch in range(param['nEpoch']):
        eta = param['etaPref'] #(0.01 if iEpoch < 20 else 0.001) * param['etaPref']
        # eta = 0.01 if iEpoch < 20 else 0.001
        for iMiniBatch in range(nMiniBatches):
            idxl = param['nMini'] * iMiniBatch
            idxr = param['nMini'] * (iMiniBatch + 1)

            lossProgress += [net.step(xTrain2D[idxl:idxr], yTrain2D[idxl:idxr], eta)]

        trainAccuracy[iEpoch] = computeAccuracy(yTrain, net.predict(xTrain2D))
        testAccuracy[iEpoch]  = computeAccuracy(yTest, net.predict(xTest2D))
        print("Did epoch", iEpoch,
              "::: Train accuracy", np.round(trainAccuracy[iEpoch], 2),
              "::: Test accuracy", np.round(testAccuracy[iEpoch], 2))

    # Plot results
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].set_title("Loss function")
    ax[1].set_title("Accuracy")
    ax[0].set_xlabel("miniBatch index")
    ax[1].set_xlabel("Epoch")
    ax[0].plot(lossProgress)
    ax[1].plot(trainAccuracy, label='train')
    ax[1].plot(testAccuracy, label='test')
    ax[1].legend()
    plt.show()