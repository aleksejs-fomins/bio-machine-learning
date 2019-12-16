from time import time
import numpy as np
from lib import randstat, bitwise

class HopfieldNetwork():
    def __init__(self, nNeuron, rand=True, thresh = 1.0):
        # Constants
        self.nNeuron = nNeuron
        self.nCombination = 2**nNeuron
        self.nWeight = nNeuron * (nNeuron-1) // 2
        self.minWeight = -1.0
        self.maxWeight = 1.0

        #Generate a random array of weights and thresholds
        randstat.init()

        self.setThresh(thresh)
        if rand:
            # self.weights = np.random.random_integers(-1, 1, self.nWeight)
            # self.weights = np.random.uniform(self.minWeight, self.maxWeight, self.nWeight)
            self.weights = np.zeros((self.nNeuron, self.nNeuron))
            for i in range(self.nNeuron):
                for j in range(i+1, self.nNeuron):
                    self.weights[i][j] = np.random.uniform(self.minWeight, self.maxWeight)
        else:
            self.weights = np.triu(np.full((self.nNeuron, self.nNeuron), 1.0), 1)


    def setThresh(self, thresh):
        self.thresholds = np.full(self.nNeuron, thresh)


    # Calculate energy of a given configuration of the Hopfield network
    def calcEnergy(self, combination):
        # Activation is either -1 or +1
        activation = bitwise.bitlist2activation(combination)

        # print("act", activation)
        # print("dot", np.dot(self.weights, activation))

        nrg = 0.0
        nrg -= np.dot(self.thresholds, activation)
        nrg -= np.dot(activation, np.dot(self.weights, activation))

        # for i in range(self.nNeuron):
        #     for j in range(i+1, self.nNeuron):
        #         if activation[i] == activation[j]:
        #             nrg -= self.weights[i][j] * activation[j]

        return nrg


    # Find all local minima of this network
    def getMinima(self):
        # 1) Loop over a list of all combinations and calculate energy for each combination
        comb2nrg = {}
        for i in range(self.nCombination):
            comb2nrg[i] = self.calcEnergy(bitwise.bitlist(i, self.nNeuron))

        # 2) Loop over a list of all combinations and check if each combination is a local minimum
        comb2nrgMin = {}
        for i, nrg in comb2nrg.items():

            idx1 = 0
            while idx1 < self.nNeuron:
                nrg2 = comb2nrg[bitwise.bitflip(i, idx1)]
                if nrg2 < nrg:
                    break
                idx1 += 1

            if idx1 == self.nNeuron:
                comb2nrgMin[tuple(bitwise.bitlist(i, self.nNeuron))] = nrg

        return comb2nrgMin


    # Learn a set of patterns at the same time, overwriting previous memory
    def learnBulk(self, memList):
        #self.weights.fill(-len(memList))
        self.weights.fill(0)
        for v in memList:
            a = bitwise.bitlist2activation(v)
            self.weights += np.outer(a, a)#*2
        self.weights /= len(memList)
        self.weights = np.triu(self.weights, 1)

        #print(self.weights)
