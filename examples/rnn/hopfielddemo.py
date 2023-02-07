
'''
    What this does?
      Implement a tiny Hopfield network and find all its local minima using brute force

    Conclusions?
      1) HN works to some extent with real and even integer (-1,0,1) weights
      2) By definition, HN can not simultaneously store two patterns with difference in 1 bit (like 1000 and 1010)
      3) For every minimum its inverse is also a local minimum. Adding bias does not fix the problem
      4) HN has problems with memorizing non-orthogonal patterns. Starting with 3 patterns that are all non-orthogonal to each other some memories may disappear or fake memories appear.
         More precisely, if there is at least 50% intersect within a triplet, one of them is guaranteed to not be a minimum.

    TODO
      1) Are there other limitations? What is the exact storage limit? Is it possible to avoid false minima?

'''

import numpy as np
from lib.rnn.hopfieldlib import HopfieldNetwork

def printMinima(data):
    for comb, nrg in data.items():
        print(comb, nrg)

print("\nGenerating fixed network")
net1 = HopfieldNetwork(15, False)
printMinima(net1.getMinima())

print("\nGenerating random network")
net1 = HopfieldNetwork(15, True)
printMinima(net1.getMinima())

print("\nLearning patters in bulk")
data = [
    np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]),
    np.array([1,0,0,1,1,1,1,0,0,0,0,0,0,0,0]),
    np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]),
    #np.array([1,1,1,0,0,1,1,0,0,0,0,0,0,0,0])
    #np.array([1,1,1,0,0,0,0,0,0,0,0,0,1,1,1]),
    #np.array([0,0,0,1,1,0,0,1,1,1,0,0,0,0,0]),
    #np.array([0,0,0,0,0,1,1,0,0,0,1,1,0,0,0])

    # np.array([1,0,1]),
    # np.array([0,1,1])
    #np.array([1,0,0,0,1]),
    #np.array([1,1,1,0,0])
]
net1.setThresh(0)
net1.learnBulk(data)
printMinima(net1.getMinima())