from datetime import datetime
import random
import numpy as np
import numpy.polynomial.polynomial as nppoly

# Init Random Seed
def init():
    random.seed(datetime.now())
    np.random.seed() #datetime.now().toordinal()


# Generate a random polynomial P(x) of order M defined on [0,1]
def URandomPoly(dim, argmax):
    coefficients = [random.uniform(-argmax, argmax) for i in range(0, dim+1)]
    return nppoly.Polynomial(coef=tuple(coefficients))


# Generate a sample Y_i = P(X_i) + e_i, where e_i ~ N(0, s)
def samplePoly(poly, nSample, err = 0.0, left = 0.0, right = 1.0):
    xList = np.linspace(left, right, nSample)
    if err == 0.0:
        yList = poly(xList)
    else:
        yList = [poly(x) + np.random.normal(0, err) for x in xList]

    return xList, yList
