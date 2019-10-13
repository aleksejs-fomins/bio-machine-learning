import numpy as np


# Heaviside step function
def H(x):
    return (x > 0).astype(int)


# Linear in range [0,1], clipped outside
def sig_relu_asymm(beta, hpMax):
    return hpMax * np.clip(beta, 0, 1)


# If activity below fixed mean, grow beta at constant rate until reaches 1, then set derivative to 0
# If activity above fixed mean, shrink beta at constant rate until reaches 0, then set derivative to 0
def rhs_sig_relu_asymm(beta, x, hpHMean):
    g = 1 - x / hpHMean
    return H(g)*H(1-beta) - H(-g)*H(beta)
