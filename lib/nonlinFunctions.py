import numpy as np

# Some auxiliary functions
outer_same = lambda x: np.outer(x, x)

# Some nonlinear functions
func_sig       = lambda x: 1 / (1 + np.exp(-x))
func_relu      = lambda x: x*(x > 0)
func_leakyrelu = lambda x: x*((x > 0) + 0.01*(x < 0))

def func_softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

# And their derivatives
fprim_sig       = lambda x: 0.5 / (1 + np.cosh(x))
fprim_relu      = lambda x: (x > 0).astype(float)
fprim_leakyrelu = lambda x: (x > 0) + 0.01*(x < 0)
fprim_softmax   = lambda x: np.diag(x) - outer_same(x)