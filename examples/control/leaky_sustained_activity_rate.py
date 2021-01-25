import numpy as np
import matplotlib.pyplot as plt

'''
Simulate "slider" differential equation     tau * [dx/dt] = I(t)

Using leaky units of intrinsic equation     tau * [dx/dt] = -x(t) + I(t)

Compensate leak with recurrent connectivity tau * [dx/dt] = -x(t) + I(t) + Mx(t)
'''

# Equation of Motion
rhs = lambda x, I, M, tau: (-x + I + M.dot(x)) / tau

# Timescale
T = 20000    # ms, simulation time
dt = 0.1    # ms, timestep
tau = 5     # ms, leak timescale

# Init
nNeuron = 2
x = np.random.uniform(0,1,nNeuron)
M = (np.ones((nNeuron, nNeuron)) - np.eye(nNeuron)) / (nNeuron - 1)

# Input
tSwitch = 100  # ms, time to switch input
inpVals = [3, 0, -3, 0]
noiseStd = 1
inpFunc = lambda t: inpVals[int(t / tSwitch) % len(inpVals)] + np.random.normal(0, noiseStd, nNeuron)

# Simulate
xLst = []
iLst = []
times = np.arange(0, T, dt)
for t in times:
    inp = inpFunc(t)
    x += rhs(x, inp, M, tau) * dt

    iLst += [np.mean(inp)]
    xLst += [np.mean(x)]

plt.figure()
plt.plot(times, iLst, label='inp')
plt.plot(times, xLst, label='val')
plt.legend()
plt.show()