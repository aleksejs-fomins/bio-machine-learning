import numpy as np
import matplotlib.pyplot as plt
from lib.neuro.ratelib import RateSimulator

# Constants
nNeuron = 100

# Initialization
rs = RateSimulator(0.1, 5, 20)

# Populations
rate_rhs = lambda x, i, tau: (-x + i) / tau
rs.add_population("P1", nNeuron, rate_rhs, initStrat="zero")

# Inputs
inputMask = (np.random.uniform(0,1,nNeuron) < 0.1).astype(int)
inp_func = lambda t: inputMask * np.sin(t / 5)
rs.add_input("I1", "P1", inp_func)

# Watchers
rs.add_pop_watcher("rawP1", "P1", "norm")

# Run
rs.run(100)

# Plot results
times, normsP1 = rs.get_pop_watcher_results("rawP1")
_, normsI1 = rs.get_inp_values("I1", "norm")

plt.figure()
plt.plot(times, normsP1, label='P1')
plt.plot(times, normsI1, label='I1')
plt.legend()
plt.show()