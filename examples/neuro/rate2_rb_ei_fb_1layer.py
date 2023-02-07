import numpy as np
import matplotlib.pyplot as plt
from lib.neuro.ratelib import RateSimulator

# Constants
nNeuronInp = 100
nNeuronHid = 4
T = 10000         # ms, duration of simulation

# Initialization
rs = RateSimulator(dt=0.2, tauPop=1, tauSyn=100)

# Populations
rate_rhs = lambda x, i, tau: (-x + i) / tau
rs.add_population("EXC1", nNeuronInp, rate_rhs, initStrat="zero")
rs.add_population("EXC2", nNeuronHid, rate_rhs, initStrat="zero")
rs.add_population("INH2", nNeuronHid, rate_rhs, initStrat="zero")

# Inputs
nObjects = 2
timeShow = 100
inputMasks = [(np.random.uniform(0, 1, 100) > 0.5).astype(float) for i in range(nObjects)]
inp_func = lambda t: inputMasks[int(t / timeShow) % nObjects]
rs.add_input("I1", "EXC1", inp_func)

# Synapses
id_func = lambda x: x
stdp_func = lambda W, v1, v2, tau: np.outer(v2, v1 - W.T.dot(v2)) / tau
rs.add_synapse("SYN_EE12", "EXC1", "EXC2", "exc", id_func, stdp_func, initStrat='urand')
rs.add_synapse("SYN_EI22", "EXC2", "INH2", "exc", id_func, plasticityFunc=None, initStrat='urand')
rs.add_synapse("SYN_IE22", "INH2", "EXC2", "inh", id_func, plasticityFunc=None, initStrat='urand')

# Watchers
rs.add_pop_watcher("rawE1", "EXC1", "raw")
rs.add_pop_watcher("rawE2", "EXC2", "raw")
rs.add_pop_watcher("rawI2", "INH2", "raw")
rs.add_syn_watcher("rawS_EE12", "SYN_EE12", "raw")

# Run
rs.run(T)

# Plot results
fig, ax = plt.subplots(nrows=3)
ax[0].set_title("Populations")
ax[1].set_title("Synapses")
ax[2].set_title("Prediction Error")

# Plot 1
for wName in ["rawE1", "rawE2", "rawI2"]:
    times, val = rs.get_pop_watcher_results(wName)
    ax[0].plot(times, [np.linalg.norm(v) for v in val], label=wName)
for wNameInp in ["I1"]:
    times, norms = rs.get_inp_values(wNameInp, "norm")
    ax[0].plot(times, norms, label=wNameInp)

# Plot 2
times, synVals = rs.get_syn_watcher_results("rawS_EE12")
ax[1].plot(times[100:], [np.linalg.norm(sv) for sv in synVals][100:])

# Plot 3
_, valE1 = rs.get_pop_watcher_results("rawE1")
_, valE2 = rs.get_pop_watcher_results("rawE2")

errs = [np.linalg.norm(e1 - w1.T.dot(e2)) for e1,e2,w1 in zip(valE1, valE2, synVals)]
ax[2].semilogy(times, errs)

ax[0].legend()
plt.show()
