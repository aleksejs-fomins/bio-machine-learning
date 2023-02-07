import numpy as np
import matplotlib.pyplot as plt
from brian2 import start_scope, run, Hz, ms, NeuronGroup, Synapses, StateMonitor

'''
Balance equation for sustained firing
1. Each neuron has firing rate r
2. Each neuron receives recurrent input (n-1) * w * r   per second
3. The recurrent input has to produce firing rate r.
Thus, we need to solve the equation

dv/dt = -v/tau + I,  where I = (n-1) * w * r

Solution is I = (vmax - v) / tau / (1 - exp(-1/(r*tau)))

So optimal input weight w depends on the firing rate r 
So, there is no such weight w that can result in sustained activity for a wide range of firing rates
However, if r*tau >> 1, then

I ~ (vmax - v) * r

and thus 

w ~ (vmax - v) / (n - 1)
'''

start_scope()

A = 5
nNeuron = 20
period = 100*ms
tau = 5*ms
eqs = '''
dv/dt = (I-v)/tau : 1
I = A * floor(0.5 + sin(2 * pi * t / period)) : 1
'''

group1 = NeuronGroup(nNeuron, eqs, threshold='v>1', reset='v=0', method='euler')
monitor1 = StateMonitor(group1, variables=True, record=True)
syn1 = Synapses(group1, group1, on_pre='v_post += 1 / (nNeuron - 1)')
syn1.connect(condition='i != j')

run(500*ms)

plt.figure()
plt.plot(monitor1.t/ms, np.mean(monitor1.v, axis=0), label='v')
plt.plot(monitor1.t/ms, monitor1.I[0], label='I')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend(loc='best')
plt.show()