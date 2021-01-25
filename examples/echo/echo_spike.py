from brian2 import ms, mV, Hz

from lib.echo.spike import echo_spike
from lib.vis.spikeplots import plot_avg_spikes, rate_time_bars


p = {
    "N"         : 100,      # Number of neurons
    "p_conn"    : 0.10,     # Connection probability
    "nu_DV"     : 10.0*mV,  # Neuronal Noise Voltage per spike
    "nu_FREQ"   : 200*Hz,   # Neuronal Noise Frequency
    "nu_p"      : 0.5,      # Fraction of neurons that receive input
    "T0"        : 1.0,      # Average threshold
    "LIF_V_TAU" : 20*ms,    # Neuron leak timescale
    "LIF_V_0"   : 0.0*mV,   # Neuron base voltage
    "LIF_T_0"   : 50.0*mV,  # Neuron spike threshold
    "LIF_DV"    : 7.0*mV   # Neuron spike threshold
}

# Init and run
startTime = 0*ms
tStim = 500*ms
runTime = 2000*ms

spikeIdxs, spikeTimes = echo_spike(p, tStim, runTime)

plot_avg_spikes(spikeIdxs, spikeTimes, startTime, runTime, p['N'], 'avgspikes.png')

# rate_time_bars(spikeIdxs, spikeTimes, startTime, runTime,  p['N'], 'test_rate_bars.avi', binsize=5*ms, tauconv=30*ms)
