import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV, Hz

import lib.neuro.traces_lib as traces_lib
from lib.vis.movie1d import time_bars


def plot_avg_spikes(spikeIdxs, spikeTimes, startTime, runTime, nNeuron, outname):
    # Convert spikes to rates
    spikesTimesNeuron = traces_lib.spikes2lists(spikeIdxs, spikeTimes, nNeuron)
    avgRatePerNeuron = np.array([len(sp) / runTime for sp in spikesTimesNeuron]).astype(float)
    avgNetworkRate_t, avgNetworkRate_r = traces_lib.spikes2rate(spikeTimes, startTime, runTime, 10 * ms)

    # Plot
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].plot(np.sort(avgRatePerNeuron))
    ax[1].plot(avgNetworkRate_t, avgNetworkRate_r / nNeuron)
    ax[2].plot(spikesTimesNeuron[0], np.ones(len(spikesTimesNeuron[0])), '.')
    ax[2].plot(spikesTimesNeuron[1], np.ones(len(spikesTimesNeuron[1])) + 1, '.')
    ax[2].plot(spikesTimesNeuron[2], np.ones(len(spikesTimesNeuron[2])) + 2, '.')
    ax[1].set_xlim([0, float(runTime)])
    ax[2].set_xlim([0, float(runTime)])
    ax[2].set_ylim([0, 3.5])
    ax[0].set_title("Time-average activity, per neuron, sorted")
    ax[1].set_title("Neuron-average activity, per time")
    ax[2].set_title("Example neuron activity, per time")

    plt.savefig(outname)
    plt.close()


def rate_time_bars(spikeIdxs, spikeTimes, startTime, runTime, nNeuron, outfname,
                   dpi=100, figW=600, figH=600, logy=False, binsize=10*ms, tauconv=None):
    spikesTimesNeuron = traces_lib.spikes2lists(spikeIdxs, spikeTimes, nNeuron)
    rateNeuron = [traces_lib.spikes2rate(t, startTime, runTime, binsize, tauconv=tauconv)[1] for t in spikesTimesNeuron]
    time_bars(np.array(rateNeuron).T, outfname, dpi=dpi, figW=figW, figH=figH, logy=logy)

