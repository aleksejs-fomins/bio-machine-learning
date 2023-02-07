import numpy as np
from scipy.ndimage import gaussian_filter1d
from lib.vis.opencv_lib import cvWriter


# Generate Poisson spikes within a given time interval, given rate for each neuron
def rates2spikes(rateArr, tMin, tMax):
    out_i = []
    out_t = []
    for i in range(len(rateArr)):
        t = tMin
        if rateArr[i] > 0:
            while t <= tMax:
                t += np.random.exponential(1 / rateArr[i])
                if t <= tMax:
                    out_i.append(i)
                    out_t.append(t)

    # Convert lists to numpy and sort w.r.t spike time
    out_i_np = np.array(out_i)
    out_t_np = np.array(out_t)
    inds = out_t_np.argsort()
    return out_i_np[inds], out_t_np[inds]
    

# From list of spike times and neuron indices,
# separate spikes to individual neurons
def spikes2lists(indList, tList, indMax):
#     npInd, npT = np.array(indList), np.array(tList)
#     return [npT[npInd == i] for i in range(indMax)]
    
    spikeLists = [[] for i in range(indMax)]
    for ind, t in zip(indList, tList):
        spikeLists[ind] += [t]
        
    return spikeLists


def spikes2rate(spike_times, starttime, runtime, binsize, tauconv=None):
    bincount = int((runtime - starttime) / binsize)
    counts, bin_edges = np.histogram(spike_times, bins=bincount, range=[float(starttime), float(starttime+runtime)])
    rate  = counts / binsize
    times = (bin_edges[1:] + bin_edges[:-1]) / 2

    if tauconv is not None:
        l = binsize / tauconv
        rate = gaussian_filter1d(rate, 1/l)
        # x = np.arange(50)
        # l = binsize / tauconv
        # k = np.exp(-l*x)
        # rate = np.convolve(rate, k)[:len(rate)]

    return times, rate


# Takes list of spike times and neuron indices, writes average frame to a file
def spikes2rateVideo(filePathName, frameDim, indList, tList, tMin, tMax, nStep, tau, maxRate=100):
    # Compute number of neurons form frame size
    # Note that some neurons might not spike
    nNeuron = frameDim[0] * frameDim[1]
    
    # Compute time step
    dt = (tMax - tMin) / nStep
    
    # Compute learning rate, maximized at 1
    # If learning rate is not maximal, spikes leave exponentially decaying trace in next frames
    alpha = min(1, dt / tau) if tau > 0 else 1
    
    # Calculate neuron firing rate change per spike directly in pixels
    # maximal expected rate should correspond to maximal intensity
    inpPerSpike = 255 / maxRate / tau if alpha < 1 else 255
    
    # Init
    tFrame = tMin
    neuronVec = np.zeros(nNeuron)
    
    print("Started writing video", filePathName, "of", nStep, "frames using time step", dt)
    
    # Open video writer
    with cvWriter(filePathName, frameDim, codec='MJPG') as vid:
        for ind, t in zip(indList, tList):
            if t < tFrame + dt:
                # If spike is in this bin, add it to the bin
                # Clip to maximal intensity if neuron activity too high
                neuronVec[ind] += inpPerSpike
            else:
                while t >= tFrame + dt:
                    # Whenever spike time exceeds current bin, update bin and write frame to video file
                    vid.write(np.clip(neuronVec, 0, 255).reshape(frameDim))
                    neuronVec *= (1 - alpha)
                    tFrame += dt

        # If spikes stopped before the last frame was reached,
        # continue calculating frames just based on old spikes
        while tFrame < tMax:
            vid.write(np.clip(neuronVec, 0, 255).reshape(frameDim))
            neuronVec *= (1 - alpha)
            tFrame += dt

# # Take interval from tMin to tMax, and split it into nStep steps
# # For each neuron, bin all spikes into these intervals
# def spikes2rates(spikeList, tMin, tMax, nStep):
    