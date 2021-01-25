import numpy as np

from brian2 import start_scope, prefs, run
from brian2 import NeuronGroup, PoissonGroup, Synapses, SpikeMonitor, StateMonitor, TimedArray

import lib.neuro.brian2wrapper as brian2wrapper


def echo_spike(p, tStim, tTot):
    start_scope()
    prefs.codegen.target = "numpy"

    ################################
    # Compute properties
    ################################
    dvInpSpike = p['nu_DV']   # Voltage increase per noise spike
    dvExcSpike = p['LIF_DV']  # Voltage increase per lateral spike
    # dvExcSpike = p['LIF_T_0'] / (1.0 * p['N'] * p['p_conn'])  # Voltage increase per lateral spike

    print("typical spike threshold", p['LIF_T_0'])
    print("typical potential change per noise spike", dvInpSpike)
    print("typical potential change per lateral spike", dvExcSpike)

    ################################
    # Create input population
    ################################
    nTStimPost = int(tTot / tStim) - 1  # After input switched off, 0-input will be repeated nTStimPost*tStim timesteps
    patternInp = (np.random.uniform(0, 1, p['N']) < p['nu_p']).astype(int)
    pattern0 = np.zeros(p['N'])
    rates_all = np.array([patternInp] + [pattern0]*nTStimPost) * p['nu_FREQ']
    rateTimedArray = TimedArray(rates_all, dt=tStim)
    gInp = PoissonGroup(p['N'], rates="rateTimedArray(t, i)")
    # gInp = PoissonGroup(p['N'], p['nu_FREQ'])

    ################################
    # Create reservoir population
    ################################
    gExc = brian2wrapper.NeuronGroupLIF(p['N'], p['LIF_V_0'], p['LIF_T_0'], p['LIF_V_TAU'])

    ################################
    # Create synapses
    ################################
    sInpExc = Synapses(gInp, gExc, on_pre='v_post += dvInpSpike', method='exact')
    sExcExc = Synapses(gExc, gExc, on_pre='v_post += dvExcSpike', method='exact')

    ################################
    # Connect synapses
    ################################
    # * Input and LIF one-to-one
    # * LIF neurons to each other sparsely
    sInpExc.connect(j='i')
    sExcExc.connect(p=p['p_conn'])

    ################################
    # Init Monitors
    ################################
    #spikemonInp = SpikeMonitor(gInp)
    spikemonExc = SpikeMonitor(gExc)

    ################################
    # Run simulation
    ################################
    run(tTot)

    return np.array(spikemonExc.i), np.array(spikemonExc.t)
