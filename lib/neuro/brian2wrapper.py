from brian2 import NeuronGroup, Synapses


def NeuronGroupLIF(N, V0, VMAX, TAU):
    nsp = {'V0': V0, 'VMAX': VMAX, 'TAU': TAU}
    return NeuronGroup(N, 'dv/dt = (V0-v)/TAU : volt', threshold='v > VMAX', reset='v = V0', method='exact', namespace=nsp)


def NeuronGroupLIF_HP(N, V0, TAU_V, RATE_AVG, T_MIN, T_MAX, ETA_T):
    nsp = {'V0': V0, 'TAU_V': TAU_V, 'RATE_AVG': RATE_AVG, 'T_MIN': T_MIN, 'T_MAX': T_MAX, 'ETA_T': ETA_T}
    # Membrane potential V leaks over time and is reinforced by input
    # Neural spike rate R counts spikes of this neuron within a decaying time window
    eqs_dyn = '''
    dv/dt = (V0 - v) / TAU_V : volt
    dT/dt = -ETA_T * (T - T_MIN) * (T_MAX - T) * RATE_AVG : volt
    '''

    # On spike, the threshold is increased
    eqs_reset = '''
    v = V0
    T += ETA_T * (T - T_MIN) * (T_MAX - T)
    '''

    return NeuronGroup(N, eqs_dyn, threshold='v > T', reset=eqs_reset, namespace=nsp)


def SynapsesPlastic(G1, G2, plasticity_model):
    if plasticity_model['TYPE'][:4] == 'STDP':
        #         # Extract parameters
        #         TAU_PRE = plasticity_model['TAU_PRE']
        #         TAU_POST = plasticity_model['TAU_POST']
        #         DW_FORW = plasticity_model['DW_FORW']
        #         DW_BACK = plasticity_model['DW_BACK']
        #         DV_SPIKE = plasticity_model['DV_SPIKE']
        #         REL_W_MIN = plasticity_model['REL_W_MIN']
        #         REL_W_MAX = plasticity_model['REL_W_MAX']

        # Two auxiliary variables track decaying trace of
        # presynaptic and postsynaptic spikes
        syn_eq = '''
        dzpre/dt = -zpre/TAU_PRE : 1 (event-driven)
        dzpost/dt = -zpost/TAU_POST : 1 (event-driven)
        '''

        # In case of homeostatic synaptic plasticity, the weight
        # also decays to baseline value over time
        if 'HP' in plasticity_model['TYPE']:
            syn_eq += 'dw/dt = (REL_W_0 - w)/TAU_HP : 1 (event-driven)' + '\n'
        else:
            syn_eq += 'w : 1' + '\n'

        # On spike increase decaying variable by fixed amount
        # Increase weight by the value of the decaying variable
        # from the other side of the synapse
        # Truncate weight if it exceeds maximum
        syn_pre_eq = '''
        zpre += 1
        w = clip(w + DW_FORW * zpost, REL_W_MIN, REL_W_MAX)
        v_post += DV_SPIKE * w
        '''

        syn_post_eq = '''
        zpost += 1
        w = clip(w + DW_BACK * zpre, REL_W_MIN, REL_W_MAX)
        '''

        return Synapses(G1, G1, syn_eq, on_pre=syn_pre_eq, on_post=syn_post_eq, namespace=plasticity_model)
    else:
        raise ValueError('Unexpected Plasticity type', plasticity_model['TYPE'])