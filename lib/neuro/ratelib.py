import numpy as np

'''
RATELIB: Simulates a several arbitrarily-connected populations of rate neurons 

Current Features
[+] Each population has its own update function
[+] Each synapse set has its own update function (synapse set pairs all neurons from A->B)
[+] Can have copy synapses - non-plastic synapses that are copies of other synapses

Future Features
[+] Watchers: norm(x), norm(dx), norm(W), norm(dW)
[ ] Sparse connectivity
[ ] Performance, accuracy: Enable forwards-integration
[ ] Neuron-intrinsic plasticity
[ ] Multiple parameters per neuron/synapse
[ ] Performance: Allow some synapses not to be plastic
'''

class RateSimulator:
    def __init__(self, dt, tauPop, tauSyn):
        self.dt = dt
        self.tauPop = tauPop
        self.tauSyn = tauSyn
        self.popVal = {}
        self.popFunc = {}
        self.synSrcTrg = {}
        self.synVal = {}
        self.synType = {}
        self.synTransferFunc = {}
        self.synPlasticityFunc = {}
        self.synCopyNameSrcTrgType = {}
        self.inpFunc = {}
        self.popWatcher = {}
        self.popWatcherHist = {}
        self.synWatcher = {}
        self.synWatcherHist = {}

    def _init_array(self, shape, initStrat):
        if initStrat == "zero":
            return np.zeros(shape)
        elif initStrat == "urand":
            return np.random.uniform(0, 1, shape)
        else:
            raise ValueError("Unexpected init strategy", initStrat)

    def _update_dict_val(self, d, k, v):
        if k not in d:
            d[k] = v
        else:
            d[k] += v

    def _syn_pref_by_type(self, type):
        return -1 if type == "inh" else 1

    def _calc_syn_update(self, nameSyn, nameP1, synType=None):
        W = self.synVal[nameSyn]
        f = self.synTransferFunc[nameSyn]
        v1 = self.popVal[nameP1]
        if synType is None:
            mult = self._syn_pref_by_type(self.synType[nameSyn])
        else:
            mult = self._syn_pref_by_type(synType)
        return mult * f(W.dot(v1))

    def _calc_all_update(self, t):
        inpDict = {}

        # Calculate external inputs
        for nameInp, (nameP1, inputFunc) in self.inpFunc.items():
            self._update_dict_val(inpDict, nameP1, inputFunc(t))

        # Calculate synaptic inputs from plastic synapses
        for nameSyn, (nameP1, nameP2) in self.synSrcTrg.items():
            self._update_dict_val(inpDict, nameP2, self._calc_syn_update(nameSyn, nameP1))

        # Calculate synaptic inputs from copy-synapses
        for nameSyn, (nameCopy, nameP1, nameP2, synType) in self.synCopyNameSrcTrgType.items():
            self._update_dict_val(inpDict, nameP2, self._calc_syn_update(nameCopy, nameP1, synType=synType))

        # Calculate neuronal updates
        dv = {}
        for nameP1, inp in inpDict.items():
            v1 = self.popVal[nameP1]
            f = self.popFunc[nameP1]
            self._update_dict_val(dv, nameP1, f(v1, inp, self.tauPop))

        # Calculate synaptic updates. Only iterate over synapses that have a defined plasticity function
        #  other synapses are assumed to be fixed.
        dW = {}
        for nameSyn, f in self.synPlasticityFunc.items():
            nameP1, nameP2 = self.synSrcTrg[nameSyn]
            W = self.synVal[nameSyn]
            v1 = self.popVal[nameP1]
            v2 = self.popVal[nameP2]
            self._update_dict_val(dW, nameSyn, f(W, v1, v2, self.tauSyn))

        return dv, dW

    def _watch(self, val, watcherType):
        if watcherType == "raw":
            return np.copy(val)
        elif watcherType == "norm":
            return np.linalg.norm(val)
        else:
            raise ValueError("Unexpected watcher type", watcherType)

    def _watch_pop(self, nameP1, watcherType):
        return self._watch(self.popVal[nameP1], watcherType)

    def _watch_syn(self, nameS1, watcherType):
        return self._watch(self.synVal[nameS1], watcherType)

    def add_population(self, nameP1, nNeuron, updateFunc, initStrat="zero"):
        assert nameP1 not in self.popVal.keys()

        self.popVal[nameP1] = self._init_array(nNeuron, initStrat)
        self.popFunc[nameP1] = updateFunc

    def add_synapse(self, nameSyn, nameP1, nameP2, synType, transferFunc, plasticityFunc=None, initStrat="urand"):
        assert nameP1 in self.popVal.keys()
        assert nameP2 in self.popVal.keys()
        assert nameSyn not in self.synVal.keys()

        nNeuronSrc = self.popVal[nameP1].shape[0]
        nNeuronTrg = self.popVal[nameP2].shape[0]
        synShape = (nNeuronTrg, nNeuronSrc)

        self.synSrcTrg[nameSyn] = (nameP1, nameP2)
        self.synVal[nameSyn] = self._init_array(synShape, initStrat)
        self.synType[nameSyn] = synType
        self.synTransferFunc[nameSyn] = transferFunc
        if plasticityFunc is not None:
            self.synPlasticityFunc[nameSyn] = plasticityFunc

    def add_synapse_copy(self, nameSyn, nameCopy, nameP1, nameP2, type):
        assert nameP1 in self.popVal.keys()
        assert nameP2 in self.popVal.keys()
        assert nameCopy in self.synVal.keys()
        assert nameSyn not in self.synVal.keys()
        assert nameSyn not in self.synCopyNameSrcTrgType.keys()
        self.synCopyNameSrcTrgType[nameSyn] = (nameCopy, nameP1, nameP2, type)

    def add_input(self, nameInp, nameP1, inputFunc):
        assert nameP1 in self.popVal.keys()
        assert nameInp not in self.inpFunc.keys()

        self.inpFunc[nameInp] = (nameP1, inputFunc)

    def add_pop_watcher(self, namePW, nameP1, watcherType):
        assert nameP1 in self.popVal.keys()
        assert namePW not in self.popWatcher.keys()

        self.popWatcher[namePW] = (nameP1, watcherType)
        self.popWatcherHist[namePW] = [self._watch_pop(nameP1, watcherType)]

    def add_syn_watcher(self, nameSW, nameS1, watcherType):
        assert nameS1 in self.synVal.keys()
        assert nameSW not in self.popWatcher.keys()

        self.synWatcher[nameSW] = (nameS1, watcherType)
        self.synWatcherHist[nameSW] = [self._watch_syn(nameS1, watcherType)]

    def get_pop_watcher_results(self, namePW):
        return self.times, self.popWatcherHist[namePW]

    def get_syn_watcher_results(self, nameSW):
        return self.times, self.synWatcherHist[nameSW]

    def get_inp_values(self, nameInp, watcherType):
        inputFunc = self.inpFunc[nameInp][1]
        return self.times, [self._watch(inputFunc(t), watcherType) for t in self.times]

    def run(self, T):
        nT = int(T / self.dt) + 1
        self.times = np.arange(nT) * self.dt

        # Ignore 0-th timestep as it is the initialization timestep
        for i, t in enumerate(self.times[1:]):
            dvDict, dWDict = self._calc_all_update(t)

            # Update population values
            for nameP, dv in dvDict.items():
                self.popVal[nameP] += dv * self.dt

            # Update synapse values
            # Disallow negative synaptic values
            for nameSyn, dW in dWDict.items():
                self.synVal[nameSyn] = np.clip(self.synVal[nameSyn] + dW * self.dt, 0, None)

            # Watch populations
            for namePW, (nameP1, watcherType) in self.popWatcher.items():
                self.popWatcherHist[namePW] += [self._watch_pop(nameP1, watcherType)]

            # Watch synapses
            for nameSW, (nameS1, watcherType) in self.synWatcher.items():
                self.synWatcherHist[nameSW] += [self._watch_syn(nameS1, watcherType)]
