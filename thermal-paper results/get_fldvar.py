import os,sys
sys.path.append('../')
sys.path.append('./bubbles_codes/')
from plotting import *
from bubble_tools import *
from experiment import *

tmp=4
phi0, temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.array([nLat, lamb, phi0, temp])
print('Experiment', exp_params)

simList = np.array(np.arange(minSim, 1000), dtype='int')

variances = np.empty((len(simList)))
for sind, sim in enumerate(simList):
    path2sim       = sim_location(*exp_params, sim)
    real, outcome  = get_realisation(nLat, sim, phieq, path2sim)
    variances[sim] = np.std(real[0,0])
    print('sim', sim)

np.save(stdinit_file(*exp_params, minSim, maxSim), variances)
print('All Done.')
