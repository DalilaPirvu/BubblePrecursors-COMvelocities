import os,sys
sys.path.append('../')
sys.path.append('./bubbles_codes/')
from plotting import *
from bubble_tools import *
from experiment import *

tmp = 0
temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.array([nLat, lamb, phi0, temp])
print('Experiment', exp_params)

get_energy  = True
get_powspec = True
get_EMT     = True

aa=0
div=10

simList = np.array(np.linspace(minSim, maxSim+1, div+1), dtype='int')
divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])
sims_decayed = np.load(sims_decayed_file(*exp_params, minSim, maxSim, nTimeMAX))

tlist1 = np.linspace(1, 100, 100)
tlist2 = np.linspace(101, nTimeMAX-1, 100)
tlist = np.array(np.concatenate(([0], tlist1, tlist2)), dtype='int')

if True:
    asim, bsim = divdata[aa]
    powspec_data, energy_data, EMT_data = [[]]*3

    for sim in np.arange(asim, bsim):
        if sim not in sims_decayed[:,0]: continue

        path2sim = sim_location(*exp_params, sim)
        real, _ = get_realisation(nLat, sim, path2sim, 2, phieq)
        nC, nT, nN = np.shape(real)

        tlistcut = tlist[tlist<nT]
        real = real[:, tlistcut, :]
        print('Simulation, duration:', sim, nT)

        if get_energy:
            energy = 0.5*real[1]**2. + 0.5*real[2]**2. + V(real[0],lamb)
            energy = np.trapz(energy, xlist, axis=-1)
            energy_data.append(energy)

        if get_EMT:
            emt = real[1] * real[2]
            emt = np.trapz(emt, xlist, axis=-1)
            EMT_data.append(emt)

        if get_powspec:
            # only field and momentum
            fftreal = np.abs(np.fft.fft(real[:-1], axis=-1)/nLat)**2.
            fftreal[:,0] = 0. # subtracting the mean
            # shape fftreal: # field or momentum, # time slices, # modes
            powspec_data.append(fftreal)

    if get_energy:
        np.save(toten_tlist_file(*exp_params, asim, bsim), energy_data)
    if get_EMT:
        np.save(emt_tlist_file(*exp_params, asim, bsim), EMT_data)
    if get_powspec:
        np.save(powspec_tlist_file(*exp_params, asim, bsim), powspec_data)

# once all are finished, concatenate all partial lists
# and optionally remove partial lists at the end
if False:
    ALL_energy_data  = np.load(toten_tlist_file(*exp_params, *divdata[0]))
    ALL_EMT_data     = np.load(emt_tlist_file(*exp_params, *divdata[0]))
    ALL_powspec_data = np.load(powspec_tlist_file(*exp_params, *divdata[0]))

    for inds in divdata[1:]:
        ALL_energy_data  = np.concatenate((ALL_energy_data,  np.load(toten_tlist_file(*exp_params, *inds))), axis=0)
        ALL_EMT_data     = np.concatenate((ALL_EMT_data,     np.load(emt_tlist_file(*exp_params, *inds))), axis=0)
        ALL_powspec_data = np.concatenate((ALL_powspec_data, np.load(powspec_tlist_file(*exp_params, *inds))), axis=0)

    np.save(toten_tlist_file(*exp_params, minSim, maxSim),   [tlist, ALL_energy_data])
    np.save(emt_tlist_file(*exp_params, minSim, maxSim),     [tlist, ALL_EMT_data])
    np.save(powspec_tlist_file(*exp_params, minSim, maxSim), [tlist, ALL_powspec_data])

    for inds in divdata:
        os.remove(toten_tlist_file(*exp_params, *inds))
        os.remove(emt_tlist_file(*exp_params, *inds))
        os.remove(powspec_tlist_file(*exp_params, *inds))

print('All Done.')
