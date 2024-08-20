import os,sys
sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('../')
sys.path.append('./bubbles_codes/')
from plotting import *
from bubble_tools import *
from experiment import *

get_energy  = False
get_powspec = False
get_vars    = True

get_partial_stats = True
get_all_stats = False

tmp=2
phi0, temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.array([nLat, lamb, phi0, temp])
print('Experiment', exp_params)

aa=0
div=10

simList = np.array(np.linspace(minSim, maxSim, div+1), dtype='int')
divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])

tlist1 = np.arange(0, nTimeMAX, 1)
tlist2 = np.arange(0, nTimeMAX, 50)

if get_partial_stats:
    asim, bsim = divdata[aa]
    nttot2 = len(tlist2)
    nttot1 = len(tlist1)

    energy_data  = np.empty((bsim-asim, nTimeMAX))  
    EMT_data     = np.empty((bsim-asim, nTimeMAX))  
    stdEMT0      = np.empty((bsim-asim))
    powspec_data = np.empty((bsim-asim, 2, nttot2, nLat))
    vardata      = np.empty((bsim-asim, 2, nttot1))
    vardata[:], energy_data[:], EMT_data[:], stdEMT0[:], powspec_data[:] = 'nan', 'nan', 'nan', 'nan', 'nan'

    for sind, sim in enumerate(np.arange(asim, bsim)):
        path2sim      = sim_location(*exp_params, sim)
        real, outcome = get_realisation(nLat, sim, phieq, path2sim)
        nC, nT, nN    = np.shape(real)

        if get_energy:
            if nT > nTimeMAX:
                real = real[:, :nTimeMAX, :] # remove bubble from sim
            nC, nT, nN = np.shape(real)
            fld, mom, grd = real[0], real[1], real[2]
            print('Simulation, duration:', sim, nC, nN, nT)

            cds1 = (tlist1 < nT)
            energy_data[sind, cds1] = np.sum(0.5*mom**2. + 0.5*grd**2. + V(fld, lamb), axis=-1)
            EMT_data[sind, cds1]    = np.sum(mom * grd, axis=-1)
            stdEMT0[sind]           = np.std(mom[0] * grd[0])

        if get_powspec:
            if nT != nTimeMAX:
                real = real[:, :max(1, nT-3*nLat//4), :] # remove bubble from sim
            nC, nT, nN = np.shape(real)
            fld, mom, grd = real[0], real[1], real[2]
            print('Simulation, duration:', sim, nC, nN, nT)

            cds2 = (tlist2 < nT)
            cut2 = tlist2[cds2]
            fftfld, fftmom = np.empty((2, nttot2, nLat))
            fftfld[:], fftmom[:] = 'nan', 'nan'
            fftfld[cds2] = np.abs(np.fft.fft(fld[cut2,:], axis=-1)/nLat)**2.
            fftmom[cds2] = np.abs(np.fft.fft(mom[cut2,:], axis=-1)/nLat)**2.
            fftfld[cds2,0], fftmom[cds2,0] = 0., 0. # subtracting the mean
            powspec_data[sind, 0] = fftfld
            powspec_data[sind, 1] = fftmom

        if get_vars:
            if nT != nTimeMAX:
                real = real[:, :max(1, nT-3*nLat//4), :] # remove bubble from sim
            nC, nT, nN = np.shape(real)
            fld, mom = real[0], real[1]
            print('Simulation, duration:', sim, nC, nN, nT)

            cds3 = (tlist1 < nT)
            cut3 = tlist1[cds3]
            vardata[sind, 0, cut3] = np.var(fld, axis=-1)
            vardata[sind, 1, cut3] = np.var(mom, axis=-1)

    if get_energy:
        np.save(toten_tlist_file(*exp_params, asim, bsim), energy_data)
        np.save(emt_tlist_file(*exp_params, asim, bsim), EMT_data)
        np.save(stdemt0_tlist_file(*exp_params, asim, bsim), stdEMT0)

    if get_powspec:
        np.save(powspec_tlist_file(*exp_params, asim, bsim), powspec_data)

    if get_vars:
        np.save(varians_tlist_file(*exp_params, asim, bsim), vardata)
    print('Done', asim, bsim)


# once finished, merge partial lists
if get_all_stats:
    ALL_energy_data   = np.load(toten_tlist_file(*exp_params, *divdata[0]))
    ALL_EMT_data      = np.load(emt_tlist_file(*exp_params, *divdata[0]))
    ALL_stdEMT0_data  = np.load(stdemt0_tlist_file(*exp_params, *divdata[0]))
    ALL_powspec_data  = np.load(powspec_tlist_file(*exp_params, *divdata[0]))
    ALL_variance_data = np.load(varians_tlist_file(*exp_params, *divdata[0]))

    for inds in divdata[1:]:
        print(inds)
        ALL_energy_data   = np.concatenate((ALL_energy_data,   np.load(toten_tlist_file(*exp_params, *inds))),   axis=0)
        ALL_EMT_data      = np.concatenate((ALL_EMT_data,      np.load(emt_tlist_file(*exp_params, *inds))),     axis=0)
        ALL_stdEMT0_data  = np.concatenate((ALL_stdEMT0_data,  np.load(stdemt0_tlist_file(*exp_params, *inds))), axis=0)
        ALL_powspec_data  = np.concatenate((ALL_powspec_data,  np.load(powspec_tlist_file(*exp_params, *inds))), axis=0)
        ALL_variance_data = np.concatenate((ALL_variance_data, np.load(varians_tlist_file(*exp_params, *inds))), axis=0)

    print('Stored at:', powspec_tlist_file(*exp_params, minSim, maxSim))
    np.save(toten_tlist_file(*exp_params, minSim, maxSim),   [tlist1, ALL_energy_data])
    np.save(emt_tlist_file(*exp_params, minSim, maxSim),     [tlist1, ALL_EMT_data])
    np.save(stdemt0_tlist_file(*exp_params, minSim, maxSim), ALL_stdEMT0_data)
    np.save(powspec_tlist_file(*exp_params, minSim, maxSim), [tlist2, ALL_powspec_data])
    np.save(varians_tlist_file(*exp_params, minSim, maxSim), [tlist1, ALL_variance_data])

    if False:
        # optionally remove partial lists
        for inds in divdata:
            os.remove(toten_tlist_file(*exp_params, *inds))
            os.remove(emt_tlist_file(*exp_params, *inds))
            os.remove(stdemt0_tlist_file(*exp_params, *inds))
            os.remove(powspec_tlist_file(*exp_params, *inds))
            os.remove(varians_tlist_file(*exp_params, *inds))

print('All Done.')