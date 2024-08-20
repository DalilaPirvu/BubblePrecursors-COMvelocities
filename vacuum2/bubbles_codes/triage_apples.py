import os,sys
sys.path.append('../')
sys.path.append('./bubbles_codes/')
sys.path.append('/home/dpirvu/python_stuff/')
from plotting import *
from bubble_tools import *
from experiment import *

# Classify decays
get_partial_stats = False
get_all_stats = True

tmp=0
phi0, lamb, sigmafld, minSim, maxSim, right_Vmax, normal = get_model(tmp)
exp_params = np.array([nLat, lamb, phi0, temp])
print('Experiment', exp_params)

if get_partial_stats:
    aa=0
    div=10

    simList = np.array(np.linspace(minSim, maxSim, div+1), dtype='int')
    divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])
    asim, bsim = divdata[aa]

    # investigate a little more these quantities
    crit_thresh = right_Vmax.x + 2.*sigmafld

    for sim in np.arange(asim, bsim):
        path2sim = sim_location(*exp_params, sim)
        path2CLEANsim = clean_sim_location(*exp_params, sim)

        if os.path.exists(path2sim):
            sizeSim = os.path.getsize(path2sim)
            # lh -l to find out the size of undecayed sims in bytes
            if sizeSim != bytesUNDECAYED:

                # if sizeSim is not corresponding to nTimeMAX, then decay happened at t_decay = nL-nLat/2 as per fortran condition
                # find out size of bubbles on average at t_decay
                # for collisions removal, can just impose cutoff at nT - X, where X is t_decay/2
                # this is because PBC and walls travelling at v=1 will cause bubble to wrap around the box by the amount X
                # this of course neglect double events
                # more generally: compute volume average <cos(phi(x))> = c. This is c=-1 at FV and c=1 at TV
                #                 t_decay is computed at c=-0.7
                #                 t_overshoot can be computed at c=|0.3|, where c grows linearly to 1 and then starts to decrease
                real, outcome = get_realisation(nLat, sim, phieq, path2sim)
                tdecay = max(0, np.shape(real)[1]-nLat//6)
                real, crit_rad = centre_bubble(real, tdecay, phieq, crit_thresh)
            #    real = remove_collisions(real, phieq)

                np.save(path2CLEANsim, [real, sim, tdecay, outcome])
                print('Simulation', sim, ', outcome', outcome, ', duration', np.shape(real)[1], ', tdecay, radius', tdecay, crit_rad)

# Code below centralizes all results. Smaller files load faster.
if get_all_stats:
    undecayed_sims, decayed_sims, decay_times = [], [], []
    for sim in range(minSim, maxSim):

        path2sim = sim_location(*exp_params, sim)
        path2CLEANsim = clean_sim_location(*exp_params, sim)
        if not os.path.exists(path2sim) or not os.path.exists(path2CLEANsim):
            outcome = 2
            undecayed_sims.append([sim, outcome])
            print(sim, outcome)

        elif os.path.exists(path2CLEANsim):
            real, sim, tdecay, outcome = np.load(path2CLEANsim)
            decayed_sims.append([sim, outcome])
            decay_times.append([sim, tdecay])
            print(sim, outcome, tdecay)

    np.save(sims_notdecayed_file(*exp_params, minSim, maxSim, nTimeMAX), undecayed_sims)
    np.save(sims_decayed_file(*exp_params, minSim, maxSim, nTimeMAX), decayed_sims)
    np.save(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX), decay_times)
    print('All saved.')

    # Optionally remove undecayed sims to save space
    if False:
        for sim, output in undecayed_sims:
            path2sim = sim_location(*exp_params, sim)
            os.remove(path2sim)
        
print('All Done.')
