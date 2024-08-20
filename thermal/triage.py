from bubble_tools import *
from experiment import *

# Classify decays
#minSim = 0
#maxSim = 2000
#minSim = 2000
#maxSim = 4000
minSim = 4000
maxSim = 6000

for tmp, temp in enumerate(Tlist):
    if tmp!=1: continue

    ph0       = phi0List[tmp]
    sigmafld  = fluct_stdev(m2eff, ph0, temp)
    print('Starting T, phi0, m2, sigma:', temp, ph0, m2eff, sigmafld)

    decay_too_fast_path = sims_that_decay_too_fast_file(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)
    do_not_decay_path   = sims_that_do_not_decay_file(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)
    good_decays_path    = sims_that_decay_fine_file(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)

    if not os.path.exists(good_decays_path + '.npy'):
#    if False:
        decay_too_fast = []
        do_not_decay   = []
        good_decays    = []
        for sim in range(minSim, maxSim):
            path_sim = sim_location(nLat, lamb, ph0, temp, sim)
            if os.path.exists(path_sim):
                outcome = triage(nLat, nTimeMAX, phieq, sigmafld, path_sim)
                print(sim, outcome)

                if outcome == 3:
                    decay_too_fast.append(np.asarray([sim, outcome]))
                elif outcome == 2:
                    do_not_decay.append(np.asarray([sim, outcome]))
                else:
                    good_decays.append(np.asarray([sim, outcome]))

        print('Simulations that decay too fast: ', decay_too_fast)
        print('Simulations that do not decay within nTMax ', nTimeMAX, ' : ', do_not_decay)
        print('Simulations that decay within nTMax ', nTimeMAX, ' : ', good_decays)

        np.save(decay_too_fast_path + '.npy', decay_too_fast)
        np.save(do_not_decay_path   + '.npy', do_not_decay)
        np.save(good_decays_path    + '.npy', good_decays)

    print('Importing triage data.')
    decay_too_fast = np.asarray(np.load(decay_too_fast_path + '.npy'))
    do_not_decay   = np.asarray(np.load(do_not_decay_path   + '.npy'))
    good_decays    = np.asarray(np.load(good_decays_path    + '.npy'))

    if False:
        print('Removing fast decays.')
        for sim, output in decay_too_fast:
            path_sim = sim_location(nLat, lamb, ph0, temp, sim)
            os.remove(path_sim)

    if False:
        print('Removing undecayed sims.')
        for sim, output in do_not_decay:
            path_sim = sim_location(nLat, lamb, ph0, temp, sim)
            os.remove(path_sim)

    if True:
        print('Cleaning up bubbles.')
        for sim, outcome in good_decays:
            path_clean_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)
            if not os.path.exists(path_clean_sim+'.npy'):

                path_sim = sim_location(nLat, lamb, ph0, temp, sim)
                if os.path.exists(path_sim):
                    real, _ = get_realisation(nLat, sim, path_sim, outcome, phieq)

                    real = remove_collisions(real, phieq, crit_rad)
                    real, tdecay = centre_bubble(real, nLat, phieq, crit_thresh, crit_rad)

                    np.save(path_clean_sim, [real, sim, tdecay])
#                    os.remove(path_sim)

                    print('Simulation', sim, ', outcome', outcome, ', duration', len(real[0]), ', tdecay', tdecay)
                else:
                    print('Simulation', sim, ' does not exist.')
