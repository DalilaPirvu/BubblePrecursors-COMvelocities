from bubble_tools_vac import *
from experiment_vac import *

# Classify decays
minSim = 0
maxSim = 2000

for pp, ph0 in enumerate(phi0List):
    if pp!=1: continue
   
    lamb      = lambList[pp]
    sigmafld  = fluct_stdev(lamb, ph0, temp)

    ### Useful
    right_Vmax  = sco.minimize_scalar(V, args=lamb, bounds=(np.pi, 2*np.pi), method='bounded')
    left_Vmax   = sco.minimize_scalar(V, args=lamb, bounds=(0    ,   np.pi), method='bounded')
    amp_thresh    = phieq + 5.*sigmafld
    crit_thresh   = phieq + 5.*sigmafld
    tv_thresh     = phieq + 5.*sigmafld
    crit_rad      = 80

    print('Looking at at lambda, T, phi0, m2, sigma:', lamb, temp, ph0, m2(lamb), sigmafld)

    if False:
        csim = maxSim-minSim
        aa=0
        bb=1
        asim = aa*csim//1
        bsim = bb*csim//1

        print('Cleaning up bubbles.')
        for sim in range(minSim, maxSim)[asim : bsim]:
            path_clean_sim   = clean_sim_location(nLat, lamb, ph0, temp, sim)
            if not os.path.exists(path_clean_sim+'.npy'):

                path_sim = sim_location(nLat, lamb, ph0, temp, sim)
                if os.path.exists(path_sim):
                    sizeSim = os.path.getsize(path_sim)
                    if sizeSim == 2650800448:
                        outcome = 2
                    else:
                        outcome = triage(nLat, nTimeMAX, phieq, sigmafld, path_sim)
                    print(sim, outcome)

                    if outcome != 2:
                        real, _ = get_realisation(nLat, sim, path_sim, outcome, phieq)
                        real    = remove_collisions(real, phieq, crit_rad)
                        real, tdecay = centre_bubble(real, nLat, phieq, crit_thresh, crit_rad)

                        np.save(path_clean_sim, [real, sim, tdecay, outcome])
                        print('Simulation', sim, ', outcome', outcome, ', duration', len(real[0]), ', tdecay', tdecay)

    path_sims_notdecayed = path_nodecay_sims(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)
    path_sims_decayed    = path_decayed_sims(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)
    path_decaytimes      = './data/tdecaylists_lamb'+str('%.4f'%lamb)+'_phi0'+str('%.4f'%ph0)+'_temp'+str('%.4f'%temp)

    if True:
        sims_notdecayed, sims_decayed, sim_decaytimes = [], [], []
        for sim in range(minSim, maxSim):
            path_sim = sim_location(nLat, lamb, ph0, temp, sim)

            if not os.path.exists(path_sim):
                outcome = 2
                sims_notdecayed.append(np.asarray([sim, outcome]))
            else:
                sizeSim = os.path.getsize(path_sim)
                if sizeSim == 2650800448:
                    outcome = 2
                    sims_notdecayed.append(np.asarray([sim, outcome]))

            path_clean_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)+'.npy'
            if os.path.exists(path_clean_sim):
                real, sim, tdecay, outcome = np.asarray(np.load(path_clean_sim))
                sims_decayed.append(np.asarray([sim, outcome]))
                sim_decaytimes.append(np.asarray([sim, tdecay]))
                print(sim, outcome, tdecay)

        #print('Simulations that do not decay within nTMax: ', sims_notdecayed)
        print('Simulations that decay within nTMax: ', sims_decayed)
        print('Simulations decay times: ', sim_decaytimes)

       # np.save(path_sims_notdecayed +'.npy', np.asarray(sims_notdecayed))
        np.save(path_sims_decayed+'.npy', np.asarray(sims_decayed))
        np.save(path_decaytimes  +'.npy', np.asarray(sim_decaytimes))

    print('Importing triage data.')
    sims_notdecayed = np.asarray(np.load(path_sims_notdecayed+'.npy'))
    sims_decayed    = np.asarray(np.load(path_sims_decayed+'.npy'))
    sim_decaytimes  = np.asarray(np.load(path_decaytimes+'.npy'))

    if False:
        print('Removing undecayed sims.')
        for sim, output in sims_notdecayed:
            path_sim = sim_location(nLat, lamb, ph0, temp, sim)
            if os.path.exists(path_sim):
                sizeSim = os.path.getsize(path_sim)
                if sizeSim == 2650800448:
                    os.remove(path_sim)
            

