# To run this script, in a separate terminal type:
#### python3 deboost.py >> output.txt
from bubble_tools_vac import *
from experiment_vac import *

import time
import functools
from concurrent.futures import ProcessPoolExecutor

# Classify decays
minSim = 0
maxSim = 3000


tmp=0

ph0      = phi0List[tmp]
lamb     = lambList[tmp]
sigmafld = fluct_stdev(lamb, ph0, temp)

### Useful
right_Vmax  = sco.minimize_scalar(V, args=lamb, bounds=(np.pi, 2*np.pi), method='bounded')
left_Vmax   = sco.minimize_scalar(V, args=lamb, bounds=(0    ,   np.pi), method='bounded')
crit_thresh   = phieq + 5.*sigmafld
crit_rad      = 50

sigmafld = fluct_stdev(lamb, ph0, temp)
ampList  = np.linspace(phieq + 4.*sigmafld, phieq + 5.5*sigmafld, 30)
xList    = np.arange(80, 3*crit_rad, 20)
print('Looking at at lambda, T, phi0, m2, sigma:', lamb, temp, ph0, m2(lamb), sigmafld)

path_sims_decayed = path_decayed_sims(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)+'.npy'
if os.path.exists(path_sims_decayed):
    sims_decayed  = np.asarray(np.load(path_sims_decayed))

    simList = []
    for sim, outcome in sims_decayed:
        path_clean_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)+'.npy'
        path_rest_sim  = bubble_at_rest(nLat, lamb, ph0, temp, sim)+'.npy'
        if not os.path.exists(path_rest_sim):
            if os.path.exists(path_clean_sim):
                real, sim, tdecay, outcome = np.asarray(np.load(path_clean_sim))
                if tdecay > 1024:
                    simList.append(sim)
    simList = np.asarray(simList)

    print('Remaining sims to compute:', simList)

    csim = len(simList)
    aa=0
    bb=1
    asim = aa*csim//20
    bsim = bb*csim//20

    for sim in simList[asim : bsim]:
        print('Starting simulation', sim)
        path_clean_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)
        path_rest_sim  = bubble_at_rest(nLat, lamb, ph0, temp, sim)
        fullreal, sim, tdecay, outcome = np.load(path_clean_sim+'.npy')

        try:
            bubble = np.asarray([fullreal[0]]) # this is to speed up the boosting in anticipation of the N-column simulations 
            beta, stbeta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, dx, False)
            bubble = multiply_bubble(bubble, lightc, phieq, beta)
            bool, vellist = True, []
            if np.isnan(beta):
                print('Dead end at step 0.')
                bool = False

            while np.abs(beta) > 0.05:
                if np.abs(beta) > 0.9:
                    beta = np.sign(beta)*random.randint(20,28)/40.
                vellist.append(beta)

                bubble = boost_bubble(bubble, nLat, lightc, phieq, beta, crit_thresh, crit_rad, normal)
                beta, stbeta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, dx, False)
                if np.isnan(beta):
                    print('Dead end.')
                    bool = False
                    break

            if bool:
                vellist.append(beta)
                totbeta = get_totvel_from_list(vellist)
                fullreal = boost_bubble(fullreal, nLat, lightc, phieq, totbeta, crit_thresh, crit_rad, normal)

                fullreal = space_save(fullreal, phieq, crit_thresh, crit_rad, win=400)
                np.save(path_rest_sim, np.asarray([sim, fullreal, totbeta, beta]))

                print('Total vel, final vel, vel list:', totbeta, beta, vellist)
        except:
            print('Simulation'+str(sim)+' skipped due to unknown error.')
            continue
print('All Done.')
