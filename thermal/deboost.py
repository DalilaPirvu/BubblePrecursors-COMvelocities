# To run this script, in a separate terminal type:
#### python3 deboost.py >> output.txt
from bubble_tools import *
from experiment import *

import time
import functools
from concurrent.futures import ProcessPoolExecutor

# Classify decays
#minSim = 0
#maxSim = 2000
minSim = 2000
maxSim = 4000
#minSim = 4000
#maxSim = 6000


tmp=3

temp     = Tlist[tmp]
ph0      = phi0List[tmp]
sigmafld = fluct_stdev(m2eff, ph0, temp)
#ampList  = np.linspace(phieq + 5.5*sigmafld, phieq + 6.5*sigmafld, 30) # for stragglers, especially at lower T
#ampList  = np.linspace(phieq + 4.5*sigmafld, phieq +5.5*sigmafld, 30) # for T=0.1
ampList  = np.linspace(phieq + 4.*sigmafld, phieq + 6.*sigmafld, 30) # for T=0.12 and T=0.11
xList    = np.arange(50, 71, 10)
print('Looking at at temperature T, phi0, m2, sigma:', temp, ph0, m2eff, sigmafld)

good_decays_path = sims_that_decay_fine_file(nLat, lamb, ph0, temp, minSim, maxSim, nTimeMAX)
if os.path.exists(good_decays_path+'.npy'):
    good_decays  = np.asarray(np.load(good_decays_path + '.npy'))

    simList = []
    for sim, outcome in good_decays:

        loc_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)+'.npy'
        if os.path.exists(loc_sim):

            rest_sim = bubble_at_rest(nLat, lamb, ph0, temp, sim)+'.npy'
            if not os.path.exists(rest_sim):
                simList.append(sim)


    print('Remaining sims to compute:', simList)

    csim = len(simList)
    aa=0
    bb=1
    asim = aa*csim//1
    bsim = bb*csim//1

    for sim in simList[asim : bsim]:
        print('Starting simulation', sim)
        loc_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)+'.npy'
        fullreal, sim, tdecay = np.load(loc_sim)

        # this is to speed up the boosting in anticipation of the N-column simulations 
        bubble = np.asarray([fullreal[0]])

        try:
            beta, stbeta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, False)
            bool, vellist = True, []
            if np.isnan(beta):
                print('Dead end at step 0.')
                bool = False
            bubble = multiply_bubble(bubble, lightc, phieq, beta)

            while np.abs(beta) > 0.04:
                if np.abs(beta) > 0.3:
                    beta = np.sign(beta)*random.randint(2,5)/20.
                vellist.append(beta)

                bubble = boost_bubble(bubble, nLat, lightc, phieq, V, beta, crit_thresh, crit_rad, normal)
                beta, stbeta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, False)
                if np.isnan(beta):
                    print('Dead end.')
                    bool = False
                    break

            if bool:
                vellist.append(beta)
                totbeta = get_totvel_from_list(vellist)
                fullreal = boost_bubble(fullreal, nLat, lightc, phieq, V, totbeta, crit_thresh, crit_rad, normal)

                np.save(bubble_at_rest(nLat, lamb, ph0, temp, sim), np.asarray([sim, fullreal, totbeta, beta]))
                print('Total vel, final vel, vel list:', totbeta, beta, vellist)
        except:
            print('Simulation'+str(sim)+' skipped due to unknown error.')
            continue
print('All Done.')
