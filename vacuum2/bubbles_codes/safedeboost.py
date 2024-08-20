from bubble_tools import *
from experiment import *

# Classify decays
tmp=0

if tmp == 0: minSim, maxSim = 0, 3000
elif tmp == 1: minSim, maxSim = 0, 2800
elif tmp == 2: minSim, maxSim = 0, 2000

temp = tempList[tmp]
lamb = lambList[tmp]
sigmafld = fluct_stdev(lamb, phi0, temp)

### Useful
right_Vmax  = sco.minimize_scalar(Vinv, args=lamb, bounds=(np.pi, 2*np.pi), method='bounded')
left_Vmax   = sco.minimize_scalar(Vinv, args=lamb, bounds=(0    ,   np.pi), method='bounded')
amp_thresh  = right_Vmax.x+2.*sigmafld
crit_thresh = right_Vmax.x+2.*sigmafld
tv_thresh   = right_Vmax.x+2.*sigmafld
crit_rad    = 80

if tmp==0: 
    ampList = np.linspace(phieq + 3.5*sigmafld, phieq + 5.*sigmafld, 20)
elif tmp==1: 
    ampList = np.linspace(phieq + 4.*sigmafld, phieq + 6.*sigmafld, 20)
elif tmp==2: 
    ampList = np.linspace(phieq + 4.*sigmafld, phieq + 6.*sigmafld, 20) # for tmp = 2
xList = np.arange(120, 2*crit_rad, 20)
print('Looking at at lambda, T, phi0, m2, sigma:', lamb, temp, phi0, m2(lamb), sigmafld)

if os.path.exists(path_decaytimes(lamb, phi0, temp)+'.npy'):
    decaydata = np.load(path_decaytimes(lamb, phi0, temp)+'.npy')
    simList = []
    for sim, tdecay in decaydata:
        if tdecay > nLat//2:
            path_rest_sim  = bubble_at_rest(nLat, lamb, phi0, temp, sim)+'.npy'
            if not os.path.exists(path_rest_sim):
                path_clean_sim = clean_sim_location(nLat, lamb, phi0, temp, sim)+'.npy'
                if os.path.exists(path_clean_sim):
                    simList.append(sim)

    csim = len(simList)
    aa=0
    bb=1
    asim = aa*csim//3
    bsim = bb*csim//3

    ranList = simList[asim : bsim]
    random.shuffle(ranList)
    print(ranList)
    for sim in ranList:
        print('Starting simulation, temp, lambda:', sim, temp, lamb)
        path_clean_sim = clean_sim_location(nLat, lamb, phi0, temp, sim)
        path_rest_sim  = bubble_at_rest(nLat, lamb, phi0, temp, sim)
        fullreal, sim, tdecay, outcome = np.load(path_clean_sim+'.npy')

        try:
            bubble = np.asarray([fullreal[0]]) # this is to speed up the boosting in anticipation of the N-column simulations 
            beta, stbeta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, dx, False)
            bool, vellist = True, []
            if np.isnan(beta):
                print('Simulation, temp, lambda:', sim, temp, lamb, 'dead end at step 0.')
                bool = False

            while np.abs(beta) > 0.05:
                if tmp==1:
                    if np.abs(beta) > 0.4:
                        beta = np.sign(beta)*random.randint(10,15)/40.
                vellist.append(beta)

                bubble = boost_bubble(bubble, nLat, lightc, phieq, beta, crit_thresh, crit_rad, normal)
                beta, stbeta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, dx, False)
                if np.isnan(beta):
                    print('Simulation, temp, lambda:', sim, temp, lamb, ': dead end.')
                    bool = False
                    break

            if bool:
                vellist.append(beta)
                totbeta = get_totvel_from_list(vellist)
                fullreal = multiply_bubble(fullreal, lightc, phieq, totbeta)
                fullreal = boost_bubble(fullreal, nLat, lightc, phieq, totbeta, crit_thresh, crit_rad, normal)

                fullreal = space_save(fullreal, phieq, crit_thresh, crit_rad, win=400)
                np.save(path_rest_sim, np.asarray([sim, fullreal, totbeta, beta]))

                print('Simulation, temp, lambda:', sim, temp, lamb, ': total vel, final vel, vel list:', totbeta, beta, vellist)
        except:
            print('Simulation, temp, lambda:', sim, temp, lamb, ' skipped due to unknown error.')
            continue
print('All Done.')
