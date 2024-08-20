from plotting import *
from bubble_tools import *
from experiment import *

# Classify decays
tmp = 0
temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.asarray([nLat, lamb, phi0, temp])
print('Experiment', exp_params)

aa=0
div=5

decay_times = np.load(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX))
minDecTime = (nLat if temp==0 else nLat//2)
alltimes   = decay_times[:,1]
#alltimes   = decay_times[:,2] tdecayfortran
simList2Do = decay_times[alltimes>=minDecTime,0]
n2Do = len(simList2Do)
print('N = ', n2Do,'simulations to deboost:', simList2Do)

ranges2Do = np.array(np.linspace(0, n2Do+1, div), dtype='int')
divdata   = np.asarray([(ii,jj) for ii,jj in zip(ranges2Do[:-1], ranges2Do[1:])])

asim = divdata[aa]
bsim = divdata[aa+1]
ranList = simList2Do[asim : bsim]
random.shuffle(ranList)
print('Here we\'re doing the following ones:', asim, bsim, ranList)


if tmp==0:
    ampList0 = np.linspace(phieq + 4.*sigmafld, phieq + 7.*sigmafld, 20)
    ampList = np.linspace(phieq + 3.*sigmafld, phieq + 5.*sigmafld, 20)
elif tmp==1:
    ampList0 = np.linspace(phieq + 3.*sigmafld, phieq + 7.*sigmafld, 20)
    ampList = np.linspace(phieq + 3.*sigmafld, phieq + 5.*sigmafld, 20)
elif tmp==2:
    ampList0 = np.linspace(phieq + 3.5*sigmafld, phieq + 7.*sigmafld, 20)
    ampList = np.linspace(phieq + 3.5*sigmafld, phieq + 6.5*sigmafld, 20)
elif tmp==3:
    ampList0 = np.linspace(phieq + 3.5*sigmafld, phieq + 7.*sigmafld, 20)
    ampList = np.linspace(phieq + 3.5*sigmafld, phieq + 6.5*sigmafld, 20)

xList = np.arange(120, 2*crit_rad, 20)
crit_thresh = right_Vmax.x+2.*sigmafld
crit_rad = 80
thresh_av = right_Vmax.x+2.*sigmafld
rad_av = 30

for sim in ranList:
    print('Starting simulation, temp, lambda:', sim, temp, lamb)
    path2CLEANsim = clean_sim_location(*exp_params, sim)
    fullreal, sim, tdecay, tdecfortran, outcome = np.load(path2CLEANsim)

    if temp ==0: fullreal = multiply_bubble(fullreal, lightc, phieq, 0.75, normal, nLat)
    bubble = np.asarray([fullreal[0, -nLat:]]) # this is to speed up the boosting

    bool, vellist = True, []
    try:
        beta = find_COM_vel(bubble, ampList0, xList, nLat, lightc, phieq, crit_thresh, crit_rad, dx, temp, False)
        if np.isnan(beta):
            print('Simulation', sim, 'skipped. Dead end at step 0.')
            bool = False
    except:
        print('Some error within first vCOM detection. Skipped sim', sim)
        continue

    if temp == 0 and len(vellist)==0:
        beta = np.sign(beta) * 0.9

    while np.abs(beta) >= 0.05:
        roundlist = np.asarray([round(vv, 2) for vv in vellist])
        if round(beta, 2) in roundlist:
            beta = beta/2.
            if np.abs(beta) <= 0.025: break
        elif len(vellist)!=0 and np.abs(beta) > 0.1:
            if np.sign(beta) != np.sign(vellist[-1]):
                beta = beta/2.
                if np.abs(beta) <= 0.025: break
        vellist.append(beta)

        try:
            bubble = boost_bubble(bubble, nLat, lightc, phieq, beta, crit_thresh, crit_rad, thresh_av, rad_av, normal)
        except:
            print('Some error with the deboost. Skipped sim', sim)
            bool = False
            break
        try:
            beta = find_COM_vel(bubble, ampList, xList, nLat, lightc, phieq, crit_thresh, crit_rad, dx, temp, False)
            if np.isnan(beta):
                print('Simulation', sim, 'skipped. Dead end.')
                bool = False
                break
        except:
            print('Some error with vCOM detection. Skipped sim', sim)
            bool = False
            break

    if bool:
        vellist.append(beta)
        totbeta = get_totvel_from_list(vellist)

        fullreal = multiply_bubble(fullreal, lightc, phieq, totbeta, normal, nLat)
        if temp!=0: fullreal = fullreal[:, -nLat:]
 
        fullreal = boost_bubble(fullreal, nLat, lightc, phieq, totbeta, crit_thresh, crit_rad, thresh_av, rad_av, normal)
        fullreal = space_save(fullreal, phieq, crit_thresh, crit_rad, win=400)

        np.save(rest_sim_location(*exp_params, sim), [sim, fullreal, totbeta, beta])
        print('Simulation:', sim, ': total vel, final vel, vel list:', totbeta, beta, vellist)

# Centralize all total velocities. Compute average bubbles.
getvels = False
getavs = False
considerMaxVel = 0.9 # excludes faster bubbles from average

for tmp in range(len(tempList)):
    temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
    exp_params = np.array([nLat, lamb, phi0, temp])

    crit_thresh = right_Vmax.x+2.*sigmafld
    win = 300
    critSize = 20

    if getvels:
        ALLvels = []
        for sim in range(minSim, maxSim):
            path2RESTsim = rest_sim_location(*exp_params, sim)
            if os.path.exists(path2RESTsim):
                sim, real, totalvCOM, finalv = np.load(path2RESTsim)
                ALLvels.append([sim, totalvCOM])

        np.save(velocities_file(*exp_params), ALLvels)
        print('Velocities saved.', exp_params)

    if getavs:
        ALLatrest = []
        for sim in range(minSim, maxSim):      
            path2RESTsim = rest_sim_location(*exp_params, sim)
            if os.path.exists(path2RESTsim):
                sim, real, totalvCOM, finalv = np.load(path2RESTsim)
                if np.abs(totalvCOM) < considerMaxVel:
                    ALLatrest.append([sim, real])

        stacks  = stack_bubbles(ALLatrest, win, phieq, crit_thresh, critSize)
        stacks  = average_stacks(stacks, normal)
        avstack = average_bubble_stacks(stacks)
        np.save(average_file(*exp_params), avstack)
        print('Average bubble saved.', exp_params)

print('All Done.')