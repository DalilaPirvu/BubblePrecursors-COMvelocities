import os,sys
sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('../')
sys.path.append('./bubbles_codes/')
from plotting import *
from bubble_tools import *
from experiment import *
import random

# Classify decays
get_deboosts = False
get_avbubs = True

tmp=0
phi0, temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.asarray([nLat, lamb, phi0, temp])
print('Experiment', exp_params)

if get_deboosts:
    aa=0
    div=4

    decay_times = np.load(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX))
    done_sims   = np.array([sim for sim in decay_times[:,0] if os.path.exists(rest_sim_location(*exp_params, sim))])
    decay_times = np.array([decay_times[sind] for sind, ss in enumerate(decay_times[:,0]) if ss not in done_sims])

    minDecTime  = nLat*2//3
    alltimes    = decay_times[:,1]
    simList2Do  = decay_times[alltimes>=minDecTime, 0]
    n2Do        = len(simList2Do)
    print('N = ', n2Do,'simulations to deboost.')

    ranges2Do   = np.array(np.linspace(0, n2Do, div+1), dtype='int')
    if len(ranges2Do) > 1:
        divdata    = np.asarray([(ii,jj) for ii,jj in zip(ranges2Do[:-1], ranges2Do[1:])])
        asim, bsim = divdata[aa]
    else:
        asim, bsim = 0, n2Do

    ranList = simList2Do[asim : bsim]
    random.shuffle(ranList)
    print('Here we\'re deboosting the following sims:', asim, bsim, ranList)

    #threshm, threshM = right_Vmax.x + 0.5*sigmafld, right_Vmax.x + 1.5*sigmafld
    threshm, threshM = right_Vmax.x + 0.5*sigmafld, right_Vmax.x + 2.*sigmafld
    ampList = np.linspace(threshm, threshM, 20)

    crit_rad = 40
    winsize  = int(crit_rad*5) #np.array(np.linspace(crit_rad*2, crit_rad*3, 5), dtype='int')
    crit_thresh = right_Vmax.x + 2.*sigmafld

    plots=False

    print('Looking at at lambda, T, phi0, m2, sigma:', lamb, temp, phi0, m2(lamb), sigmafld)

    for sim in ranList:
        print('Starting simulation, temp, lambda:', sim, temp, lamb)
        path2CLEANsim = clean_sim_location(*exp_params, sim)
        fullreal, sim, tdecay, outcome = np.load(path2CLEANsim)

        fullreal = fullreal[:,-nLat:-nLat//4,nLat//4:-nLat//4] # this is to speed up the boosting

        bubble = fullreal[:1]
        bool, vellist = True, []
        try:
            beta = find_COM_vel(bubble, ampList, winsize, nLat, lightc, phieq, crit_thresh, crit_rad, dx, lamb, plots)
            if np.isnan(beta):
                print('Simulation, temp, lambda:', sim, temp, lamb, '. Dead end at step 0.')
                bool = False
        except:
            print('Some error within first vCOM detection. Skipped sim', sim)
            continue

        while np.abs(beta) >= 0.03 and bool:
            if len(vellist) > 0:
                copy = fullreal[:1]
                wcop = get_totvel_from_list(vellist)
                copy = boost_bubble(copy, nLat, lightc, phieq, wcop, crit_thresh, crit_rad, normal)
                vcop = find_COM_vel(copy, ampList, winsize, nLat, lightc, phieq, crit_thresh, crit_rad, dx, lamb, plots)
                if np.abs(vcop) < 0.03:
                    beta = vcop
                    break
                if np.abs(vcop) > np.abs(vellist[-1]):
                    beta = np.sign(beta) * random.randint(5,15)/100.
            vellist.append(beta)

            try:
                bubble = boost_bubble(bubble, nLat, lightc, phieq, beta, crit_thresh, crit_rad, normal)
                beta = find_COM_vel(bubble, ampList, winsize, nLat, lightc, phieq, crit_thresh, crit_rad, dx, lamb, plots)
                if np.isnan(beta):
                    bool = False
            except:
                print('Some error with deboost / finding the velocity. Skipped sim', sim)
                bool = False

        if bool:
            print('Simulation, temp, lambda:', sim, temp, lamb, 'doing final step.')
            vellist.append(beta)
            totbeta  = get_totvel_from_list(vellist)
            fullreal = boost_bubble(fullreal, nLat, lightc, phieq, totbeta, crit_thresh, crit_rad, normal)
            fullreal = space_save(fullreal, lightc, phieq, crit_thresh, crit_rad, nLat)

            path2RESTsim = rest_sim_location(*exp_params, sim)
            np.save(path2RESTsim, np.array([sim, fullreal, totbeta]))
            print('Saved. Total final velocity, vellist:', totbeta, vellist)

# Compute average bubbles.
if get_avbubs:
    crit_radList    = np.array(np.linspace(10, 60, 20), dtype='int'); print(crit_radList)
    crit_threshList = right_Vmax.x + np.linspace(1, 6, 20) * sigmafld; print(crit_threshList)

    win = 200
    considerMaxVel = 0.9 # excludes faster bubbles from average

    decayed_sims = np.load(sims_decayed_file(*exp_params, minSim, maxSim, nTimeMAX))
    mistake = np.argwhere(decayed_sims[:,1] == 1).flatten()

    all_data, all_vels = [], []
    for sim in range(minSim, maxSim):      
        path2RESTsim = rest_sim_location(*exp_params, sim)
        if os.path.exists(path2RESTsim):
            sim, bubble, totbeta = np.load(path2RESTsim)
            if sim in decayed_sims[mistake,0]:
                bubble[2] = -bubble[2]
            if np.abs(totbeta) < considerMaxVel:
                all_vels.append(np.array([sim, totbeta]))
                all_data.append(np.array([sim, bubble]))
    print('Total bubbles included:', len(all_data))

    if True:
        for cind, cth in enumerate(crit_radList):
            for tind, tsh in enumerate(crit_threshList):
                try:
                    stacks  = stack_bubbles(all_data, win, phieq, tsh, cth)
                    stacks  = average_stacks(stacks, normal)
                    avstack = average_bubble_stacks(stacks)
                    np.save(average_file(*exp_params)+'_critrad'+str(cth)+'_crittsh'+str(tsh), avstack)
                    print('Done', cind, tind)
                except: continue
    print('Done averaging.')

    loadedBubbles = np.zeros((len(crit_radList), len(crit_threshList), 2, 3, 2*win+1, 2*win+1))
    varmat        = np.zeros((len(crit_radList), len(crit_threshList)))

    for cind, cth in enumerate(crit_radList):
        for tind, tsh in enumerate(crit_threshList):
            
            loadedBubbles[cind, tind] = np.load(average_file(*exp_params)+'_critrad'+str(cth)+'_crittsh'+str(tsh)+'.npy')
            tp = 1 # 0 for average, 1 for error
            cp = 0 # 0 - field, 1 - momentum, 2 - gradient
            delt1 = 30

            bubble = loadedBubbles[cind, tind, tp, cp]
            nT, nN = np.shape(bubble)
            tl, tr, xl, xr = max(0, win-delt1), min(nT, win+delt1), max(0, win-delt1), min(nN, win+delt1)
            varmat[cind, tind] = np.mean(np.abs(bubble[tl:tr, xl:xr]))

    colmin, rowmin = np.where(varmat == np.min(varmat))
    final_crit_rad = crit_radList[colmin]
    final_crit_thresh = crit_threshList[rowmin]
    if len(final_crit_rad) != 1:
        if len(final_crit_rad) == 0: print('Failed')
        elif len(final_crit_rad) > 1:
            final_crit_rad = final_crit_rad[0]
            final_crit_thresh = final_crit_thresh[0]
    print('final_crit_rad =', final_crit_rad)
    print('final_crit_thresh = ', final_crit_thresh)

    stacks  = stack_bubbles(all_data, win, phieq, final_crit_thresh, final_crit_rad)
    stacks  = average_stacks(stacks, normal)
    avstack = average_bubble_stacks(stacks)
    np.save(average_file(*exp_params), avstack)
    print('Average bubble saved.')

print('All Done.')
