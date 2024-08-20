import os, sys
import numpy as np
import random
import scipy as scp
import scipy.optimize as sco
import scipy.signal as scs
import scipy.interpolate as intp
from functools import partial
from itertools import cycle
import scipy.ndimage
from scipy.ndimage import gaussian_filter

import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})

def extract_data(nL, path_sim):
    data = np.genfromtxt(path_sim)
    nNnT, nC = np.shape(data)
    return np.asarray([np.reshape(data[:,cc], (nNnT//nL, nL)) for cc in range(nC)])

def triage(nL, nTMax, phieq, sigmafld, path_sim):
    data = extract_data(nL, path_sim)
    shp = np.shape(data[0])
    slice = data[0,-1]
    outcome = check_decay(shp, nTMax, data[0,-1], phieq, sigmafld)
    return outcome

def get_realisation(nL, sim, path_sim, outcome, phieq):
    data = extract_data(nL, path_sim)

    if outcome == 1:
        data[0] = 2.*phieq - data[0]
        data[1] = - data[1]
    return np.asarray(data), sim

def check_decay(shp, nTMax, slice, phieq, sigmafld):
    nT, nN = shp
    refl_slice = reflect_against_equil(slice, phieq)

    # if it decays too fast, discard simulation
    # check if nN is too much
    if nT < 2.*nN:
        return 3

    # if it doesn't decay until nTMax, 
    if nT == nTMax+1:
        return 2
 #       if np.std(slice) < 2.*sigmafld:
 #           return 2
 #       if np.mean(refl_slice) < phieq+2.*sigmafld:
 #           return 2

    right_phi = np.count_nonzero(slice > phieq+np.pi)
    left_phi = np.count_nonzero(slice < phieq-np.pi)
    if right_phi>left_phi:
        return 0
    else:
        return 1

def simple_imshow(simulation, extent_floats, title='Sim'):
    fig, ax0 = plt.subplots(1, 1, figsize=(6,5))
    im0 = plt.imshow(simulation, aspect='auto', interpolation='none', extent=extent_floats, origin='lower', cmap='PRGn')
    clb = plt.colorbar(im0, ax = ax0)
    ax0.set_title(title); plt.show()
    return

def remove_collisions(real, phi_init, crit_rad):
    bubble  = real[0]
    T, N    = np.shape(bubble)
    bubble  = reflect_against_equil(bubble, phi_init)
    counts  = np.count_nonzero(bubble > phi_init + 2.*np.pi, axis=1)
    tdecap  = np.where(counts[::-1] == 0)[0][0]
    tdecap  = min(T - tdecap - N//8, T)
    return real[:, :tdecap]

def centre_bubble(real, nL, phi_init, crit_thresh, crit_rad):
    bubble = real[0]
    T, N   = np.shape(bubble)
    t_centre, x_centre = find_nucleation_center(bubble, phi_init, crit_thresh, crit_rad)
    real   = np.roll(real, N//2-x_centre, axis=-1)
    tamp   = T - 2*nL
    tamp   = max(0, tamp)
    return real[:, tamp:], t_centre

def index_at_size(arr, size):
    return np.argmin(np.abs(arr - size))

def bubble_counts_at_equal_time(bubble, thresh):
    return np.count_nonzero(bubble > thresh, axis=1)

def bubble_counts_at_equal_space(bubble, thresh):
    return np.count_nonzero(bubble > thresh, axis=0)

def reflect_against_equil(bubble, phi_init):
    return abs(bubble-phi_init) + phi_init

def find_nucleation_center(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)
    refl_bubble = reflect_against_equil(bubble, phi_init)
    bubble_counts = bubble_counts_at_equal_time(refl_bubble, crit_thresh)
    t0 = index_at_size(bubble_counts, crit_rad)
    bubble_counts = bubble_counts_at_equal_space(bubble[:min(t0+crit_rad,T-1)], crit_thresh)
    x0 = np.argmax(bubble_counts)
    return min(T-1,t0), min(N-1,x0)

def find_nucleation_center2(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)
    refl_bubble = reflect_against_equil(bubble, phi_init)
    bubble_counts = bubble_counts_at_equal_time(refl_bubble, crit_thresh)
    t0 = index_at_size(bubble_counts, crit_rad)
    bubble_counts = np.argwhere(bubble[t0]>crit_thresh)
    x0 = int(np.mean(bubble_counts))
    return min(T-1,t0), min(N-1,x0)

def find_t_max_width(bubble, light_cone, phi_init, crit_thresh, crit_rad, t0):
    T, N = np.shape(bubble)
    bubble = bubble[t0:]
    refl_bubble = reflect_against_equil(bubble, phi_init)

    bubble_counts = bubble_counts_at_equal_time(refl_bubble, crit_thresh)
    bubble_diffs = bubble_counts[1:] - bubble_counts[:-1]

    tmax = np.argwhere(bubble_diffs[::-1] >= light_cone*2).flatten()
    if len(tmax) > 0:
        for ii, jj in zip(tmax[:-1], tmax[1:]):
            if jj-ii == 1:
                tmax = ii
                break
    else:
        tmax = 1
    return T-tmax

def multiply_bubble(bubble, light_cone, phi_init, vCOM):
    if vCOM<0:
        bubble = bubble[:,:,::-1]
    C, T, N = np.shape(bubble)
    # multiplies bubbles so tail is kept without pbc
    bubble = np.asarray([np.tile(bubble[col], fold(vCOM)) for col in range(C)])
    TT, NN = np.shape(bubble[0])
    for t in range(TT):
        a, b = int((TT-t)/light_cone) + N, int((TT-t)/light_cone/3.) - N//4
        x1, x2 = np.arange(a, NN), np.arange(b)
        x1, x2 = x1 - a, x2 - (b-NN)
        for x in np.append(x1, x2):
            if 0 <= x < NN:
                bubble[0,t,x] = phi_init
    if vCOM<0:
        bubble = bubble[:,:,::-1]
    bubble = defeet_bubble(bubble, phi_init)
    return bubble

def defeet_bubble(bubble, phi_init):
    means = np.mean(bubble[0], axis=-1)
    Tamp = np.argwhere(means-phi_init>1e-10).flatten()
    if len(Tamp)==0:
        Tamp = [0]
    return bubble[:,Tamp[0]:,:]

def tanh(x, r0L, r0R, dr, vL, vR):
    wL, wR = dr/gamma(vL), dr/gamma(vR)
    return ( np.tanh( (x - r0L)/wL ) + np.tanh( (r0R - x)/wR ) ) * np.pi/2. + np.pi

def tanh_fit(bubble_slice, axis, prior):
    bounds = ((axis[0], 0, 0, 0, 0), (0, axis[-1], axis[-1], 1, 1))
    tanfit, _ = sco.curve_fit(tanh, axis, bubble_slice, p0=prior, bounds=bounds)
 #   plt.plot(axis, bubble_slice, 'r-'); plt.plot(axis, tanh(axis, *tanfit), 'g-'); plt.show()
    return tanfit

def hypfit_right_mover(tt, rr):
    hyperbola    = lambda t, a, b, c: np.sqrt(c + (t - b)**2.) + a
    try:
        prior    = (float(min(rr)), float(tt[np.argmin(rr)]), 1e3)
        fit, _   = sco.curve_fit(hyperbola, tt, rr, p0 = prior)
        traj     = hyperbola(tt, *fit)
        return traj
    except:
        return []

def hypfit_left_mover(tt, ll):
    hyperbola    = lambda t, d, e, f: - np.sqrt(f + (t - e)**2.) + d
    try:
        prior    = (float(max(ll)), float(tt[np.argmax(ll)]), 1e3)
        fit, _   = sco.curve_fit(hyperbola, tt, ll, p0 = prior)
        traj     = hyperbola(tt, *fit)
        return traj
    except:
        return []

def get_velocities(rrwallfit, llwallfit):
    uu = np.gradient(rrwallfit) #wall travelling with the COM
    vv = np.gradient(llwallfit) #wall travelling against
    uu[np.abs(uu)>=1.] = np.sign(uu[np.abs(uu)>=1.])*(1.-1e-15)
    vv[np.abs(vv)>=1.] = np.sign(vv[np.abs(vv)>=1.])*(1.-1e-15)
    uu[np.isnan(uu)] = np.sign(vv[-1])*(1.-1e-15)
    vv[np.isnan(vv)] = np.sign(uu[-1])*(1.-1e-15)

    # centre of mass velocity
    aa = ( 1.+uu*vv-np.sqrt((-1.+uu**2.)*(-1.+vv**2.)))/( uu+vv)
    # instantaneous velocity of wall
    bb = (-1.+uu*vv+np.sqrt((-1.+uu**2.)*(-1.+vv**2.)))/(-uu+vv)
    return uu, vv, aa, bb

def find_COM_vel(real, ampList, xList, nL, light_cone, phi_init, crit_thresh, crit_rad, plots):
    real = gaussian_filter(real[0], 1, mode='nearest')

    nT, nN = np.shape(real)
    t_cen, x_centre = find_nucleation_center(real, phi_init, crit_thresh, crit_rad)
    t_max_width = find_t_max_width(real, light_cone, phi_init, crit_thresh, crit_rad, t_cen-nL)
    if t_cen > t_max_width:
        t_cen = t_max_width

    t_centre, x_centre = find_nucleation_center(real[:t_cen], phi_init, crit_thresh, crit_rad)
    t_stop = min(t_centre + crit_rad//2, t_max_width)

    betas = np.zeros((len(ampList), len(xList)))
    for xx, x_size in enumerate(xList):
        for vv, v_size in enumerate(ampList):
            vel_plots = (True if (xx%(len(xList)-1)==0 and vv%(len(ampList)-1)==0 and plots) else False)
            betas[vv, xx] = get_COM_velocity(real, phi_init, crit_thresh, crit_rad, t_cen, t_stop, v_size, x_size, vel_plots)
    if plots:
        simple_imshow(betas, [xList[0],xList[-1],ampList[0],ampList[-1]], r'Mean $v = $'+str('%.2f'%np.nanmean(betas)))
    return np.nanmean(betas), np.nanstd(betas)


def get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, t_cen, t_stop, vvv, xxx, plots):
    t_centre, x_centre = find_nucleation_center(simulation[:t_cen], phi_init, crit_thresh, crit_rad)
    nT, nN  = np.shape(simulation)
    tl  = max(0, t_centre-crit_rad)
    tr  = min(nT-1, t_stop)
    xl  = max(0, x_centre-xxx)
    xr  = min(nN-1, x_centre+xxx)
    xN  = xr-xl
 #   print(t_centre, x_centre, tl, tr, xl, xr)

    data_list, prior, target = [], None, xN/2.
    for tt in range(tr, tl, -1):
        slice = simulation[tt, xl:xr]
        if np.count_nonzero(slice > vvv) == 0:
            break
        target = int(np.mean(np.argwhere(slice>vvv)))
        coord_list = np.arange(xN) - target
        try:
            r0L,r0R,dr,vL,vR = tanh_fit(slice, coord_list, prior)
            prior = r0L,r0R,dr,vL,vR
            data_list.append([r0L+target, r0R+target])
        except:
            break

    # right wall travels along with COM and left wall against
    data_list = np.asarray(data_list)[::-1]
    ll, rr = data_list[:,0], data_list[:,1]

    # fit walls to hyperbola
    ttwallfit = np.arange(tr, tt, -1)[::-1]
    llwallfit = hypfit_left_mover(ttwallfit, ll)
    rrwallfit = hypfit_right_mover(ttwallfit, rr)
    if len(llwallfit)==0 or len(rrwallfit)==0:
        return 'nan'

    # get velocities from derivative of best fit to wall trajectory
    uu, vv, aa, bb = get_velocities(rrwallfit, llwallfit)
    try:
        vCOM = aa[np.nanargmin(np.abs(uu - vv))]
    except:
        return 'nan'

    if plots:
        fig, [ax0, ax1] = plt.subplots(1, 2, figsize = (12, 5))
        win = 50
        ext = [max(0, x_centre-win), min(nN-1, x_centre+win), max(0, t_centre-win), min(nT-1, t_centre+win)]
        sni2plot = simulation[max(0,t_centre-win): min(nT-1,t_centre+win), max(0,x_centre-win): min(nN-1,x_centre+win)]
        ax0.imshow(sni2plot, aspect='auto', interpolation='none', extent=ext, origin='lower', cmap='PiYG')
        ax0.plot(rr+xl, ttwallfit, 'b')
        ax0.plot(ll+xl, ttwallfit, 'y')
        ax0.plot(rrwallfit+xl, ttwallfit, 'r:')
        ax0.plot(llwallfit+xl, ttwallfit, 'r:')
        ax0.plot(x_centre,t_centre,'ro')
        
        ax0.axhline(tl, color='darkgray', ls='-.')
        ax0.axhline(tr, color='darkgray', ls='-.')
        ax0.axvline(xl, color='darkgray', ls='-.')
        ax0.axvline(xr, color='darkgray', ls='-.')
        ax0.set_xlabel('x'); ax0.set_ylabel('t')
        ax0.legend(title=r'$xx=$'+str(xxx)+r', $vv=$'+str('%.2f'%vvv))

        ax1.plot(ttwallfit, uu, 'b:', ttwallfit, vv, 'y:')
        ax1.plot(ttwallfit, aa, 'r', label=r'v COM')
        ax1.plot(ttwallfit, bb, 'g', label=r'v walls')
        ax1.axhline(0, color='darkgray', ls=':')
        ax1.set_xlabel('t'); ax1.set_ylabel('v(t)')
        ax1.legend(); plt.show()
    return vCOM

rapidity = lambda v: np.arctanh(v)
gamma = lambda v: (1. - v**2.)**(-0.5)
fold = lambda beta: 2 if np.abs(beta) > 0.7 else 1
add_velocities = lambda v1,v2: (v1 + v2) / (1. + v1*v2)

def get_totvel_from_list(vels):
    totvel = 0.
    for ii in vels:
        totvel = add_velocities(ii, totvel)
    return totvel

def coord_pair(tt, xx, vCOM, ga, cc):
    t0 = (cc*tt + vCOM*xx) * ga/cc
    x0 = (xx + vCOM*cc*tt) * ga
    return t0, x0

def boost_bubble(old_simulation, nL, light_cone, phi_init, V, vCOM, crit_thresh, crit_rad, normal):
    # create template for new bubble
    t0, x0 = find_nucleation_center(old_simulation[0], phi_init, crit_thresh, crit_rad)    
    simulation = extend_as_needed(old_simulation, nL, normal, t0, x0)

    rest_bubble = np.ones(np.shape(simulation))
    for col, elem in enumerate(rest_bubble):
        rest_bubble[col] = elem * normal[col]

    C, T, N = np.shape(simulation)
    t0, x0 = find_nucleation_center(simulation[0], phi_init, crit_thresh, crit_rad)
    t,  x  = np.linspace(-t0, T-1-t0, T), np.linspace(-x0, N-1-x0, N)

    # deboost
    ga = gamma(vCOM)
    for col, element in enumerate(simulation):
        # interpolate image onto transformed coordinates
        g = intp.interp2d(t, x, element.T, kind = 'linear')
        # evaluate function on the mesh
        for tind, tval in enumerate(t):
            for xind, xval in enumerate(x):
                tlensed, xlensed = coord_pair(tval, xval, vCOM, ga, light_cone)
                rest_bubble[col,tind,xind] = g(tlensed, xlensed).T

    t0, x0 = find_nucleation_center(rest_bubble[0], phi_init, crit_thresh, crit_rad)    
    final_bubble = truncate_as_needed(rest_bubble, nL, t0, x0)
    return final_bubble

def truncate_as_needed(wheat, nL, t0, x0):
    C, T, N = np.shape(wheat)
    if np.abs(T-t0)>nL//2:
        wheat = wheat[:,:t0+nL//2]
    if t0>3*nL:
        wheat = wheat[:,t0-3*nL:]

    if np.abs(N-x0)>nL:
        wheat = wheat[:,:,:x0+nL]
    if x0>nL:
        wheat = wheat[:,:,x0-nL:]
    return wheat

def extend_as_needed(wheat, nL, normal, t0, x0):
    C, T, N = np.shape(wheat)

    if np.abs(T-t0)<nL//2:
        fusilli = np.empty((C,T+nL,N))
        for col, elem in enumerate(wheat):
            box = np.ones((nL,N)) * normal[col]
            fusilli[col] = np.concatenate((box,elem))
    else:
        fusilli = wheat

    wheat = fusilli
    C, T, N = np.shape(fusilli)
    if np.abs(N-x0)<nL//2:
        fusilli = np.empty((C,T,N+nL//2))
        for col, elem in enumerate(wheat):
            box = np.ones((T,nL//2)) * normal[col]
            fusilli[col] = np.concatenate((elem,box), axis=1)
    else:
        fusilli = wheat

    wheat = fusilli
    C, T, N = np.shape(fusilli)
    if x0<nL//2:
        fusilli = np.empty((C,T,N+nL//2))
        for col, elem in enumerate(wheat):
            box = np.ones((T,nL//2)) * normal[col]
            fusilli[col] = np.concatenate((box,elem), axis=1)
    else:
        fusilli = wheat
    return fusilli



#### Tools for averaging bubbles

def quadrant_coords(real, phi_init, crit_thresh, crit_rad, plots, maxwin, typex):
    nC, nT, nN = np.shape(real)
    if typex==1:
        tcen, xcen = find_nucleation_center(real[0], phi_init, crit_thresh, crit_rad)
    elif typex==2:
        tcen, xcen = find_nucleation_center2(real[0], phi_init, crit_thresh, crit_rad)
    else:
        print('There are only two options. Defaulting back to option 1.')
        tcen, xcen = find_nucleation_center2(real[0], phi_init, crit_thresh, crit_rad)

    aa,bb = max(0, xcen-maxwin), min(nN-1, xcen+maxwin)
    cc,dd = max(0, tcen-maxwin), min(nT-1, tcen+maxwin)

    aaL, bbL = np.arange(aa, xcen), np.arange(xcen, bb+1)
    ccL, ddL = np.arange(cc, tcen), np.arange(tcen, dd+1)

    ddd,bbb = np.meshgrid(ddL,bbL,sparse='True')
    upright_quad = real[:,ddd,bbb]
    ddd,aaa = np.meshgrid(ddL,aaL,sparse='True')
    upleft_quad = real[:,ddd,aaa]
    ccc,bbb = np.meshgrid(ccL,bbL,sparse='True')
    lowright_quad = real[:,ccc,bbb]
    ccc,aaa = np.meshgrid(ccL,aaL,sparse='True')
    lowleft_quad = real[:,ccc,aaa]

    if False:
        fig, ax = plt.subplots(2,2,figsize=(9,7))
        ext00, ext01 = [bbL[0],bbL[-1],ddL[0],ddL[-1]], [aaL[0],aaL[-1],ddL[0],ddL[-1]]
        ax[0,0].imshow(upright_quad[0], interpolation='none', extent=ext00, origin='lower', cmap='PiYG')
        ax[0,0].set_xlabel('x'); ax[0,0].set_ylabel('t')
        ax[0,1].imshow(upleft_quad[0], interpolation='none', extent=ext01, origin='lower', cmap='PiYG')
        ax[0,1].set_xlabel('x'); ax[0,1].set_ylabel('t')

        ext10, ext11 = [bbL[0],bbL[-1],ccL[0],ccL[-1]], [aaL[0],aaL[-1],ccL[0],ccL[-1]]
        ax[1,0].imshow(lowright_quad[0], interpolation='none', extent=ext10, origin='lower', cmap='PiYG')
        ax[1,0].set_xlabel('x'); ax[1,0].set_ylabel('t')
        ax[1,1].imshow(lowleft_quad[0], interpolation='none', extent=ext11, origin='lower', cmap='PiYG')
        ax[1,1].set_xlabel('x'); ax[1,1].set_ylabel('t'); plt.show()
    return upright_quad, upleft_quad, lowright_quad, lowleft_quad

def stack_bubbles(data, plots, maxwin, phi_init, crit_thresh, crit_rad, typex):
    upright_stack, upleft_stack, lowright_stack, lowleft_stack = ([] for ii in range(4))
    for real, velexamp, sim in data:
        if sim%50==0: print('Sim', sim)
        print('sim', sim)
        nC, nT, nN = np.shape(real)

        if plots:
            tcen, xcen = find_nucleation_center2(real[0], phi_init, crit_thresh, crit_rad)
            tl,tr = max(0, tcen-maxwin), min(nT-1, tcen+maxwin)
            xl,xr = max(0, xcen-maxwin), min(nN-1, xcen+maxwin)

            fig = plt.figure(figsize = (5, 4))
            ext = [xl,xr,tl,tr]
            plt.title('Sim '+str(sim)+' '+str(velexamp))
            plt.imshow(real[0,tl:tr,xl:xr], interpolation='none', extent=ext, origin='lower', cmap='PRGn')
            plt.plot(xcen,tcen,'bo')
            plt.xlabel('x'); plt.ylabel('t'); plt.show()

        ur, ul, lr, ll = quadrant_coords(real, phi_init, crit_thresh, crit_rad, plots, maxwin, typex)
        upright_stack.append(ur)
        upleft_stack.append(ul)
        lowright_stack.append(lr)
        lowleft_stack.append(ll)
    return upright_stack, upleft_stack, lowright_stack, lowleft_stack

def average_stacks(data, plots, normal):
    upright_stack, upleft_stack, lowright_stack, lowleft_stack = data
    nS = len(upright_stack)
    nC = len(upleft_stack[0])

    av_mat, av_err_mat = [], []
    for col in range(nC):
        av_mat.append([])
        av_err_mat.append([])
        for ijk, corner in enumerate(data): #for each quadrant
            max_rows = np.asarray([len(corner[ss][col][:,0]) for ss in range(len(corner))])
            max_cols = np.asarray([len(corner[ss][col][0,:]) for ss in range(len(corner))])
            average_matrix = np.ones((np.amax(max_rows), np.amax(max_cols)))*normal[col]
            copy = np.asarray([average_matrix] * len(corner))

            for ss, simulation in enumerate(corner):
                real = simulation[col]
                nT,nN = np.shape(real)
                if ijk%2==0:
                    copy[ss,:nT,:nN] = real
                else:
                    real = real[::-1,:]
                    copy[ss,:nT,:nN] = real

            av_mat[col].append(np.nanmean(copy, axis=0))
            if col == 0 and plots:
                nT,nN = np.shape(av_mat[col][-1])
                ext = [0,nN,0,nT]
                plt.imshow(av_mat[col][-1], interpolation='none', extent=ext, origin='lower', cmap='viridis'); plt.show()
            av_err_mat[col].append(np.nanstd(copy, axis=0)/len(copy))
    return av_mat, av_err_mat

def average_bubble_stacks(data):
    whole_bubbles = []
    for bb, bub in enumerate(data):
        whole_bubbles.append([])
        for col, bubcol in enumerate(bub):
            top = np.concatenate((bubcol[1][::-1], bubcol[0]), axis=0)
            bottom = np.concatenate((bubcol[3][::-1], bubcol[2]), axis=0)
            whole_bubbles[-1].append(np.concatenate((bottom, top), axis=1).transpose())
    return np.asarray(whole_bubbles)


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
