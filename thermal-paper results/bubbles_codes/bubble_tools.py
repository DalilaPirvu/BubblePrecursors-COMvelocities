import os, sys
import numpy as np
import random

from functools import partial
from itertools import cycle

import scipy as scp
import scipy.optimize as sco
import scipy.signal as scs
import scipy.interpolate as intp

from scipy.integrate import odeint
from scipy.signal import find_peaks, peak_widths

from scipy.interpolate import interp2d,interp1d
import scipy.interpolate as si

import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d


from plotting import *
from experiment import *


def extract_data(nL, path_sim):
    data = np.genfromtxt(path_sim)
    nNnT, nC = np.shape(data)
    return np.asarray([np.reshape(data[:,cc], (nNnT//nL, nL)) for cc in range(nC)])

def get_realisation(nL, sim, phieq, path_sim):
    data = extract_data(nL, path_sim)
    slice = data[0,-1,:]
    outcome = check_decay(slice, phieq)

    if outcome == 1:
        data[0] = 2.*phieq - data[0]
        data[1] = - data[1]
        data[2] = - data[2]
    return np.asarray(data), outcome

def check_decay(slice, phieq):
    refl_slice = reflect_against_equil(slice, phieq)

    right_phi = np.count_nonzero(slice > phieq+np.pi)
    left_phi  = np.count_nonzero(slice < phieq-np.pi)
    if right_phi > left_phi:
        return 0
    else:
        return 1

def remove_collisions(real, phi_init):
    nC, nT, nN = np.shape(real)
    trc = max(0, nT-nN)
    bubble  = real[0, trc:]

    c_t = np.mean(np.cos(bubble), axis=-1)
    smooth_c_t = gaussian_filter(c_t, nN//50, mode='nearest')
    tcut = np.argmax(smooth_c_t)
    return real[:, :(trc + tcut)]

def centre_bubble(real, tdecay, phi_init, crit_thresh):
    nC, nT, nN = np.shape(real)
    critbubble = real[0, tdecay, :]
    try:
        x_centre, crit_rad = find_middle(critbubble, phi_init, crit_thresh)
        real = np.roll(real, nN//2 - x_centre, axis=-1)
    except:
        real = reflect_against_equil(real, phi_init)
        critbubble = real[0, tdecay, :]

        x_centre, crit_rad = find_middle(critbubble, phi_init, crit_thresh)
        real = np.roll(real, nN//2 - x_centre, axis=-1)

    # apply twice to be sure
    critbubble = real[0, tdecay, :]
    x_centre, crit_rad = find_middle(critbubble, phi_init, crit_thresh)
    real = np.roll(real, nN//2 - x_centre, axis=-1)

    # apply thrice to be sure
    critbubble = real[0, tdecay, :]
    x_centre, crit_rad = find_middle(critbubble, phi_init, crit_thresh)
    real = np.roll(real, nN//2 - x_centre, axis=-1)

    tamp = max(0, nT - 10*nN)
    return real[:, tamp:], crit_rad/2.

def bubble_counts_at_fixed_t(bubble, thresh):
    return np.count_nonzero(bubble > thresh, axis=1)

def bubble_counts_at_fixed_x(bubble, thresh):
    return np.count_nonzero(bubble > thresh, axis=0)

def reflect_against_equil(bubble, phi_init):
    return np.abs(bubble-phi_init) + phi_init

def find_middle(critslice, phi_init, crit_thresh):
    coords = np.argwhere(critslice > crit_thresh)
    return int(round(np.mean(coords))), len(coords)

def find_nucleation_center(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)
    bubble_counts = bubble_counts_at_fixed_t(bubble, crit_thresh)
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    bubble_counts = bubble_counts_at_fixed_x(bubble[:int(min(t0+crit_rad,T))], crit_thresh)
    x0 = np.argmax(bubble_counts)
    return min(T-1,t0), min(N-1,x0)

def find_nucleation_center2(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)
    bubble_counts = bubble_counts_at_fixed_t(bubble, crit_thresh)
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    slice = bubble[t0, int(N/2-crit_rad*2):int(N/2+crit_rad*2)]
    bubble_counts = np.argwhere(slice>crit_thresh)

    x0 = int(np.mean(bubble_counts))
    return min(T-1,t0), min(N-1,x0+int(N/2-crit_rad*2))

def find_t_max_width(bubble, light_cone, phi_init, crit_thresh, crit_rad, t0):
    T, N = np.shape(bubble)
    bubble = bubble[t0:]
    refl_bubble = reflect_against_equil(bubble, phi_init)

    bubble_counts = bubble_counts_at_fixed_t(refl_bubble, crit_thresh)
    bubble_diffs = bubble_counts[1:] - bubble_counts[:-1]

    tmax = np.argwhere(bubble_diffs[::-1] >= light_cone*2).flatten()
    out = next((ii for ii, jj in zip(tmax[:-1], tmax[1:]) if jj-ii == 1), 1)
    return T-out


def multiply_bubble(bubble, light_cone, phi_init, vCOM, normal, nL):
    # multiplies bubbles so causal tail is unfolded from PBC
    if vCOM<0:
        bubble = bubble[:,:,::-1]
    C, T, N = np.shape(bubble)
    bubble = np.asarray([np.tile(bubble[col], fold(vCOM)) for col in range(C)])
    TT, NN = np.shape(bubble[0])
    mn = np.mean(bubble[0])
    for t in range(TT):
        a, b = int((TT-t)/light_cone) + N, int((TT-t)/light_cone/3.) - N//4
        x1, x2 = np.arange(a, NN), np.arange(b)
        x1, x2 = x1 - a, x2 - (b-NN)
        for x in np.append(x1, x2):
            if 0 <= x < NN:
                bubble[0,t,x] = mn
    if vCOM<0:
        bubble = bubble[:,:,::-1]
    return bubble

def retired_tanh_profile(x, r0L, r0R, vL, vR, dr, a):
    wL, wR = dr/gamma(vL), dr/gamma(vR)
    return ( np.tanh( (x - r0L)/wL ) + np.tanh( (r0R - x)/wR ) ) * a + np.pi

def tanh_profile(x, r0L, r0R, vL, vR):
    return ( np.tanh( (x - r0L)/vL ) + np.tanh( (r0R - x)/vR ) ) * np.pi/2. + np.pi

def tanh_profile2(x, r0L, r0R, vL, vR, a):
    return ( np.tanh(vL * x - r0L) + np.tanh(r0R - vR * x) ) * a + np.pi

def get_profile_bf(xlist, phibubble, prior):
	bounds = ((xlist[0], 0, 0, 0, 0, 0.*np.pi/2.), (0, xlist[-1], 1, 1, xlist[-1], 1.5*np.pi/2.))
	tanfit, _ = sco.curve_fit(retired_tanh_profile, xlist, phibubble, p0=prior, bounds=bounds)
	return tanfit

def hypfit_right_mover(tt, rr):
#    hyperbola  = lambda t, a, b, c: np.sqrt(c + (t - b)**2.) + a
    hyperbola  = lambda t, a, b, c: np.sqrt(c + b*t + t**2.) + a
    try:
        prior   = (float(min(rr)), float(tt[np.argmin(rr)]), 1e3)
        fit, _  = sco.curve_fit(hyperbola, tt, rr, p0 = prior)
        traj    = hyperbola(tt, *fit)
        return traj
    except:
        return []

def hypfit_left_mover(tt, ll):
#    hyperbola  = lambda t, d, e, f: - np.sqrt(f + (t - e)**2.) + d
    hyperbola  = lambda t, d, e, f: - np.sqrt(f + e*t + t**2.) + d
    try:
        prior   = (float(max(ll)), float(tt[np.argmax(ll)]), 1e3)
        fit, _  = sco.curve_fit(hyperbola, tt, ll, p0 = prior)
        traj    = hyperbola(tt, *fit)
        return traj
    except:
        return []

def get_velocities(rrwallfit, llwallfit, dx):
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

def find_COM_vel(real, fldamp, winsize, nL, light_cone, phi_init, crit_thresh, crit_rad, dx, lamb, plots=False):
    nC, nT, nN = np.shape(real)
    real = real[0]
    t_maxwid = find_t_max_width(real, light_cone, phi_init, crit_thresh, crit_rad, nT-nL)
    edge = np.abs(nT-t_maxwid)

    t_centre, x_centre = find_nucleation_center(real[(t_maxwid-nL//2):t_maxwid, edge:(nN-edge)], phi_init, crit_thresh, crit_rad)
    x_centre += edge
    t_centre += max(t_maxwid - nL//2, 0.)

    tl_stop, tr_stop = max(0., t_centre - winsize), min(nT, t_centre + winsize//2)
    xl_stop, xr_stop = max(0., x_centre - winsize), min(nN, x_centre + winsize)

#    real = gaussian_filter(real, nL/1000, mode='nearest')
    simulation = real[tl_stop:tr_stop, xl_stop:xr_stop]
    tcen, xcen = find_nucleation_center(simulation, phi_init, crit_thresh, crit_rad)

    betas = np.zeros((len(fldamp)))
    for vv, v_size in enumerate(fldamp):
        vel_plots = (True if vv%2==0 and plots else False)
        #simple_imshow(simulation, [0,tr_stop-tl_stop,0,xr_stop-xl_stop], 'tr')

        betas[vv] = get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, v_size, dx, lamb, tcen, xcen, vel_plots)
    if plots: plt.plot(fldamp, betas, marker='o'); plt.title(r'Mean $v = $'+str('%.2f'%np.nanmean(betas))); plt.show()

    return np.nanmean(betas)

def get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, vvv, dx, lamb, tcen, xcen, plots=False):
    nT, nN = np.shape(simulation)
    data_list, prior, target = [], None, nN/2.
    for tt in reversed(range(nT)):
        slice = simulation[tt]

        try: target = int(np.round(np.nanmean(np.argwhere(slice > vvv))))
        except: break

        coord_list = np.arange(nN) - target
        try:
        	prior = get_profile_bf(coord_list, slice, prior)
        	r0L,r0R,vL,vR,_,_ = prior

        	curve = retired_tanh_profile(coord_list, *prior)
        #	int0 = np.argwhere(coord_list == 0.)
        #	ind1 = int0 + np.argmin(np.abs(slice[coord_list>=0] - vvv))
        #  	ind2 = np.argmin(np.abs(slice[coord_list<0] - vvv))
        #  	r0L, r0R = coord_list[ind1], coord_list[ind2]
        	data_list.append([tt, r0L+target, r0R+target])

        	if False:
        		if tt%100!=0: continue
        		plt.plot(coord_list, slice, 'bo', ms=1)
        		plt.plot(coord_list, curve, 'go', ms=1)
        		plt.axhline(vvv, ls=':', color='k')
        		plt.title('t='+str(tt)); plt.show()

        except:
            continue

    # get velocities from derivatives of trajectories
    try:
        # right wall travels along with COM and left wall against
        data_list = np.array(data_list)[::-1]
        ttwallfit, ll, rr = data_list[:,0], data_list[:,1] - xcen, data_list[:,2] - xcen

        # fit walls to hyperbola
        llwallfit = hypfit_left_mover(ttwallfit, ll)
        rrwallfit = hypfit_right_mover(ttwallfit, rr)

        if False:
            fig, ax0 = plt.subplots(1, 1, figsize = (3, 3))
            ax0.plot(rr*np.sqrt(m2(lamb))*dx, ttwallfit*np.sqrt(m2(lamb))*dx, color='blueviolet', ls=':', linewidth=1, label='rr')
            ax0.plot(ll*np.sqrt(m2(lamb))*dx, ttwallfit*np.sqrt(m2(lamb))*dx, color='orangered', ls=':', linewidth=1, label='ll')
            try:
	            ax0.plot(rrwallfit*np.sqrt(m2(lamb))*dx, ttwallfit*np.sqrt(m2(lamb))*dx, color='blueviolet', ls='-.', linewidth=1, label='r hyp')
	            ax0.plot(llwallfit*np.sqrt(m2(lamb))*dx, ttwallfit*np.sqrt(m2(lamb))*dx, color='orangered', ls='-.', linewidth=1, label='l hyp')
            except:
	            ax0.axhline(ttwallfit[0])
            ax0.set_xlabel('x'); ax0.set_ylabel('t')
            ax0.legend(title=r'$\phi=$'+str(vvv)); plt.show()

        uu, vv, aa, bb = get_velocities(rrwallfit, llwallfit, dx)
        indix = np.nanargmin(np.abs(uu - vv))
        vCOM = aa[indix]
    except:
        return 'nan'

    if plots:
        fig, ax = plt.subplots(1, 1, figsize = (3, 2))
     #   tcen, xcen = find_nucleation_center(simulation, phi_init, crit_thresh, crit_rad)
        dx2plot = np.sqrt(m2(lamb))*dx
        ext = np.array([-xcen, nN-xcen, -tcen, nT-tcen])*dx2plot
        im0 = ax.imshow(simulation, interpolation='antialiased', extent=ext, origin='lower', cmap='RdBu', aspect='auto')

        ax.plot(rr*dx2plot, (ttwallfit-tcen)*dx2plot, color='k', ls='--', linewidth=0.5)
        ax.plot(ll*dx2plot, (ttwallfit-tcen)*dx2plot, color='k', ls='--', linewidth=0.5)

        ax.plot(rrwallfit*dx2plot, (ttwallfit-tcen)*dx2plot, color='k', ls='-', linewidth=0.7)
        ax.plot(llwallfit*dx2plot, (ttwallfit-tcen)*dx2plot, color='k', ls='-', linewidth=0.7)

        cbar = plt.colorbar(im0, ax=ax)#, ticks=mticker.MultipleLocator(np.pi/2), format=mticker.FuncFormatter(multiple_formatter()))
        cbar.ax.set_title(r'$\bar{\phi}$')
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.grid(ls=':', color='darkgray', alpha=0.5)
        ax.set(ylabel=r'$m \, t$'); ax.set(xlabel=r'$m \, r$')
        ax.set_aspect(1.5)
     #   plt.savefig('./plots/sim_original2.pdf', rasterize=True, dpi=500)
     #   plt.savefig('./plots/sim_deboosted2.pdf', rasterize=True, dpi=500)
        plt.show()
      #  np.save('./data/bubble_data_original_1.npy', np.array([rr, ll, rrwallfit, llwallfit, ttwallfit-tcen, simulation, ext]))
      #  np.save('./data/bubble_data_deboosted_1.npy', np.array([rr, ll, rrwallfit, llwallfit, ttwallfit-tcen, simulation, ext]))
      #  np.save('./data/bubble_data_original_2.npy', np.array([rr, ll, rrwallfit, llwallfit, ttwallfit-tcen, simulation, ext]))
      #  np.save('./data/bubble_data_deboosted_2.npy', np.array([rr, ll, rrwallfit, llwallfit, ttwallfit-tcen, simulation, ext]))

    return vCOM

rapidity = lambda v: np.arctanh(v)
gamma    = lambda v: (1. - v**2.)**(-0.5)
fold     = lambda beta: 3
#fold     = lambda beta: 2 if 0.8 > np.abs(beta) > 0.7 else 3 if np.abs(beta) > 0.8 else 1
addvels  = lambda v1,v2: (v1 + v2) / (1. + v1*v2)

def get_totvel_from_list(vels):
    totvel = 0.
    for ii in vels:
        totvel = addvels(ii, totvel)
    return totvel

def coord_pair(tt, xx, vCOM, ga, cc):
    t0 = (cc*tt + vCOM*xx) * ga/cc
    x0 = (xx + vCOM*cc*tt) * ga
    return t0, x0

def boost_bubble(simulation, nL, light_cone, phi_init, vCOM, crit_thresh, crit_rad, normal):
    # create template for new bubble
    rest_bubble = np.ones(np.shape(simulation))
    for col, elem in enumerate(rest_bubble):
        rest_bubble[col] = elem * normal[col]

    C, T, N = np.shape(simulation)
    t0, x0 = find_nucleation_center(simulation[0], phi_init, crit_thresh, crit_rad//3)
    t, x = np.linspace(-t0, T-1-t0, T), np.linspace(-x0, N-1-x0, N)

    # boost factor
    ga = gamma(vCOM)
    for col, element in enumerate(simulation):

        # interpolate image onto 2d rectangular grid
        g = interp2d(t, x, element.T, kind = 'cubic', bounds_error=True)

        # evaluate function on the transformed coordinates
#        for tind, tval in enumerate(t):
#            for xind, xval in enumerate(x):
#                tlensed, xlensed = coord_pair(tval, xval, vCOM, ga, light_cone)
#                rest_bubble[col,tind,xind] = g(tlensed, xlensed).T
        for tind, tval in enumerate(t):
            tlensed, xlensed = coord_pair(tval, x, vCOM, ga, light_cone)
            interpolated = si.dfitpack.bispeu(g.tck[0], g.tck[1], g.tck[2], g.tck[3], g.tck[4], tlensed, xlensed)[0]
            rest_bubble[col,tind,:] = interpolated.T
    return rest_bubble


#### Tools for averaging bubbles

def quadrant_coords(real, phi_init, crit_thresh, crit_rad, maxwin, plots=False):
    nC, nT, nN = np.shape(real)
    tcen, xcen = find_nucleation_center(real[0], phi_init, crit_thresh, crit_rad)
    if tcen == 0 or tcen == nT-1 or xcen == 0 or xcen == nN-1:
        return None

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

def stack_bubbles(data, maxwin, phi_init, crit_thresh, crit_rad, plots=False):
    upright_stack, upleft_stack, lowright_stack, lowleft_stack = ([] for ii in range(4))
    for sim, real in data:
        #if sim%50==0: print('Sim', sim)
        nC, nT, nN = np.shape(real)

        if plots:
            tcen, xcen = find_nucleation_center(real[0], phi_init, crit_thresh, crit_rad)
            tl,tr = max(0, tcen-maxwin), min(nT-1, tcen+maxwin)
            xl,xr = max(0, xcen-maxwin), min(nN-1, xcen+maxwin)

            fig = plt.figure(figsize = (5, 4))
            ext = [xl,xr,tl,tr]
            plt.title('Sim '+str(sim))
            plt.imshow(real[0,tl:tr,xl:xr], interpolation='none', extent=ext, origin='lower', cmap='PRGn')
            plt.plot(xcen,tcen,'bo')
            plt.xlabel('x'); plt.ylabel('t'); plt.show()

        try:
            ur, ul, lr, ll = quadrant_coords(real, phi_init, crit_thresh, crit_rad, maxwin, plots)
        except:
            print('Skip sim', sim)
            continue
        upright_stack.append(ur)
        upleft_stack.append(ul)
        lowright_stack.append(lr)
        lowleft_stack.append(ll)
    return upright_stack, upleft_stack, lowright_stack, lowleft_stack

def average_stacks(data, normal, plots=False):
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
                plt.imshow(av_mat[col][-1], interpolation='none', extent=ext, origin='lower', cmap='viridis')
                plt.show()
            av_err_mat[col].append(np.nanvar(copy, axis=0)/len(copy))
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

def space_save(real, light_cone, phi_init, crit_thresh, crit_rad, nL):
    nT, nN = np.shape(real[0])

    t_maxwid = find_t_max_width(real[0], light_cone, phi_init, crit_thresh, crit_rad, nT-nL//2)
    edge = np.abs(nT - t_maxwid)
    tcen, xcen = find_nucleation_center(real[0,:t_maxwid, edge:(nN-edge)], phi_init, crit_thresh, crit_rad)
    xcen += edge

    win = nL//4
    tl,tr = max(0, tcen-win), min(nT-1, tcen+win)
    xl,xr = max(0, xcen-win), min(nN-1, xcen+win)
    return real[:,tl:tr,xl:xr]


def lin_fit_times(times,num,tmin,tmax):
    """
    Given a collection of decay times, do a linear fit to
    the logarithmic survival probability between given times

    Input
      times : array of decay times
      num   : original number of samples
      tmin  : minimum time to fit inside
      tmax  : maximum time to fit inside
    """
    t = np.sort(times)
    p = survive_prob(times, num)
    ii = np.where( (t>tmin) & (t<tmax) )
    return np.polyfit(t[ii], np.log(p[ii]), deg=1)


# To do: Debug more to ensure all offsets are correct.
# I've done a first go through and I think they're ok
def survive_prob(t_decay, num_samp):
    """
    Return the survival probability as a function of time.

    Input:
      t_decay  : Decay times of trajectories
      num_samp : Total number of samples in Monte Carlo

    Note: Since some trajectories may not decay, t_decay.size isn't always num_sampe

    Output:
      prob     : Survival Probability

    These can be plotted as plt.plot(t_sort,prob) to get a survival probability graph.
    """
    frac_remain = float(num_samp-t_decay.size)/float(num_samp)
    prob = 1. - np.linspace(1./num_samp, 1.-frac_remain, t_decay.size, endpoint=True)
    return prob

def get_line(dataset, slope, offset):
    return dataset * slope + offset

def f_surv(times, ntot):
    return np.array([1.-len(times[times<=ts])/ntot for ts in times])
