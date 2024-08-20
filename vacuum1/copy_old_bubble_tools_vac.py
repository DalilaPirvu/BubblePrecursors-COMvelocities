# To run this script, in a separate terminal type:
#### python3 old_deboost.py >> output.txt
from bubble_tools_vac import *

import time
import functools
from concurrent.futures import ProcessPoolExecutor

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


#act = False
act = True


### Paramns 
nLat        = 512
nTimeMAX    = 10000
lSims       = 0
nSims       = 1000
lamb        = 1.5

Tlist       = np.asarray([0., 0.1])
phi0List    = np.asarray([2.*np.pi/3., 2.*np.pi/4.])

#### CONSTANTS
knyq        = nLat//2+1
nu          = 2e-3
omega       = 0.25*50.*2.*nu**0.5
delt        = (nu/2.)**0.5*lamb
rho         = 200.*2.*(nu)**0.5*2.**(-3)
m2eff       = 4.*nu*(-1.+lamb**2)
lenLat      = 2 * 50. / (2.*nu)**0.5
dx          = lenLat/nLat
dk          = 2.*np.pi/lenLat

alph        = 8.
phi_init    = np.pi
mask        = 4*phi_init
nCols       = 2
dt          = dx/alph
dtout       = dt*alph
light_cone  = dx/dtout

# Lattice
lattice     = np.arange(nLat)
xlist       = lattice*dx
klist       = np.roll((lattice - nLat//2)*dk, nLat//2)
inv_phases  = np.exp(1j*np.outer(xlist, klist))
dit_phases  = nLat**-1. * np.exp(-1j*np.outer(xlist, klist))

#### SPECTRA
# Free field (constant mass term) field modes \phi_k
norm                  = lambda ph0: 1./ ph0 / np.sqrt(2. * lenLat)

w2                    = lambda m2: klist**2. + m2
free_eigenbasis       = lambda m2, ph0: np.asarray([norm(ph0) / w2(m2)[k]**0.25 if kk!=0 else 0. for k,kk in enumerate(klist)])
free_pspec            = lambda m2, ph0: np.abs(free_eigenbasis(m2, ph0))**2
free_fluct_stdev      = lambda m2, ph0: np.sqrt(np.sum(free_pspec(m2, ph0)))

thermal_eigenbasis    = lambda m2, ph0, te: free_eigenbasis(m2, ph0) * np.sqrt(2./(np.exp(w2(m2)**0.5/te)-1.))
thermal_pspec         = lambda m2, ph0, te: np.abs(thermal_eigenbasis(m2, ph0, te))**2
thermal_fluct_stdev   = lambda m2, ph0, te: np.sqrt(np.sum(thermal_pspec(m2, ph0, te)))

pspec         = lambda m2, ph0, te: thermal_pspec(m2, ph0, te) if te!=0 else free_pspec(m2, ph0)
fluct_stdev   = lambda m2, ph0, te: thermal_fluct_stdev(m2, ph0, te) if te!=0 else free_fluct_stdev(m2, ph0)


### POTENTIAL
V    = lambda x: -(-np.cos(x) + 0.5 * lamb**2. * np.sin(x)**2. - 1) * 4. * nu
dV   = lambda x: -( np.sin(x) + 0.5 * lamb**2. * np.sin(2.*x)) * 4. * nu

right_Vmax  = sco.minimize_scalar(V, bounds=(np.pi, 2*np.pi), method='bounded')
left_Vmax   = sco.minimize_scalar(V, bounds=(0, np.pi), method='bounded')

upper_phi_bound  = sco.fsolve(V, 5.5)[0]
lower_phi_bound  = sco.fsolve(V, 0.5)[0]

### Paths to files
root_dir      = '/gpfs/dpirvu/velocity_comparison/'
batch_params  = lambda nL, la, ph, te: 'x'+str(nL)+'_phi0'+str('%.4f'%ph)+'_lambda'+str('%.4f'%la)+'_T'+str('%.4f'%te) 

sim_location           = lambda nL, la, ph, te, sim: root_dir + batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields.dat'
clean_sim_location     = lambda nL, la, ph, te, sim: root_dir + 'clean_' + batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields'
bubble_at_rest         = lambda nL, la, ph, te, sim: root_dir + 'rest_bubble_' + batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields'

directions_bubbles     = lambda nL, la, ph, te: root_dir + 'directions_' + batch_params(nL,la,ph,te)
COMvelocities_bubbles  = lambda nL, la, ph, te: root_dir + 'velocitiesCOM_' + batch_params(nL,la,ph,te)

titles = [r'$\phi(x)$', r'$\partial_t \phi(x)$', r'$|\nabla \phi(x)|^2$', r'$V(\phi(x))$']




amp_thresh    = right_Vmax.x
crit_thresh   = right_Vmax.x
tv_thresh     = right_Vmax.x
crit_rad      = 40
window        = 100

if act:
    for tmp, temp in enumerate(Tlist):
        print('Looking at at temperature T = ', temp)
        ph0        = phi0List[tmp]
        sigmafld   = fluct_stdev(m2eff, ph0, temp)
        ampList    = np.linspace(phi_init + 2.8*sigmafld, phi_init + 5.*sigmafld, 20)
        xList      = np.arange(window//2, 3*window//2, 1)
        print(ph0, sigmafld, m2eff, temp, ampList)

        for sim in range(lSims, nSims):

            loc_sim = clean_sim_location(nLat, lamb, ph0, temp, sim)+'.npy'
            if not os.path.exists(loc_sim):
                print('No nucleation in realisation', sim)
                continue

            try:
                real, sim = np.load(loc_sim)
                print('Starting simulation', sim)

                beta, medbeta, stbeta = find_COM_vel(real, ampList, xList, nLat, light_cone, phi_init, crit_thresh, crit_rad, False)
                new_real = multiply_bubble(real, light_cone, phi_init, beta)

                bool, vellist = True, []
                while (np.abs(beta) > 0.05) and (np.abs(medbeta) > 0.05):
                    if np.abs(beta)>0.8:
                        beta = np.sign(beta)*0.8
                    vellist.append(beta)
                    new_real = boost_bubble(new_real, nLat, light_cone, phi_init, V, beta, crit_thresh, crit_rad)

                    beta, medbeta, stbeta = find_COM_vel(new_real, ampList, xList, nLat, light_cone, phi_init, crit_thresh, crit_rad, False)
                    if np.isnan(beta):
                        bool = False
                        print('Dead end.')
                        break

                if bool:
                    totbeta = get_totvel_from_list(vellist)
                    np.save(bubble_at_rest(nLat, lamb, ph0, temp, sim), [new_real, np.asarray([sim, totbeta, beta])])
                    print('Total vel, final vel, vel list:', totbeta, beta, vellist)
            except:
                print('Simulation'+str(sim)+' skipped due to unknown error.')
                continue
        print('Continuing with next set of parameters.')
    print('All Done.')            


if False:
    
            partialFunctionCls = functools.partial(get_scrCLs, cellDeltaTau, unlensedCL, ell2max)            
            with ProcessPoolExecutor(num_workers) as executor:
                CMBDP_powspec = list(executor.map(partialFunctionCls, ells, chunksize=max(1,len(ells)//num_workers)))
            
            np.save(dirdata(MA, omega0, nZs, nMasses, ellMax)+'_CMBDP_dell'+str(dell)+'_ell2max'+str(ell2max), CMBDP_powspec)
 

            for sim in range(lSims, nSims):
                sim_loc = sim_location(nLat, lamb, ph0, temp, sim)
                if os.path.exists(sim_loc):
                    try:
                        data = triage(nLat, nTimeMAX, sim, phi_init, sigmafld, sim_loc)
                        if len(data)!=0:
                            real, sim = data
                            try:
                                real = remove_collisions(real, phi_init, thresh, crit_rad)
                                real = centre_bubble(real, phi_init, thresh, crit_rad)
                                np.save(clean_sim_location(nLat, lamb, ph0, temp, sim), [real,sim])
                            except:
                                continue
                        print('Done ph0, temp, sim', ph0, temp, sim)
                        os.remove(sim_location(nLat, lamb, ph0, temp, sim))
                    except:
                        continue

            
            
###############################################################
###############################################################
                ## Old, discarded functions ##
###############################################################
###############################################################
###############################################################


def find_slice_peaks(field_slice, peak_threshold):
    """ Finds x coordinate of peaks with height above some threshold. """
    peak_coord = scs.find_peaks(field_slice, height = peak_threshold)[0]
    # below we mind potential boundary discontinuities
    if field_slice[-1] > peak_threshold and field_slice[0] > peak_threshold:
        if field_slice[0] > field_slice[-1] and field_slice[0] > field_slice[1]:
            peak_coord.append(0)
        if field_slice[-1] > field_slice[0] and field_slice[-1] > field_slice[-2]:
            peak_coord.append(len(field_slice)-1)
    return peak_coord

def truncateNum(num, decimal_places):
    StrNum = str(num)
    p = StrNum.find(".") + 1 + decimal_places
    return float(StrNum[0:p])

def boundaries_bubble(bubble):
    T, N = np.shape(bubble[0])
    real = np.asarray([bubble[0,:,x] for x in range(N//2)])
    for indxL, lines in enumerate(zip(real, real[1:])):
        if np.mean(lines[0]-lines[1])!=0.:
            break
    real = np.asarray([bubble[0,:,x] for x in range(N-1,N//2,-1)])
    for indxR, lines in enumerate(zip(real, real[1:])):
        if np.mean(lines[0]-lines[1])!=0.:
            break
    real = bubble[0,::-1]
    for indT, lines in enumerate(zip(real, real[1:])):
        if np.mean(lines[0]-lines[1])!=0.:
            break
#    print(T-indT, indxL, N-indxR)
    return bubble[:,:T-indT,indxL:N-indxR]

def direction_bubble(bubble, nLat, fluct_stdev, phi_init, crit_thresh, tv_thresh, amp_thresh, t_snip_size, x_snip_size, crit_rad, window):
    T, N = np.shape(bubble[0])
    bubble = amputate_bubble(bubble, nLat, phi_init, crit_thresh, amp_thresh, fluct_stdev, crit_rad, window)
    t_centre, x_centre = find_nucleation_center(bubble[0], phi_init, crit_thresh, crit_rad)

    tl,tr = max(0, t_centre-t_snip_size), min(T-1, t_centre)
    xl,xr = max(0, x_centre-x_snip_size), min(N-1, x_centre+x_snip_size)

    snip = bubble[:,tl:tr,xl:xr]
    snip = centre_bubble(snip, phi_init, phi_init+2.*fluct_stdev, crit_rad)
    snip = snip[0]
    T, N = np.shape(snip)
    refl_snip = reflect_against_equil(snip, phi_init)
    refl_snip = gaussian_filter(refl_snip, 1, mode='nearest')

    argcounts = np.argwhere(refl_snip[-10:] > phi_init+2.*fluct_stdev)[:,1]
    xmid = int(np.nanmean(argcounts.flatten()))

    counts = np.count_nonzero(((refl_snip > phi_init+2.*fluct_stdev) & (refl_snip<tv_thresh)), axis=0)
#    plt.plot(counts); plt.show()
    try:
        if np.sum(counts[:xmid]) > np.sum(counts[xmid:]):
            return +1.
            #right moving COM
            #hyperbolic trajectory to the left
            #null ray to the right
        else:
            return -1.
    except:
        return 0.

def amputate_bubble(bubble, nL, phi_init, crit_thresh, amp_thresh, fluct_stdev, crit_rad, window):
    t_centre, x_centre = find_nucleation_center(bubble[0], phi_init, crit_thresh, crit_rad)
    T, N = np.shape(bubble[0])
    tdecap = min(t_centre+nL//2, T-1)
    slice = bubble[0, tdecap]
    slice = reflect_against_equil(slice, phi_init)
    extent = np.where(slice > amp_thresh)[0]
    xmin = max(0, extent[0] - window)
    xmax = min(N-1, extent[-1] + window)

    std_real = np.std(bubble[0], axis=-1)
    tamputate = 0
    while std_real[tamputate] <= fluct_stdev:
        tamputate+=1
    return bubble[:, tamputate:tdecap, xmin:xmax]
