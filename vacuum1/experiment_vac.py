from bubble_tools_vac import *

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


### Params
nLat        = 2048
nTimeMAX    = 16384

lambList    = np.asarray([1.4, 1.5])
phi0List    = np.asarray([2.*np.pi/2.8, 2.*np.pi/2.8])

#### CONSTANTS
nu          = 2e-3
lenLat      = 100./(2.*nu)**0.5

phieq       = np.pi
alph        = 8.
temp        = 0.

dx          = lenLat/nLat
dk          = 2.*np.pi/lenLat
knyq        = nLat//2+1

dt          = dx/alph
dtout       = dt*alph
lightc      = dx/dtout


# Lattice
lattice     = np.arange(nLat)
xlist       = lattice*dx
klist       = np.roll((lattice - nLat//2)*dk, nLat//2)
inv_phases  = np.exp(1j*np.outer(xlist, klist))
dit_phases  = nLat**-1. * np.exp(-1j*np.outer(xlist, klist))


#### SPECTRA
# Free field (constant mass term) field modes \phi_k
m2                    = lambda la: 4.*nu*(-1.+la**2.)
norm                  = lambda ph0: 1./ ph0 / np.sqrt(2. * lenLat)
w2                    = lambda la: klist**2. + m2(la)
free_eigenbasis       = lambda la, ph0: np.asarray([norm(ph0)/(w2(la)[k]**0.25) if (kk!=0. and k!=knyq-1) else 0. for k,kk in enumerate(klist)])
free_pspec            = lambda la, ph0: np.abs(free_eigenbasis(la, ph0))**2
free_fluct_stdev      = lambda la, ph0: np.sqrt(np.sum(free_pspec(la, ph0)))

thermal_eigenbasis    = lambda la, ph0, te: free_eigenbasis(la, ph0) * np.sqrt(2./(np.exp(w2(la)**0.5/te)-1.))
thermal_pspec         = lambda la, ph0, te: np.abs(thermal_eigenbasis(la, ph0, te))**2
thermal_fluct_stdev   = lambda la, ph0, te: np.sqrt(np.sum(thermal_pspec(la, ph0, te)))

pspec         = lambda la, ph0, te: thermal_pspec(la, ph0, te) if te!=0 else free_pspec(la, ph0)
fluct_stdev   = lambda la, ph0, te: thermal_fluct_stdev(la, ph0, te) if te!=0 else free_fluct_stdev(la, ph0)


### POTENTIAL
V    = lambda x, la: ( -np.cos(x) + 0.5 * la**2. * np.sin(   x)**2. + 1. ) * 4. * nu
Vfit = lambda x, la: ( -np.cos(x) + 0.5 * la**2. * np.sin(   x)**2. + 1. ) * 4. * nu
dV   = lambda x, la: (  np.sin(x) + 0.5 * la**2. * np.sin(2.*x)          ) * 4. * nu

### Paths to files
root_dir           = '/gpfs/dpirvu/velocity_comparison/'
free_batch_params  = lambda nL, la, ph, te: 'free_x'+str(nL)+'_phi0'+str('%.4f'%ph)+'_lambda'+str('%.4f'%la)+'_T'+str('%.4f'%te) 
batch_params       = lambda nL, la, ph, te: 'x'+str(nL)+'_phi0'+str('%.4f'%ph)+'_lambda'+str('%.4f'%la)+'_T'+str('%.4f'%te) 

free_sim_location  = lambda nL, la, ph, te, sim: root_dir + free_batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields.dat'
sim_location       = lambda nL, la, ph, te, sim: root_dir + batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields.dat'
clean_sim_location = lambda nL, la, ph, te, sim: root_dir + 'clean_' + batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields'
bubble_at_rest     = lambda nL, la, ph, te, sim: root_dir + 'rest_bubble_' + batch_params(nL,la,ph,te) + '_sim'+str(sim)+'_fields'

directions_bubbles_file  = lambda nL, la, ph, te: root_dir + 'directions_' + batch_params(nL,la,ph,te)
velocities_bubbles_file  = lambda nL, la, ph, te: root_dir + 'velocitiesCOM_' + batch_params(nL,la,ph,te)
average_bubble_file      = lambda nL, la, ph, te: root_dir + 'average_bubble_' + batch_params(nL,la,ph,te)

triage_pref = lambda minS, maxS, nTM: '_minSim'+str(minS)+'_maxSim'+str(maxS)+'_up_to_nTMax'+str(nTM)

path_nodecay_sims  = lambda nL, la, ph, te, minS, maxS, nTM: root_dir+'sims_no_decay' + triage_pref(minS,maxS,nTM) + batch_params(nL,la,ph,te)
path_decayed_sims  = lambda nL, la, ph, te, minS, maxS, nTM: root_dir+'sims_good_decay' + triage_pref(minS,maxS,nTM) + batch_params(nL,la,ph,te)

titles = [r'$\phi(x)$', r'$\partial_t \phi(x)$', r'$|\nabla \phi(x)|^2$', r'$V(\phi(x))$']

# Important: standard order or columns in .dat files is:
# field, momentum, gradient field squared
normal = np.asarray([phieq, 0., 0.])
