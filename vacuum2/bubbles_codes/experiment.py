from plotting import *
from bubble_tools import *

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

### Params
nLat     = 2048
nTimeMAX = 2*nLat
bytesUNDECAYED = 662700352

phi0List = np.array([2.*np.pi/3.2, 2.*np.pi/3.2])
lambList = np.array([1.5, 1.5])

#### CONSTANTS
nu     = 2e-3
lenLat = 50./(2.*nu)**0.5
temp   = 0.
phieq  = np.pi
alph   = 32.

dx     = lenLat/nLat
dk     = 2.*np.pi/lenLat
knyq   = nLat//2+1

dt     = dx/alph
dtout  = dt*alph
lightc = dx/dtout

# this is in mass units
dx2plot = np.sqrt(4*nu)*dx
dt2plot = np.sqrt(4*nu)*dtout

# Lattice
lattice = np.arange(nLat)
xlist   = lattice*dx
klist   = np.roll((lattice - nLat//2)*dk, nLat//2)

inv_phases = np.exp(1j*np.outer(xlist, klist))
dit_phases = nLat**-1. * np.exp(-1j*np.outer(xlist, klist))

#### SPECTRA
# Free field (constant mass term) field modes \phi_k
m2   = lambda la: 4.*nu*(-1.+la**2.)
norm = lambda ph0: 1./ ph0 / np.sqrt(2. * lenLat)
w2   = lambda la: klist**2. + m2(la)

free_eigenbasis  = lambda la, ph0: np.asarray([norm(ph0)/(w2(la)[k]**0.25) if kk!=0. else 0. for k,kk in enumerate(klist)])
free_pspec       = lambda la, ph0: np.abs(free_eigenbasis(la, ph0))**2.
free_fluct_stdev = lambda la, ph0: np.sqrt(np.sum(free_pspec(la, ph0)))

thermal_eigenbasis  = lambda la, ph0, te: free_eigenbasis(la, ph0) * np.sqrt(2./(np.exp(w2(la)**0.5/te)-1.))
thermal_pspec       = lambda la, ph0, te: np.abs(thermal_eigenbasis(la, ph0, te))**2.
thermal_fluct_stdev = lambda la, ph0, te: np.sqrt(np.sum(thermal_pspec(la, ph0, te)))

pspec       = lambda la, ph0, te: thermal_pspec(la, ph0, te) if te!=0 else free_pspec(la, ph0)
fluct_stdev = lambda la, ph0, te: thermal_fluct_stdev(la, ph0, te) if te!=0 else free_fluct_stdev(la, ph0)

### POTENTIAL
V    = lambda x, la:  ( -np.cos(x) + 0.5 * la**2. * np.sin(   x)**2. + 1. ) * 4.*nu
Vinv = lambda x, la: -( -np.cos(x) + 0.5 * la**2. * np.sin(   x)**2. + 1. ) * 4.*nu
dV   = lambda x, la:  (  np.sin(x) + 0.5 * la**2. * np.sin(2.*x)          ) * 4.*nu

### Paths to files
root_dir      = '/gpfs/dpirvu/vacuum_potential/'
batch_params  = lambda nL,la,ph,te: 'x'+str(int(nL))+'_phi0'+str('%.4f'%ph)+'_lambda'+str('%.4f'%la)+'_T'+str('%.4f'%te) 
triage_pref   = lambda minS,maxS,nTM: '_minSim'+str(minS)+'_maxSim'+str(maxS)+'_up_to_nTMax'+str(nTM)
sims_interval = lambda minS,maxS: '_minSim'+str(minS)+'_maxSim'+str(maxS)

sim_location       = lambda nL,la,ph,te,sim: root_dir + batch_params(nL,la,ph,te) + '_sim'+str(sim)       + '_fields.dat'
clean_sim_location = lambda nL,la,ph,te,sim: root_dir + batch_params(nL,la,ph,te) + '_clean_sim'+str(sim) + '_fields.npy'
rest_sim_location  = lambda nL,la,ph,te,sim: root_dir + batch_params(nL,la,ph,te) + '_rest_sim'+str(sim)  + '_fields.npy'

directions_file = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_directions.npy'
velocities_file = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_velocitiesCOM.npy'
average_file    = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_average_bubble.npy'

sims_decayed_file    = lambda nL,la,ph,te,minS,maxS,nTM: root_dir + batch_params(nL,la,ph,te) + triage_pref(minS,maxS,nTM) + '_sims_decayed.npy'
sims_notdecayed_file = lambda nL,la,ph,te,minS,maxS,nTM: root_dir + batch_params(nL,la,ph,te) + triage_pref(minS,maxS,nTM) + '_sims_notdecayed.npy'
decay_times_file     = lambda nL,la,ph,te,minS,maxS,nTM: root_dir + batch_params(nL,la,ph,te) + triage_pref(minS,maxS,nTM) + '_timedecays.npy'

powspec_tlist_file = lambda nL,la,ph,te,minS,maxS: root_dir + batch_params(nL,la,ph,te) + sims_interval(minS,maxS) + '_powspec.npy'
emt_tlist_file     = lambda nL,la,ph,te,minS,maxS: root_dir + batch_params(nL,la,ph,te) + sims_interval(minS,maxS) + '_emt.npy'
stdemt0_tlist_file = lambda nL,la,ph,te,minS,maxS: root_dir + batch_params(nL,la,ph,te) + sims_interval(minS,maxS) + '_stdemt0.npy'
toten_tlist_file   = lambda nL,la,ph,te,minS,maxS: root_dir + batch_params(nL,la,ph,te) + sims_interval(minS,maxS) + '_toten.npy'
stdinit_file       = lambda nL,la,ph,te,minS,maxS: root_dir + batch_params(nL,la,ph,te) + sims_interval(minS,maxS) + '_fld_init_std.npy'

supercrit_instanton_sim_file = lambda nL,la,ph: root_dir + 'supercrit_instanton_x'+str(int(nL))+'_phi0'+str('%.4f'%ph)+'_lambda'+str('%.4f'%la) + '_fields.dat'
instanton_sim_file       = lambda nL,la,ph: root_dir + 'instanton_x'+str(int(nL))+'_phi0'+str('%.4f'%ph)+'_lambda'+str('%.4f'%la) + '_fields.dat'
critical_sim_file        = lambda nL,la,ph,te: root_dir + 'critical_' + batch_params(nL,la,ph,te) + '_sim'+str(0) + '_fields.dat'
precursor_sim_file       = lambda nL,la,ph,te: root_dir + 'precursor_' + batch_params(nL,la,ph,te) + '_sim'+str(0) + '_fields.dat'
supercrit_instanton_file = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_supercrit_critical_field_profile.npy'
instanton_file           = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_critical_field_profile.npy'

critfield_file = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_critical_average_field_profile.npy'
crittimes_file = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_critical_timeslice.npy'
critenerg_file = lambda nL,la,ph,te: root_dir + batch_params(nL,la,ph,te) + '_critical_energy.npy'

titles = [r'$\phi(x)$', r'$\partial_t \phi(x)$', r'$|\nabla \phi(x)|^2$', r'$V(\phi(x))$']

labl = lambda la,ph,te: r'$\phi_0=$'+str('%.2f'%ph)+r', $\lambda=$'+str(la)+r', $T=$'+str(te)

def get_model(tmp):
    if tmp == 0: minSim, maxSim = 0, 1000
    elif tmp == 1: minSim, maxSim = 0, 1000
    else:
        return 'Experiment not implemented.'

    phi0 = phi0List[tmp]
    lamb = lambList[tmp]
    sigmafld = fluct_stdev(lamb, phi0, temp)
    right_Vmax = sco.minimize_scalar(Vinv, args=lamb, bounds=(np.pi, 2*np.pi), method='bounded')

    # Important: standard order or columns in .dat files is:
    # field, momentum, momentum squared, gradient, gradient squared, potential
    normal = np.array([phieq, 0., 0., 0., 0., V(phieq, lamb)])
    return phi0, lamb, sigmafld, minSim, maxSim, right_Vmax, normal
