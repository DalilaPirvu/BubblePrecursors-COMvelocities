from bubble_tools_vac import *
from experiment_vac import *

tmp      = 1
ph0      = phi0List[tmp]
lamb     = lambList[tmp]
sigmafld = fluct_stdev(lamb, ph0, temp)

### Useful
right_Vmax  = sco.minimize_scalar(V, args=lamb, bounds=(np.pi, 2*np.pi), method='bounded')
left_Vmax   = sco.minimize_scalar(V, args=lamb, bounds=(0    ,   np.pi), method='bounded')
amp_thresh    = right_Vmax.x
crit_thresh   = right_Vmax.x
tv_thresh     = right_Vmax.x
crit_rad      = 80

print('Looking at at lambda, T, phi0, m2, sigma:', lamb, temp, ph0, m2(lamb), sigmafld)

get_energy = True
get_ps = True

pspec_path = './data/powspec_lamb'+str('%.4f'%lamb)+'_phi0'+str('%.4f'%ph0)
en_path = './data/energy_lamb'+str('%.4f'%lamb)+'_phi0'+str('%.4f'%ph0)

tlist = [0]+np.linspace(1, nTimeMAX-1, 30).tolist()
tlist = np.asarray(tlist, dtype = 'int')
print(tlist)

minSim=0
maxSim=25

if get_energy or get_ps:
    ps_data, energy_data = [], []
    for sim in range(minSim, maxSim):
        path_sim = sim_location(nLat, lamb, ph0, temp, sim)
        if os.path.exists(path_sim):
            real, _ = get_realisation(nLat, sim, path_sim, 2, phieq)
            print(sim, np.shape(real))
            real = real[:, :-3*nLat//4, :]

            nC, nT, nN = np.shape(real)
            if get_energy:
                energy = 0.5*real[1]**2. + 0.5*real[2]**2. + V(real[0],lamb)
                energy_data.append(np.mean(energy, axis=1))
            if get_ps:
                ps_data.append([])
                for tt in tlist:
                    try:
                        slice = real[0,tt,:]
                        fftslice = np.abs(np.fft.fft(slice/nLat))**2.
                        fftslice[0] = 0.
                        ps_data[-1].append(fftslice)
                    except:
                        break

    if get_ps:
        np.save(pspec_path+'_minsim'+str(minSim)+'_maxSim'+str(maxSim), ps_data)
    if get_energy:
        np.save(en_path+'_minsim'+str(minSim)+'_maxSim'+str(maxSim), energy_data)
