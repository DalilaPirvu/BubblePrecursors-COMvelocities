from bubble_tools import *
from experiment import *

tmp      = 0
temp     = Tlist[tmp]
ph0      = phi0List[tmp]
sigmafld = fluct_stdev(m2eff, ph0, temp)
print(temp, ph0, m2eff, sigmafld, nTimeMAX)

get_energy = True
get_ps = True

pspec_path = './data/powspec_temp'+str('%.4f'%temp)+'_phi0'+str('%.4f'%ph0)
en_path = './data/energy_temp'+str('%.4f'%temp)+'_phi0'+str('%.4f'%ph0)

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
            real = real[:, :-nLat, :]

            nC, nT, nN = np.shape(real)
            if get_energy:
                energy = 0.5*real[1]**2. + 0.5*real[2]**2. + V(real[0])
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
