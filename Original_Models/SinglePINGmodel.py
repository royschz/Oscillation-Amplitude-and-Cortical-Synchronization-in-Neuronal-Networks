import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import time

# Parameters
input_net1 = 3  # excitatory input to network 1

# Number of excitatory and inhibitory neurons
Ne = 200
Ni = 50

# Define neuronal parameters
a = np.concatenate((0.02 * np.ones(Ne), 0.1 * np.ones(Ni)))
b = np.concatenate((0.2 * np.ones(Ne), 0.2 * np.ones(Ni)))
c = np.concatenate((-65 * np.ones(Ne), -65 * np.ones(Ni)))
d = np.concatenate((8 * np.ones(Ne), 2 * np.ones(Ni)))

# Initial values
v = np.concatenate(-65 * np.ones((Ne + Ni, 1)) ) # membrane voltage
u = b * v  # membrane recovery variable
firings = []  # spike timings

simulation_time = 3000
dt = 1
Ntot = Ne + Ni

# Input strength - Gaussian input
mean_E = input_net1 * np.ones(Ne)
var_E = 3
mean_I = 4 * np.ones(Ni)
var_I = 3.5

# Synaptic conductances
gampa = np.zeros(Ne)
gaba = np.zeros(Ni)
decay_ampa = 1
decay_gaba = 7
rise_ampa = 0.15
rise_gaba = 0.2

### Connectivity matrix ###
EE = 0.05  # excitatory to excitatory
EI = 0.4  # excitatory to inhibitory
IE = 0.3  # inhibitory to excitatory
II = 0.2  # inhibitory to inhibitory

S = np.zeros((Ntot, Ntot))

# # Within network connectivity
E1_ind = np.arange(Ne)
I1_ind = np.arange(Ne, Ntot)



# E - E
S[np.ix_(E1_ind, E1_ind)] = EE * np.random.rand(Ne, Ne)
# E - I
S[np.ix_(I1_ind, E1_ind)] = EI * np.random.rand(Ni, Ne)
# I - E
S[np.ix_(E1_ind, I1_ind)] = -IE * np.random.rand(Ne, Ni)
# I - I
S[np.ix_(I1_ind, I1_ind)] = -II * np.random.rand(Ni, Ni)

start_time = time.time()



# Simulation
for t in range(0, simulation_time, dt):
    I = np.concatenate((var_E * np.random.randn(Ne) + mean_E, 
                        var_I * np.random.randn(Ni) + mean_I))  # thalamic input
    fired = np.where(v >= 30)[0]  # indices of spikes
    if len(fired) > 0:
        firings.extend([(t, neuron) for neuron in fired])
    v[fired] = c[fired]
    u[fired] += d[fired]

    gampa += dt * (0.3 * (((1 + np.tanh(v[:Ne] / 10 + 2)) / 2) * (1 - gampa) / rise_ampa - gampa / decay_ampa))
    gaba += dt * (0.3 * (((1 + np.tanh(v[Ne:] / 10 + 2)) / 2) * (1 - gaba) / rise_gaba - gaba / decay_gaba))

    I += S @ np.concatenate((gampa, gaba))  # integrate input from other neurons
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)  # step 0.5 ms
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)  # for numerical stability
    u += a * (b * v - u)

end_time = time.time()
print(f"Simulation time: {end_time - start_time} seconds")

# Spike probability for given time bin
firings = np.array(firings)
excpop1 = np.where((firings[:, 1] >= 0) & (firings[:, 1] < Ne))[0]
inhpop1 = np.where((firings[:, 1] >= Ne) & (firings[:, 1] < Ntot))[0]

# Population signal
t1, t2 = np.histogram(firings[excpop1, 0], bins=np.arange(0, t + 1))
signal1 = t1[300:]

# Plotting
plt.figure(figsize=(12, 6))
plt.suptitle(f'Network input: {input_net1}')

plt.subplot(2, 1, 1)
firing1 = firings[firings[:, 1] < Ne]
firing2 = firings[(firings[:, 1] >= Ne) & (firings[:, 1] < Ntot)]
plt.plot(firing1[:, 0], firing1[:, 1], '.', color=[0.8, 0.2, 0.2])
plt.plot(firing2[:, 0], firing2[:, 1], '.', color=[0.2, 0.2, 0.8])
plt.title('Spike Raster')
plt.xlim([700, 1500])
plt.ylabel('Neuron N')
plt.tight_layout()

plt.subplot(2, 1, 2)
f, t, Sxx = spectrogram(signal1 - np.mean(signal1), fs=1000/dt, nperseg=256, noverlap=250, nfft=256, scaling='spectrum')

# Use a colormap similar to MATLAB's
plt.imshow(Sxx, aspect='auto', extent=[t.min(), t.max(), f.min(), f.max()], origin='lower', cmap='jet')
plt.ylim(20, 150)
#plt.colorbar(label='Power')
plt.title('TFR')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')


# Highlight peak frequency
mean_power = np.nanmean(Sxx, axis=1)
peak_freq_idx = np.argmax(mean_power)
plt.text(t.max(), f[peak_freq_idx], f' \u2190 {f[peak_freq_idx]:.1f} Hz', fontsize=12, fontweight='bold')

plt.show()
