import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import time

# Task 1-5 initializations
input_net1 = 9  # excitatory input to network 1
input_net2 = 12  # excitatory input to network 2
cop_val = 0.1  # cross-network connection strength

# Number of excitatory and inhibitory neurons
Ne1, Ne2 = 200, 200
Ni1, Ni2 = 50, 50
Ne, Ni = Ne1 + Ne2, Ni1 + Ni2
Ntot = Ne + Ni

# Parameters a, b, c, d
a = np.concatenate([0.02 * np.ones(Ne), 0.1 * np.ones(Ni)])
b = np.concatenate([0.2 * np.ones(Ne), 0.2 * np.ones(Ni)])
c = np.concatenate([-65 * np.ones(Ne), -65 * np.ones(Ni)])
d = np.concatenate([8 * np.ones(Ne), 2 * np.ones(Ni)])

# Initial values
v = -65 * np.ones(Ntot)
u = b * v
firings = []
simulation_time = 10000
dt = 1

# Input strength
mean_E = np.concatenate([input_net1 * np.ones(Ne1), input_net2 * np.ones(Ne2)])
mean_I = np.concatenate([4 * np.ones(Ni1), 4 * np.ones(Ni2)])
var_E = var_I = 3

gampa = np.zeros(Ne)
gaba = np.zeros(Ni)
decay_ampa, decay_gaba = 1, 7
rise_ampa, rise_gaba = 0.15, 0.2

# Connectivity matrix
EE, EI, IE, II = 0.05, 0.4, 0.3, 0.2
EE2, EI2, IE2, II2 = 0.05, 0.4, 0.3, 0.2
C1 = C2 = C3 = C4 = cop_val / 4

S = np.zeros((Ntot, Ntot))

# Indices for excitatory/inhibitory neurons
E1_ind = np.arange(Ne1)
E2_ind = np.arange(Ne1, Ne1 + Ne2)
I1_ind = np.arange(Ne, Ne + Ni1)
I2_ind = np.arange(Ne + Ni1, Ne + Ni)

# Within network connectivity
S[np.ix_(E1_ind, E1_ind)] = EE * np.random.rand(Ne1, Ne1)
S[np.ix_(E2_ind, E2_ind)] = EE2 * np.random.rand(Ne2, Ne2)
S[np.ix_(I1_ind, E1_ind)] = EI * np.random.rand(Ni1, Ne1)
S[np.ix_(I2_ind, E2_ind)] = EI2 * np.random.rand(Ni2, Ne2)
S[np.ix_(E1_ind, I1_ind)] = -IE * np.random.rand(Ne1, Ni1)
S[np.ix_(E2_ind, I2_ind)] = -IE2 * np.random.rand(Ne2, Ni2)
S[np.ix_(I1_ind, I1_ind)] = -II * np.random.rand(Ni1, Ni1)
S[np.ix_(I2_ind, I2_ind)] = -II2 * np.random.rand(Ni2, Ni2)

# Between network connectivity
S[np.ix_(I2_ind, E1_ind)] = C2 * np.random.rand(Ni2, Ne1)
S[np.ix_(I1_ind, E2_ind)] = C1 * np.random.rand(Ni1, Ne2)
S[np.ix_(E2_ind, E1_ind)] = C3 * np.random.rand(Ne2, Ne1)
S[np.ix_(E1_ind, E2_ind)] = C4 * np.random.rand(Ne1, Ne2)

# Simulation loop
start_time = time.time()
for t in range(0, simulation_time, dt):
    I = np.concatenate([var_E * np.random.randn(Ne) + mean_E, var_I * np.random.randn(Ni) + mean_I])
    fired = np.where(v >= 30)[0]
    firings.extend([(t, neuron) for neuron in fired])
    v[fired] = c[fired]
    u[fired] += d[fired]
    
    gampa += dt * (0.3 * ((1 + np.tanh(v[:Ne] / 10 + 2)) / 2 * (1 - gampa) / rise_ampa - gampa / decay_ampa))
    gaba += dt * (0.3 * ((1 + np.tanh(v[Ne:] / 10 + 2)) / 2 * (1 - gaba) / rise_gaba - gaba / decay_gaba))
    
    I += S @ np.concatenate([gampa, gaba])
    v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)
    v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)
    u += a * (b * v - u)

end_time = time.time()
print(f'Simulation finished in {end_time - start_time:.2f} seconds')

# Spike probability and phase relation
firings = np.array(firings)
excpop1 = np.where((firings[:, 1] > 0) & (firings[:, 1] <= Ne1))[0]
excpop2 = np.where((firings[:, 1] > Ne1) & (firings[:, 1] <= Ne))[0]
inhpop1 = np.where((firings[:, 1] > Ne) & (firings[:, 1] <= Ne + Ni1))[0]
inhpop2 = np.where((firings[:, 1] > Ne + Ni1) & (firings[:, 1] <= Ne + Ni))[0]

# Population signal
signal1, _ = np.histogram(firings[excpop1, 0], bins=np.arange(0, simulation_time + 1))
signal2, _ = np.histogram(firings[excpop2, 0], bins=np.arange(0, simulation_time + 1))

# Filtering signals
Fn = 500
Fbp = [25, 55]
B, A = butter(4, [min(Fbp) / Fn, max(Fbp) / Fn], btype='bandpass')
signal1 = filtfilt(B, A, signal1[300:])
signal2 = filtfilt(B, A, signal2[300:])

# Phase relation
p1 = np.angle(hilbert(signal1))
p2 = np.angle(hilbert(signal2))
px = np.angle(np.exp(1j * p1) / np.exp(1j * p2))

# PLOTTING %%%%%%%%%%%
plt.figure(figsize=(10, 8))


# Phase-relation histogram
plt.subplot(2, 2, 2)
n, bins, _ = plt.hist(px, bins=20, density=True, color='black', edgecolor=[0.3, 0.3, 0.3])
plt.xlim([-np.pi, np.pi])
plt.xlabel('Phase-relation')
plt.ylabel('Probability')
plt.title(f'PLV = {np.abs(np.mean(np.exp(1j * px))):.2f}')
plt.show()


