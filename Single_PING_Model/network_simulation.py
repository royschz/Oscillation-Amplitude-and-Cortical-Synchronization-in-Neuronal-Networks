import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import time

class NetworkSimulation:
    def __init__(self, input_net1=20, Ne=200, Ni=50, simulation_time=3000, dt=1):
        self.input_net1 = input_net1
        self.Ne = Ne
        self.Ni = Ni
        self.simulation_time = simulation_time
        self.dt = dt
        self.Ntot = Ne + Ni
        self.peak_freq_idx = None

        # Define neuronal parameters
        self.a = np.concatenate((0.02 * np.ones(Ne), 0.1 * np.ones(Ni)))
        self.b = np.concatenate((0.2 * np.ones(Ne), 0.2 * np.ones(Ni)))
        self.c = np.concatenate((-65 * np.ones(Ne), -65 * np.ones(Ni)))
        self.d = np.concatenate((8 * np.ones(Ne), 2 * np.ones(Ni)))

        # Initial values
        self.v = np.concatenate(-65 * np.ones((Ne + Ni, 1)))  # membrane voltage
        self.u = self.b * self.v  # membrane recovery variable
        self.firings = []  # spike timings

        # Input strength - Gaussian input
        self.mean_E = input_net1 * np.ones(Ne)
        self.var_E = 3
        self.mean_I = 4 * np.ones(Ni)
        self.var_I = 3.5

        # Synaptic conductances
        self.gampa = np.zeros(Ne)
        self.gaba = np.zeros(Ni)
        self.decay_ampa = 1
        self.decay_gaba = 7
        self.rise_ampa = 0.15
        self.rise_gaba = 0.2

        # Connectivity matrix
        self.S = self._initialize_connectivity_matrix()

    def _initialize_connectivity_matrix(self):
        EE = 0.05  # excitatory to excitatory
        EI = 0.4  # excitatory to inhibitory
        IE = 0.3  # inhibitory to excitatory
        II = 0.2  # inhibitory to inhibitory

        S = np.zeros((self.Ntot, self.Ntot))

        # Within network connectivity
        E1_ind = np.arange(self.Ne)
        I1_ind = np.arange(self.Ne, self.Ntot)

        # E - E
        S[np.ix_(E1_ind, E1_ind)] = EE * np.random.rand(self.Ne, self.Ne)
        # E - I
        S[np.ix_(I1_ind, E1_ind)] = EI * np.random.rand(self.Ni, self.Ne)
        # I - E
        S[np.ix_(E1_ind, I1_ind)] = -IE * np.random.rand(self.Ne, self.Ni)
        # I - I
        S[np.ix_(I1_ind, I1_ind)] = -II * np.random.rand(self.Ni, self.Ni)

        return S

    def run_simulation(self):
        start_time = time.time()

        for t in range(0, self.simulation_time, self.dt):
            I = np.concatenate((
                self.var_E * np.random.randn(self.Ne) + self.mean_E,
                self.var_I * np.random.randn(self.Ni) + self.mean_I
            ))  # thalamic input

            fired = np.where(self.v >= 30)[0]  # indices of spikes
            if len(fired) > 0:
                self.firings.extend([(t, neuron) for neuron in fired])
            self.v[fired] = self.c[fired]
            self.u[fired] += self.d[fired]

            self.gampa += self.dt * (0.3 * (((1 + np.tanh(self.v[:self.Ne] / 10 + 2)) / 2) * (1 - self.gampa) / self.rise_ampa - self.gampa / self.decay_ampa))
            self.gaba += self.dt * (0.3 * (((1 + np.tanh(self.v[self.Ne:] / 10 + 2)) / 2) * (1 - self.gaba) / self.rise_gaba - self.gaba / self.decay_gaba))

            I += self.S @ np.concatenate((self.gampa, self.gaba))  # integrate input from other neurons
            self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)  # step 0.5 ms
            self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)  # for numerical stability
            self.u += self.a * (self.b * self.v - self.u)

        end_time = time.time()
        print(f"Simulation time: {end_time - start_time} seconds")

        self.firings = np.array(self.firings)

    def get_peak_freq(self):
       
        excpop1 = np.where((self.firings[:, 1] >= 0) & (self.firings[:, 1] < self.Ne))[0]
        t1, _ = np.histogram(self.firings[excpop1, 0], bins=np.arange(0, self.simulation_time + 1))
        signal1 = t1[300:]

        f, t, Sxx = spectrogram(signal1 - np.mean(signal1), fs=1000/self.dt, nperseg=256, noverlap=250, nfft=256, scaling='spectrum')
        mean_power = np.nanmean(Sxx, axis=1)
        self.peak_freq_idx = np.argmax(mean_power)  # Almacenar el índice de la frecuencia pico

        return f[self.peak_freq_idx]  

    def plot_results(self):
        excpop1 = np.where((self.firings[:, 1] >= 0) & (self.firings[:, 1] < self.Ne))[0]
        inhpop1 = np.where((self.firings[:, 1] >= self.Ne) & (self.firings[:, 1] < self.Ntot))[0]

        # Population signal
        t1, t2 = np.histogram(self.firings[excpop1, 0], bins=np.arange(0, self.simulation_time + 1))
        signal1 = t1[300:]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Network input: {self.input_net1}')

        plt.subplot(2, 1, 1)
        firing1 = self.firings[self.firings[:, 1] < self.Ne]
        firing2 = self.firings[(self.firings[:, 1] >= self.Ne) & (self.firings[:, 1] < self.Ntot)]
        plt.plot(firing1[:, 0], firing1[:, 1], '.', color=[0.8, 0.2, 0.2])
        plt.plot(firing2[:, 0], firing2[:, 1], '.', color=[0.2, 0.2, 0.8])
        plt.title('Spike Raster')
        plt.xlim([700, 1500])
        plt.ylabel('Neuron N')
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        f, t, Sxx = spectrogram(signal1 - np.mean(signal1), fs=1000/self.dt, nperseg=256, noverlap=250, nfft=256, scaling='spectrum')

        # Use a colormap similar to MATLAB's
        plt.imshow(Sxx, aspect='auto', extent=[t.min(), t.max(), f.min(), f.max()], origin='lower', cmap='jet')
        plt.ylim(20, 60)
        plt.title('TFR')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power')

        # Highlight peak frequency
        mean_power = np.nanmean(Sxx, axis=1)
        peak_freq_idx = np.argmax(mean_power)
        plt.text(t.max(), f[peak_freq_idx], f' \u2190 {f[peak_freq_idx]:.1f} Hz', fontsize=12, fontweight='bold')

        plt.show()

if __name__ == "__main__":
    sim = NetworkSimulation()
    sim.run_simulation()
    sim.plot_results()