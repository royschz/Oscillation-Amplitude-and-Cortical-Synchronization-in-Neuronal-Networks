import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import time

class NeuronNetworkSimulation:
    def __init__(self, input_net1=9, input_net2=12, cop_val=0.1, simulation_time=10000, dt=1):
        self.input_net1 = input_net1
        self.input_net2 = input_net2
        self.cop_val = cop_val
        self.simulation_time = simulation_time
        self.dt = dt

        # Number of excitatory and inhibitory neurons
        self.Ne1, self.Ne2 = 200, 200
        self.Ni1, self.Ni2 = 50, 50
        self.Ne, self.Ni = self.Ne1 + self.Ne2, self.Ni1 + self.Ni2
        self.Ntot = self.Ne + self.Ni

        # Initialize parameters a, b, c, d
        self.a = np.concatenate([0.02 * np.ones(self.Ne), 0.1 * np.ones(self.Ni)])
        self.b = np.concatenate([0.2 * np.ones(self.Ne), 0.2 * np.ones(self.Ni)])
        self.c = np.concatenate([-65 * np.ones(self.Ne), -65 * np.ones(self.Ni)])
        self.d = np.concatenate([8 * np.ones(self.Ne), 2 * np.ones(self.Ni)])

        # Initial values
        self.v = -65 * np.ones(self.Ntot)
        self.u = self.b * self.v
        self.firings = []

        # Input strength
        self.mean_E = np.concatenate([self.input_net1 * np.ones(self.Ne1), self.input_net2 * np.ones(self.Ne2)])
        self.mean_I = np.concatenate([4 * np.ones(self.Ni1), 4 * np.ones(self.Ni2)])
        self.var_E = self.var_I = 3

        # Synaptic conductances
        self.gampa = np.zeros(self.Ne)
        self.gaba = np.zeros(self.Ni)
        self.decay_ampa, self.decay_gaba = 1, 7
        self.rise_ampa, self.rise_gaba = 0.15, 0.2

        # Connectivity matrix
        self.S = np.zeros((self.Ntot, self.Ntot))
        self.create_connectivity()

    def create_connectivity(self):
        """Sets up within and between network connectivity."""
        EE, EI, IE, II = 0.05, 0.4, 0.3, 0.2
        EE2, EI2, IE2, II2 = 0.05, 0.4, 0.3, 0.2
        C1 = C2 = C3 = C4 = self.cop_val / 4

        E1_ind = np.arange(self.Ne1)
        E2_ind = np.arange(self.Ne1, self.Ne1 + self.Ne2)
        I1_ind = np.arange(self.Ne, self.Ne + self.Ni1)
        I2_ind = np.arange(self.Ne + self.Ni1, self.Ne + self.Ni)

        # Within network connectivity
        self.S[np.ix_(E1_ind, E1_ind)] = EE * np.random.rand(self.Ne1, self.Ne1)
        self.S[np.ix_(E2_ind, E2_ind)] = EE2 * np.random.rand(self.Ne2, self.Ne2)
        self.S[np.ix_(I1_ind, E1_ind)] = EI * np.random.rand(self.Ni1, self.Ne1)
        self.S[np.ix_(I2_ind, E2_ind)] = EI2 * np.random.rand(self.Ni2, self.Ne2)
        self.S[np.ix_(E1_ind, I1_ind)] = -IE * np.random.rand(self.Ne1, self.Ni1)
        self.S[np.ix_(E2_ind, I2_ind)] = -IE2 * np.random.rand(self.Ne2, self.Ni2)
        self.S[np.ix_(I1_ind, I1_ind)] = -II * np.random.rand(self.Ni1, self.Ni1)
        self.S[np.ix_(I2_ind, I2_ind)] = -II2 * np.random.rand(self.Ni2, self.Ni2)

        # Between network connectivity
        self.S[np.ix_(I2_ind, E1_ind)] = C2 * np.random.rand(self.Ni2, self.Ne1)
        self.S[np.ix_(I1_ind, E2_ind)] = C1 * np.random.rand(self.Ni1, self.Ne2)
        self.S[np.ix_(E2_ind, E1_ind)] = C3 * np.random.rand(self.Ne2, self.Ne1)
        self.S[np.ix_(E1_ind, E2_ind)] = C4 * np.random.rand(self.Ne1, self.Ne2)

    def run_simulation(self):
        """Runs the network simulation."""
        start_time = time.time()
        for t in range(0, self.simulation_time, self.dt):
            I = np.concatenate([self.var_E * np.random.randn(self.Ne) + self.mean_E,
                                self.var_I * np.random.randn(self.Ni) + self.mean_I])
            fired = np.where(self.v >= 30)[0]
            self.firings.extend([(t, neuron) for neuron in fired])
            self.v[fired] = self.c[fired]
            self.u[fired] += self.d[fired]

            self.gampa += self.dt * (0.3 * ((1 + np.tanh(self.v[:self.Ne] / 10 + 2)) / 2 * (1 - self.gampa) / self.rise_ampa
                                            - self.gampa / self.decay_ampa))
            self.gaba += self.dt * (0.3 * ((1 + np.tanh(self.v[self.Ne:] / 10 + 2)) / 2 * (1 - self.gaba) / self.rise_gaba
                                           - self.gaba / self.decay_gaba))

            I += self.S @ np.concatenate([self.gampa, self.gaba])
            self.v += 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)
            self.v += 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)
            self.u += self.a * (self.b * self.v - self.u)

        end_time = time.time()
        print(f'Simulation finished in {end_time - start_time:.2f} seconds')

    def analyze_results(self):
        """Analyzes the simulation results: spike probability and phase relation."""
        self.firings = np.array(self.firings)
        excpop1 = np.where((self.firings[:, 1] > 0) & (self.firings[:, 1] <= self.Ne1))[0]
        excpop2 = np.where((self.firings[:, 1] > self.Ne1) & (self.firings[:, 1] <= self.Ne))[0]

        signal1, _ = np.histogram(self.firings[excpop1, 0], bins=np.arange(0, self.simulation_time + 1))
        signal2, _ = np.histogram(self.firings[excpop2, 0], bins=np.arange(0, self.simulation_time + 1))

        Fn = 500
        Fbp = [25, 55]
        B, A = butter(4, [min(Fbp) / Fn, max(Fbp) / Fn], btype='bandpass')
        signal1 = filtfilt(B, A, signal1[300:])
        signal2 = filtfilt(B, A, signal2[300:])

        p1 = np.angle(hilbert(signal1))
        p2 = np.angle(hilbert(signal2))
        px = np.angle(np.exp(1j * p1) / np.exp(1j * p2))

        return signal1, signal2, px

    def plot_results(self, signal1, signal2, px):
        """Plots the spike raster and phase relation results."""
        plt.figure(figsize=(10, 8))
        # Phase-relation histogram
        n, bins, _ = plt.hist(px, bins=20, density=True, color='black', edgecolor=[0.3, 0.3, 0.3])
        plt.xlim([-np.pi, np.pi])
        plt.xlabel('Phase-relation')
        plt.ylabel('Probability')
        plt.title(f'PLV = {np.abs(np.mean(np.exp(1j * px))):.2f}')
        plt.tight_layout()
        plt.show()
        
        

