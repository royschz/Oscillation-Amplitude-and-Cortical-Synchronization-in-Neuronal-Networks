from network_simulation import NetworkSimulation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import inv

def run_single_simulation():
    currents = np.linspace(5, 25, 50)
    frequencies = np.zeros((5, 50))
    
    for reps in range(5):
        for i, current in enumerate(currents):
            sim = NetworkSimulation(input_net1=current, Ne=200, Ni=50, simulation_time=3000, dt=1)
            sim.run_simulation()
            frequencies[reps, i] = sim.get_peak_freq()
    
    y = np.mean(frequencies, axis=0)
    
    X = np.ones((50, 2))
    X[:, 1] = currents
    
    b = inv((X.T).dot(X)).dot(X.T).dot(y)
    

    plt.plot(currents, y)
    plt.xlabel('Current')
    plt.ylabel('Frequencies')
    plt.title('Frequency vs Current')
    plt.show()
    
    return b  

def run_multiple_simulations(n_simulations=20):
    results_list = []
    
    for i in range(n_simulations):
        b = run_single_simulation()
        
        result = {
            'Simulation': i + 1,
            'Intercept': b[0],  
            'Slope': b[1]       
        }
        results_list.append(result)  
    
    results_b = pd.DataFrame(results_list)  
    return results_b

if __name__ == "__main__":
    simulation_results = run_multiple_simulations(20)
    
    print(simulation_results)
    simulation_results.to_csv('simulation_results.csv', index=False)

    