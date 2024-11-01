import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

simulation_data = pd.read_csv('/Users/rodrigosanchez/Documents/Thesis /PINGpy/simulation_results.csv')

contrast = np.linspace(0, 100, 100)

simulation_results = {}

for index, row in simulation_data.iterrows():
    MF = row['Intercept']
    MI = row['Slope']
    current_values = [(25 - MF + 0.25 * c) / MI for c in contrast]
    simulation_results[f"Simulation_{int(row['Simulation'])}"] = current_values
    
    plt.figure(figsize=(10 ,6))
    for sim, current_values in simulation_results.items():
        plt.plot(contrast, current_values, label=sim)
        
plt.xlabel("Contrast (c)")
plt.ylabel("Current (I)")
plt.title("Current as a Function of Contrast for Each Simulation")
plt.legend()
plt.grid(True)
plt.show()

results_df = pd.DataFrame(simulation_results)
results_df.insert(0, "Contrast", contrast)  
results_df.to_csv("calculated_contrast_current_results.csv", index=False)